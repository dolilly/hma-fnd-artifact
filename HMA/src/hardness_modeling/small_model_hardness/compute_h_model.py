#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------
# HMA - Section 3.1.1: Small-model Uncertainty Hardness (h_model)
#
# Fine-tune a small pretrained model (e.g. RoBERTa) and use its
# predictive uncertainty as a hardness signal:
#     h_model = 1 - 2 * |p(true_label) - 0.5|
# Outputs y_pred_model / p_real / p_fake / h_model per row.
# Row-level safe, supports both csv and xlsx.
#
# Usage:
#   python compute_h_model.py --root . --dataset Constraint \
#       --model models/roberta-base
# ------------------------------------------------------------

import argparse
import random
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerBase,
)

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------
# Robust table loader
# -----------------------
def load_table(path: Path) -> pd.DataFrame:
    """
    Load csv or xlsx with robust encoding handling.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        for enc in ["utf-8", "gbk", "gb2312", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError(f"Cannot decode CSV file: {path}")

    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    else:
        raise ValueError(f"Unsupported file format: {path}")

# -----------------------
# Normalize columns
# -----------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # original id
    if "id" not in df.columns:
        df["id"] = np.arange(len(df)).astype(str)

    # row-level unique id
    df["_row_id"] = np.arange(len(df)).astype(int)

    if "text" not in df.columns:
        for cand in ["content", "tweet", "sentence", "raw_text", "title"]:
            if cand in df.columns:
                df.rename(columns={cand: "text"}, inplace=True)
                break
        else:
            raise ValueError("No text column found")

    if "label" not in df.columns:
        raise ValueError("No label column found")

    df["id"] = df["id"].astype(str)
    df["text"] = df["text"].astype(str).fillna("")

    # ---- SAFE label handling ----
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna()]
    df["label"] = df["label"].astype(int)

    return df.reset_index(drop=True)

# -----------------------
# Dataset
# -----------------------
class TextClsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_len,
        )
        enc["labels"] = row["label"]
        enc["_row_id"] = row["_row_id"]
        return enc

# -----------------------
# Safe collator
# -----------------------
class SafeDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features):
        row_ids = [f.pop("_row_id") for f in features]
        labels = torch.tensor([f.pop("labels") for f in features], dtype=torch.long)

        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels
        batch["_row_id"] = row_ids
        return batch

# -----------------------
# h_model
# -----------------------
def h_from_probs_and_labels(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    py = probs[np.arange(len(labels)), labels]
    return 1.0 - 2.0 * np.abs(py - 0.5)

# -----------------------
# Metrics
# -----------------------
@torch.no_grad()
def eval_loss_and_metrics(model, loader, device, loss_fn):
    model.eval()
    losses, y_true, y_pred = [], [], []

    for batch in loader:
        batch.pop("_row_id", None)
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = loss_fn(outputs.logits, labels)
        losses.append(loss.item())

        preds = outputs.logits.argmax(dim=-1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    fake_f1 = f1_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)

    return {
        "loss": float(np.mean(losses)),
        "acc": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf1,
        "fake_f1": fake_f1,
    }

# -----------------------
# Train one epoch
# -----------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn, scaler, use_amp):
    model.train()
    for batch in loader:
        batch.pop("_row_id", None)
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

# -----------------------
# Prediction helper
# -----------------------
@torch.no_grad()
def predict_df(model, loader, device):
    model.eval()
    rows = []

    for batch in loader:
        row_ids = batch.pop("_row_id")
        labels = batch.pop("labels").numpy()
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        h_vals = h_from_probs_and_labels(probs, labels)

        for i in range(len(row_ids)):
            rows.append({
                "_row_id": row_ids[i],
                "y_pred_model": preds[i],
                "p_real": probs[i, 0],
                "p_fake": probs[i, 1],
                "h_model": h_vals[i],
            })

    return pd.DataFrame(rows)

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    root = Path(args.root)
    data_dir = root / "data" / args.dataset

    # ---- flexible file names ----
    def pick_file(name):
        for ext in [".csv", ".xlsx", ".xls"]:
            p = data_dir / f"{name}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No {name}.csv/xlsx found in {data_dir}")

    train_df = normalize_columns(load_table(pick_file("train")))
    val_df   = normalize_columns(load_table(pick_file("val")))
    test_df  = normalize_columns(load_table(pick_file("test")))

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, local_files_only=True
    ).to(device)

    collator = SafeDataCollator(tokenizer)

    train_loader = DataLoader(
        TextClsDataset(train_df, tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        TextClsDataset(val_df, tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )
    test_loader = DataLoader(
        TextClsDataset(test_df, tokenizer, args.max_len),
        batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    counts = np.bincount(train_df["label"].values)
    weights = counts.sum() / (2.0 * counts)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float, device=device)
    )

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = root / "checkpoints" / args.dataset / sanitize_name(args.model) / time.strftime("%Y%m%d-%H%M%S")
    ensure_dir(run_dir)

    history = []
    best_val_loss = float("inf")
    wait = 0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, loss_fn, scaler, use_amp
        )

        train_metrics = eval_loss_and_metrics(model, train_loader, device, loss_fn)
        val_metrics   = eval_loss_and_metrics(model, val_loader, device, loss_fn)

        print(
            f"Epoch {epoch} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f}"
        )

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            wait = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            wait += 1
            if wait >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

    pd.DataFrame(history).to_csv(
        run_dir / f"metrics_train_val_{args.dataset}.csv", index=False
    )

    model.load_state_dict(torch.load(run_dir / "best.pt"))

    test_metrics = eval_loss_and_metrics(model, test_loader, device, loss_fn)
    pd.DataFrame([test_metrics]).to_csv(
        run_dir / f"metrics_test_{args.dataset}.csv", index=False
    )

    for split, df in [("train", train_df), ("test", test_df)]:
        loader = DataLoader(
            TextClsDataset(df, tokenizer, args.max_len),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator
        )
        pred = predict_df(model, loader, device)
        out = df.merge(pred, on="_row_id", how="left", sort=False)
        out.drop(columns=["_row_id"], inplace=True)
        out.to_csv(
            run_dir / f"{split}_with_h_model_{args.dataset}.csv",
            index=False
        )

    print(f"Stage-1 done. Results saved to {run_dir}")

if __name__ == "__main__":
    main()
