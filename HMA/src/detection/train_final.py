#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------
# HMA - Final Detector Training (on the augmented training set)
#
# Re-train the small detector (e.g. RoBERTa), warm-started from the
# best Stage-A checkpoint, on the DeepSeek-augmented training set.
# Selects the best epoch by validation macro-F1 (early stopping) and
# reports test metrics + saves predictions.
#
# Usage:
#   python train_final.py --root . --dataset Constraint \
#       --model models/roberta-base \
#       --train_aug <..._Augmented_Dataset_deepseek_R1.xlsx> \
#       --init_ckpt <best.pt>
# ------------------------------------------------------------

import argparse
import random
import time
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
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Data utilities
# ------------------------------------------------------------
def normalize_columns(df):
    if "text" not in df.columns:
        for c in ["content", "tweet", "sentence", "raw_text", "title"]:
            if c in df.columns:
                df.rename(columns={c: "text"}, inplace=True)
                break
        else:
            raise ValueError("No text column found")
    if "label" not in df.columns:
        raise ValueError("No label column found")
    df["text"] = df["text"].astype(str).fillna("")
    df["label"] = df["label"].astype(int)
    return df

def load_table(path):
    if str(path).endswith(".xlsx"):
        return pd.read_excel(path)
    elif str(path).endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format")

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class TextClsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_len,
        )
        enc["labels"] = row["label"]
        return enc

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
@torch.no_grad()
def eval_metrics(model, loader, device, loss_fn):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    for batch in loader:
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
    fake_f1 = f1_score(
        y_true, y_pred, labels=[1], average="macro", zero_division=0
    )
    return {
        "loss": float(np.mean(losses)),
        "accuracy": acc,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf1,
        "fake_f1": fake_f1,
    }

# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------
@torch.no_grad()
def predict_dataset(model, loader, device):
    model.eval()
    preds = []
    probs = []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        prob = torch.softmax(logits, dim=-1)
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        probs.extend(prob.cpu().numpy())
    return np.array(preds), np.array(probs)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn):
    model.train()
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--train_aug", type=str, required=True)
    parser.add_argument("--init_ckpt", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.root)
    data_dir = root / "data" / args.dataset

    train_df = normalize_columns(load_table(args.train_aug))[["text","label"]]
    val_df = normalize_columns(load_table(data_dir / "val.xlsx"))
    test_df = normalize_columns(load_table(data_dir / "test.xlsx"))

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, local_files_only=True
    ).to(device)

    # warm start
    state_dict = torch.load(args.init_ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(TextClsDataset(train_df, tokenizer, args.max_len),
                              batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(TextClsDataset(val_df, tokenizer, args.max_len),
                            batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    test_loader = DataLoader(TextClsDataset(test_df, tokenizer, args.max_len),
                             batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    # class weights
    counts = np.bincount(train_df["label"].values, minlength=2)
    weights = counts.sum() / (2.0 * counts)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device=device))

    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )

    combo = Path(args.train_aug).parent.name
    run_dir = root / "checkpoints" / args.dataset / "final_aug" / combo / time.strftime("%Y%m%d-%H%M%S")
    ensure_dir(run_dir)

    best_macro_f1 = 0
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
        val_metrics = eval_metrics(model, val_loader, device, loss_fn)
        print(f"Epoch {epoch} | val_loss={val_metrics['loss']:.4f} | val_macroF1={val_metrics['macro_f1']:.4f}")
        history.append({"epoch": epoch, **val_metrics})
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            wait = 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            wait += 1
            if wait >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

    # 保存验证指标
    pd.DataFrame(history).to_excel(run_dir / "metrics_val.xlsx", index=False)

    # load best model
    model.load_state_dict(torch.load(run_dir / "best.pt"))

    # 保存测试集指标
    test_metrics = eval_metrics(model, test_loader, device, loss_fn)
    pd.DataFrame([test_metrics]).to_excel(run_dir / "metrics_test.xlsx", index=False)
    print("Test metrics saved.")

    # 对增强训练集预测
    train_preds, train_probs = predict_dataset(model, train_loader, device)
    train_out = train_df.copy()
    train_out["pred"] = train_preds
    train_out["prob_0"] = train_probs[:, 0]
    train_out["prob_1"] = train_probs[:, 1]
    train_out.to_excel(run_dir / "train_aug_predictions.xlsx", index=False)

    # 对测试集预测
    test_preds, test_probs = predict_dataset(model, test_loader, device)
    test_out = test_df.copy()
    test_out["pred"] = test_preds
    test_out["prob_0"] = test_probs[:, 0]
    test_out["prob_1"] = test_probs[:, 1]
    test_out.to_excel(run_dir / "test_predictions.xlsx", index=False)

    print("Training and prediction finished.")
    print("All results saved to:", run_dir)

if __name__ == "__main__":
    main()