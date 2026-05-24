#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------
# HMA - Section 3.2.2: Stage-2 Gate (how strongly to augment)
#
# For samples flagged as hard (m_hard==1), predict two semantic
# conditional thresholds (t1, t2) from [CLS || h_x], partitioning the
# difficulty axis into Light / Medium / Strong. Weakly supervised by
# an ambiguity ranking derived from the best small-model checkpoint.
# Outputs t1_sem / t2_sem / p_L / p_M / p_S / stage2_level_pred.
#
# Usage:
#   python gate_stage2.py \
#       --input_xlsx  <..._train_with_stage1_gate.xlsx> \
#       --output_xlsx <..._train_with_stage2_gate.xlsx> \
#       --hf_model_dir <small model dir> --best_ckpt <best.pt>
# ------------------------------------------------------------

import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# =====================
# Utils
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =====================
# Dataset
# =====================
class Stage2Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        h = float(row["h_x"])
        amb = float(row["amb"])
        _id = str(row["id"])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["h_x"] = torch.tensor(h, dtype=torch.float32)
        item["amb"] = torch.tensor(amb, dtype=torch.float32)
        item["id"] = _id
        return item


# =====================
# Stage-2 Model (CLS + hx)
# =====================
class SemanticThresholdNet(nn.Module):
    """
    t1(x) = sigmoid(g1([CLS||hx]))
    t2(x) = t1 + softplus(g2([CLS||hx]))
    """

    def __init__(self, backbone, hidden=128, eps=1e-4):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 输入维度 768 + 1，因为拼接了 hx
        self.mlp = nn.Sequential(
            nn.Linear(768 + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
        self.eps = eps

    def forward(self, input_ids, attention_mask, h_x):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # CLS token [B,768]

        h_x = h_x.unsqueeze(1)  # [B,1]
        cls_h = torch.cat([cls, h_x], dim=1)  # 拼接 CLS + hx -> [B,769]

        a, b = self.mlp(cls_h).chunk(2, dim=1)
        t1 = torch.sigmoid(a.squeeze(1))
        delta = F.softplus(b.squeeze(1)) + self.eps
        t2 = torch.clamp(t1 + delta, max=1.0 - self.eps)
        return t1, t2


# =====================
# Loss helpers
# =====================
def soft_intensity_probs(h, t1, t2, gamma):
    pL = torch.sigmoid(gamma * (t1 - h))
    pS = torch.sigmoid(gamma * (h - t2))
    pM = 1.0 - pL - pS
    pM = torch.clamp(pM, 0.0, 1.0)
    s = pL + pM + pS + 1e-12
    return pL / s, pM / s, pS / s


def rank_loss_pairs(t, amb, num_pairs=256):
    B = t.size(0)
    if B < 2:
        return torch.tensor(0.0, device=t.device)

    i = torch.randint(0, B, (num_pairs,), device=t.device)
    j = torch.randint(0, B, (num_pairs,), device=t.device)
    mask = amb[i] > amb[j]
    if mask.sum() == 0:
        return torch.tensor(0.0, device=t.device)

    return F.softplus(t[i][mask] - t[j][mask]).mean()


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser("Stage-2 Augmentation Intensity Gate")

    parser.add_argument("--input_xlsx", type=str, required=True)
    parser.add_argument("--output_xlsx", type=str, required=True)
    parser.add_argument("--hf_model_dir", type=str, required=True)
    parser.add_argument("--best_ckpt", type=str, required=True)

    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_infer", type=int, default=32)
    parser.add_argument("--batch_train", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--rho_L", type=float, default=0.3)
    parser.add_argument("--rho_M", type=float, default=0.4)
    parser.add_argument("--rho_S", type=float, default=0.3)

    parser.add_argument("--gamma", type=float, default=20.0)
    parser.add_argument("--delta_min", type=float, default=0.08)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ------------------
    # Load data
    # ------------------
    df = pd.read_excel(args.input_xlsx)
    need_cols = ["id", "text", "h_x", "m_hard"]
    miss = [c for c in need_cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    df_hard = df[df["m_hard"] == 1].copy().reset_index(drop=True)
    print("Stage-2 samples:", len(df_hard))

    # ------------------
    # Load best h_model
    # ------------------
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.hf_model_dir, num_labels=2, local_files_only=True
    ).to(device)

    state_dict = torch.load(args.best_ckpt, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "bert"):
        backbone = model.bert
    elif hasattr(model, "roberta"):
        backbone = model.roberta
    elif hasattr(model, "deberta"):
        backbone = model.deberta
    else:
        raise ValueError(f"Unknown backbone in model: {type(model)}")

    # ------------------
    # Compute ambiguity
    # ------------------
    class InferDataset(Dataset):
        def __init__(self, df):
            self.df = df

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            enc = tokenizer(
                str(row["text"]),
                truncation=True,
                padding="max_length",
                max_length=args.max_len,
                return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            return item

    dl_inf = DataLoader(InferDataset(df_hard), batch_size=args.batch_infer, shuffle=False)
    margins = []
    with torch.no_grad():
        for batch in tqdm(dl_inf, desc="Compute ambiguity"):
            batch = {k: v.to(device) for k, v in batch.items()}
            prob = torch.softmax(model(**batch).logits, dim=-1)
            top2 = torch.topk(prob, k=2, dim=-1).values
            margins.extend((top2[:, 0] - top2[:, 1]).cpu().numpy())

    m = np.array(margins)
    ql, qh = np.quantile(m, 0.05), np.quantile(m, 0.95)
    m = np.clip(m, ql, qh)
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    df_hard["amb"] = 1.0 - m

    # ------------------
    # Train Stage-2
    # ------------------
    stage2 = SemanticThresholdNet(backbone).to(device)
    opt = torch.optim.AdamW(stage2.parameters(), lr=args.lr)
    dl_train = DataLoader(Stage2Dataset(df_hard, tokenizer, args.max_len),
                          batch_size=args.batch_train, shuffle=True)

    for ep in range(args.epochs):
        stage2.train()
        losses = []
        for batch in tqdm(dl_train, desc=f"Epoch {ep + 1}/{args.epochs}"):
            opt.zero_grad()
            h = batch.pop("h_x").to(device)
            amb = batch.pop("amb").to(device)
            batch_inputs = {k: v.to(device) for k, v in batch.items() if k not in ["id", "token_type_ids"]}
            t1, t2 = stage2(**batch_inputs, h_x=h)

            L_rank = rank_loss_pairs(t2, amb) + 0.5 * rank_loss_pairs(t1, amb)
            pL, pM, pS = soft_intensity_probs(h, t1, t2, args.gamma)
            L_budget = ((pL.mean() - args.rho_L) ** 2 +
                        (pM.mean() - args.rho_M) ** 2 +
                        (pS.mean() - args.rho_S) ** 2)
            gap = torch.relu(args.delta_min - (t2 - t1)).pow(2).mean()

            loss = L_rank + L_budget + 2.0 * gap
            loss.backward()
            opt.step()
            losses.append(loss.item())

        print(f"[Epoch {ep + 1}] loss={np.mean(losses):.4f}")

    # ------------------
    # Inference
    # ------------------
    stage2.eval()
    dl_eval = DataLoader(Stage2Dataset(df_hard, tokenizer, args.max_len),
                         batch_size=64, shuffle=False)
    records = []
    with torch.no_grad():
        for batch in dl_eval:
            ids = batch["id"]
            h = batch["h_x"].to(device)
            batch_inputs = {k: v.to(device) for k, v in batch.items() if
                            k not in ["id", "h_x", "amb", "token_type_ids"]}
            t1, t2 = stage2(**batch_inputs, h_x=h)
            pL, pM, pS = soft_intensity_probs(h, t1, t2, args.gamma)
            prob = torch.stack([pL, pM, pS], dim=1).cpu().numpy()
            level = prob.argmax(axis=1)
            for i in range(len(ids)):
                records.append([ids[i], t1[i].item(), t2[i].item(), prob[i, 0], prob[i, 1], prob[i, 2], level[i]])

    pred_df = pd.DataFrame(records, columns=["id", "t1_sem", "t2_sem", "p_L", "p_M", "p_S", "stage2_level_pred"])
    df["id"] = df["id"].astype(str)
    pred_df["id"] = pred_df["id"].astype(str)
    out = df.merge(pred_df, on="id", how="left")
    out.loc[out["m_hard"] == 0, "stage2_level_pred"] = -1
    out.to_excel(args.output_xlsx, index=False)
    print("Saved:", args.output_xlsx)


if __name__ == "__main__":
    main()