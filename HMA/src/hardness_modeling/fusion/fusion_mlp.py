#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------
# HMA - Section 3.1.3: Gated Fusion Network (overall difficulty h_x)
#
# A small MLP fuses the small-model and LLM hardness signals plus
# three discrete behavioral signals (small-model error, LLM error,
# prediction disagreement) into an overall difficulty score h_x.
#
# Input columns (kept intact & in order):
#   id | text | label | y_pred_model | h_model | y_pred_LLM | h_LLM
# Appended columns (in this exact order):
#   e_model | e_llm | disagree | hard_target | h_x
#
# Usage:
#   python fusion_mlp.py --root . --dataset Constraint
# ------------------------------------------------------------

import os
import random
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Args
# -----------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--stage", type=str, default="stage_a")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# -----------------------
# Reproducibility
# -----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------
# Dataset
# -----------------------
class GatingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------
# Gating MLP
# -----------------------
class DifficultyMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()   # h_x ∈ (0,1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cpu")

    # -------- paths --------
    data_dir = os.path.join(args.root, "data", args.stage)
    input_file = os.path.join(
        data_dir, f"{args.dataset}_train_with_h.xlsx"
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(input_file)

    print("Loading:", input_file)

    # -------- load data --------
    df = pd.read_excel(input_file)

    # -------- strict column check --------
    required_cols = [
        "id", "text", "label",
        "y_pred_model", "h_model",
        "y_pred_LLM", "h_LLM"
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")

    # -------- feature construction (append only) --------

    # error indicators
    df["e_model"] = (df["y_pred_model"] != df["label"]).astype("float32")
    df["e_llm"]   = (df["y_pred_LLM"]   != df["label"]).astype("float32")

    # disagreement
    df["disagree"] = (
        df["y_pred_model"] != df["y_pred_LLM"]
    ).astype("float32")

    # soft target
    df["hard_target"] = (df["e_model"] + df["e_llm"]) / 2.0

    # -------- training data --------
    feature_cols = ["h_model", "h_LLM", "e_model", "e_llm", "disagree"]
    X = df[feature_cols].values.astype("float32")
    y = df["hard_target"].values.astype("float32")

    print("Feature shape:", X.shape)
    print("Hard target distribution:",
          np.unique(y, return_counts=True))

    loader = DataLoader(
        GatingDataset(X, y),
        batch_size=args.batch_size,
        shuffle=True
    )

    # -------- model --------
    model_gate = DifficultyMLP(
        input_dim=len(feature_cols),
        hidden_dim=8
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_gate.parameters(), lr=args.lr)

    # -------- train --------
    for epoch in range(args.epochs):
        model_gate.train()
        total_loss = 0.0

        for Xb, yb in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            pred = model_gate(Xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch+1}/{args.epochs}, "
                f"gating loss: {total_loss / len(loader):.4f}"
            )

    # -------- compute h_x (append as last column) --------
    model_gate.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        h_x = model_gate(X_tensor).cpu().numpy()

    df["h_x"] = h_x   # ← 只追加，不改前面的任何列

    # -------- save full table --------
    output_file = os.path.join(
        data_dir, f"{args.dataset}_train_with_hx_mlp.xlsx"
    )
    df.to_excel(output_file, index=False)

    print("Saved:", output_file)

    # -------- save checkpoint --------
    save_dir = os.path.join(
        args.root, "checkpoints", "gating_mlp", args.dataset
    )
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        model_gate.state_dict(),
        os.path.join(save_dir, "model.pt")
    )

    with open(os.path.join(save_dir, "config.json"),
              "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "feature_cols": feature_cols,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed
        }, f, indent=2)

    print("Gating MLP checkpoint saved.")
    print("Stage-A (fusion MLP) done.")

if __name__ == "__main__":
    main()
