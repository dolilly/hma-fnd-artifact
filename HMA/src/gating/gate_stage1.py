#!/usr/bin/env python
# coding: utf-8
# ------------------------------------------------------------
# HMA - Section 3.2.1: Stage-1 Gate (whether to augment)
#
# Learns a global threshold tau over the overall difficulty h_x to
# decide which samples are "hard" enough to augment. Trained with a
# supervised term (hard_target) plus a budget regularizer toward the
# target augmentation ratio rho. Outputs p_gate / tau / m_hard.
#
# Usage:
#   python gate_stage1.py \
#       --input_xlsx  <..._train_with_hx_mlp.xlsx> \
#       --output_xlsx <..._train_with_stage1_gate.xlsx>
# ------------------------------------------------------------

import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =====================
# Utils
# =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =====================
# Gate Module
# =====================
class LearnableThresholdGate(nn.Module):
    """
    p(x) = sigmoid( beta * (h_x - tau) )
    tau = sigmoid(alpha)
    """
    def __init__(self, init_tau=0.5):
        super().__init__()
        alpha = torch.log(
            torch.tensor(init_tau) / (1.0 - torch.tensor(init_tau))
        )
        self.alpha = nn.Parameter(alpha.float())

    def forward(self, h_x, beta):
        tau = torch.sigmoid(self.alpha)
        p = torch.sigmoid(beta * (h_x - tau))
        return p, tau


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser("Stage-1 Augmentation Gate")

    parser.add_argument("--input_xlsx", type=str, required=True,
                        help="Path to *_train_with_hx_mlp.xlsx")
    parser.add_argument("--output_xlsx", type=str, required=True,
                        help="Output xlsx with stage1 gate results")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--rho", type=float, default=0.2,
                        help="Target augmentation ratio")
    parser.add_argument("--lambda_budget", type=float, default=5.0)

    parser.add_argument("--beta_min", type=float, default=2.0)
    parser.add_argument("--beta_max", type=float, default=20.0)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------
    # Load data
    # ---------------------
    df = pd.read_excel(args.input_xlsx)

    required_cols = ["id", "h_x", "hard_target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    h_x = torch.tensor(df["h_x"].values, dtype=torch.float32).to(device)
    y_sup = torch.tensor(df["hard_target"].values, dtype=torch.float32).to(device)

    # ---------------------
    # Init gate
    # ---------------------
    gate = LearnableThresholdGate(init_tau=0.5).to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=args.lr)
    bce = nn.BCELoss()

    # ---------------------
    # Training
    # ---------------------
    gate.train()
    for epoch in range(args.epochs):
        # beta annealing
        beta = args.beta_min + (args.beta_max - args.beta_min) * (
            epoch / max(1, args.epochs - 1)
        )
        beta = torch.tensor(beta, device=device)

        optimizer.zero_grad()

        p, tau = gate(h_x, beta)

        loss_sup = bce(p, y_sup)
        loss_budget = (p.mean() - args.rho) ** 2
        loss = loss_sup + args.lambda_budget * loss_budget

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"[Epoch {epoch+1:03d}] "
                f"loss={loss.item():.4f} "
                f"sup={loss_sup.item():.4f} "
                f"budget={loss_budget.item():.4f} "
                f"tau={tau.item():.4f} "
                f"p_mean={p.mean().item():.4f}"
            )

    # ---------------------
    # Inference (hard decision)
    # ---------------------
    gate.eval()
    with torch.no_grad():
        beta = torch.tensor(args.beta_max, device=device)
        p_gate, tau = gate(h_x, beta)

    df["p_gate"] = p_gate.cpu().numpy()
    df["tau"] = float(tau.cpu().item())
    df["m_hard"] = (df["p_gate"] > 0.5).astype(int)

    # ---------------------
    # Save
    # ---------------------
    df.to_excel(args.output_xlsx, index=False)
    print("Saved Stage-1 gate result to:", args.output_xlsx)


if __name__ == "__main__":
    main()
