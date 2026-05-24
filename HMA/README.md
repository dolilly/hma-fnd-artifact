# HMA: Hardness-aware Fine-grained LLM Augmentation for Fake News Detection

HMA is a difficulty-aware data augmentation framework for fake news
detection. It (1) models per-sample classification difficulty, (2) decides
**whether** and **how strongly** to augment each hard sample via a two-stage
gate, and (3) generates fine-grained, multi-style augmentations with an LLM,
before re-training the final detector on the augmented set.

## Framework Overview

```
                 ┌─────────────────────────── Stage A: Difficulty Modeling ───────────────────────────┐
  raw sample ──► │  3.1.1  Small-model uncertainty (RoBERTa)  ─► h_model                                │
                 │  3.1.2  LLM semantic complexity (Soft Prompt + Llama-3-8B)  ─► h_LLM                  │
                 │  3.1.3  Gated fusion network  ─► h_x  (overall difficulty)                            │
                 └───────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                 ┌─────────────────────────── Stage B: Two-stage Gate ──────────────────────────────────┐
                 │  3.2.1  Stage-1 gate: learn threshold τ  ─► m_hard  (augment or not)                  │
                 │  3.2.2  Stage-2 gate: per-sample thresholds ─► Light / Medium / Strong                │
                 └───────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                 ┌─────────────────────────── Stage C: Augmentation ────────────────────────────────────┐
                 │  3.3  DeepSeek-R1: Rewrite / Expand / Disguise × Light / Medium / Strong              │
                 └───────────────────────────────────────────────────────────────────────────────────┘
                                                  │
                          Final detector: re-train RoBERTa on the augmented training set
```

## Repository Structure

```
HMA/
├── README.md
├── requirements.txt
├── .env.example                 # template for DEEPSEEK_API_KEY
├── .gitignore
├── src/
│   ├── difficulty_modeling/
│   │   ├── small_model_hardness/
│   │   │   └── compute_h_model.py        # 3.1.1  small-model uncertainty -> h_model
│   │   ├── llm_hardness/
│   │   │   ├── train_llm_hardness.py     # 3.1.2  Soft Prompt + LLM -> h_LLM (unified)
│   │   │   └── README.md
│   │   └── fusion/
│   │       └── fusion_mlp.py             # 3.1.3  gated fusion MLP -> h_x
│   ├── gating/
│   │   ├── gate_stage1.py                # 3.2.1  Stage-1 gate (threshold tau, m_hard)
│   │   └── gate_stage2.py                # 3.2.2  Stage-2 gate (intensity level)
│   ├── augmentation/
│   │   └── deepseek_augment/
│   │       ├── run_augmentation.py       # 3.3   DeepSeek-R1 augmentation (unified)
│   │       └── README.md
│   └── detection/
│       └── train_final.py                # final detector trained on augmented set
├── configs/
│   ├── llm_hardness/                     # 5 YAML configs (one per dataset)
│   │   ├── constraint.yaml
│   │   ├── pheme.yaml
│   │   ├── twitter15.yaml
│   │   ├── twitter16.yaml
│   │   └── weibo.yaml
│   └── augmentation/
│       ├── constraint.yaml  pheme.yaml  twitter15.yaml  twitter16.yaml  weibo.yaml
│       └── prompts/
│           ├── prompts_en_medical.json   # Constraint
│           ├── prompts_en_generic.json   # PHEME / Twitter15 / Twitter16
│           └── prompts_zh_weibo.json     # Weibo
├── scripts/
│   ├── run_llm_hardness_all.sh
│   ├── run_augmentation_all.sh
│   └── run_pipeline_constraint.sh        # end-to-end example for one dataset
└── data/
    └── README.md                         # dataset layout (raw data git-ignored)
```

## Datasets

Constraint, PHEME, Twitter15, Twitter16, Weibo. See `data/README.md` for the
expected layout and label conventions.

## Installation

```bash
git clone <your-repo-url> HMA && cd HMA
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

You also need local copies of the backbone models (e.g. `roberta-base`,
`xlm-roberta-base`, `Meta-Llama-3-8B-Instruct`). Point the config / CLI paths
to wherever you store them. All paths in the configs are **placeholders**
(`models/...`, `data/...`) — adjust them to your environment.

## Quick Start

### Stage A — Difficulty modeling

```bash
# 3.1.1 small-model uncertainty
python src/difficulty_modeling/small_model_hardness/compute_h_model.py \
  --root . --dataset Constraint --model models/roberta-base

# 3.1.2 LLM semantic complexity (run one or all datasets)
python src/difficulty_modeling/llm_hardness/train_llm_hardness.py \
  --config configs/llm_hardness/constraint.yaml
bash scripts/run_llm_hardness_all.sh

# 3.1.3 fusion -> h_x
python src/difficulty_modeling/fusion/fusion_mlp.py --root . --dataset Constraint
```

### Stage B — Two-stage gate

```bash
python src/gating/gate_stage1.py \
  --input_xlsx  data/stage_a/Constraint_train_with_hx_mlp.xlsx \
  --output_xlsx data/stage_b/Constraint_train_with_stage1_gate.xlsx

python src/gating/gate_stage2.py \
  --input_xlsx  data/stage_b/Constraint_train_with_stage1_gate.xlsx \
  --output_xlsx data/stage_b/Constraint_train_with_stage2_gate.xlsx \
  --hf_model_dir models/roberta-base \
  --best_ckpt    checkpoints/Constraint/best.pt
```

### Stage C — Augmentation + final training

```bash
export DEEPSEEK_API_KEY=sk-xxxx          # or: cp .env.example .env && edit it
bash scripts/run_augmentation_all.sh

python src/detection/train_final.py \
  --root . --dataset Constraint --model models/roberta-base \
  --train_aug results/augmented/5-roberta-base+llama3-8b/Constraint_Augmented_Dataset_deepseek_R1.xlsx \
  --init_ckpt checkpoints/Constraint/best.pt
```

A single-dataset end-to-end example is in `scripts/run_pipeline_constraint.sh`.

## Default Configuration

RoBERTa + Llama-3-8B + DeepSeek-R1, augmentation budget ρ = 0.2, virtual
tokens = 60. Per-dataset hyper-parameters are in the YAML configs (see the
table in `src/difficulty_modeling/llm_hardness/README.md`).

## Security Note

No secrets are committed. The DeepSeek API key is read from
`DEEPSEEK_API_KEY` (env var or `.env`, both git-ignored). If a key was ever
exposed, revoke it in the DeepSeek console and generate a new one.

## Reproducibility

All scripts set a fixed random seed (default 42). The Soft Prompt
initialization texts and augmentation prompts were verified
character-for-character against the original per-dataset scripts during
consolidation.
