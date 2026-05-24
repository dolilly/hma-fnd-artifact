#!/usr/bin/env bash
# End-to-end HMA pipeline for ONE dataset (example: Constraint).
# Adjust paths (--root, --model, --data) to your local setup before running.
set -e
cd "$(dirname "$0")/.."

ROOT="."
DATASET="Constraint"
SMALL_MODEL="models/roberta-base"          # 占位符
LLM_MODEL="models/Meta-Llama-3-8B-Instruct" # 占位符

# --- 3.1.1 小模型不确定性难度 h_model ---
python src/difficulty_modeling/small_model_hardness/compute_h_model.py \
  --root "${ROOT}" --dataset "${DATASET}" --model "${SMALL_MODEL}"

# --- 3.1.2 LLM 语义复杂性难度 h_LLM ---
python src/difficulty_modeling/llm_hardness/train_llm_hardness.py \
  --config configs/llm_hardness/constraint.yaml

# --- 3.1.3 门控融合得到综合难度 h_x ---
#   需先将 h_model 与 h_LLM 合并到 {dataset}_train_with_h.xlsx
python src/difficulty_modeling/fusion/fusion_mlp.py \
  --root "${ROOT}" --dataset "${DATASET}"

# --- 3.2.1 Stage-1 门控（是否增强，学习阈值 tau）---
python src/gating/gate_stage1.py \
  --input_xlsx  data/stage_a/${DATASET}_train_with_hx_mlp.xlsx \
  --output_xlsx data/stage_b/${DATASET}_train_with_stage1_gate.xlsx

# --- 3.2.2 Stage-2 门控（强度分配 Light/Medium/Strong）---
python src/gating/gate_stage2.py \
  --input_xlsx  data/stage_b/${DATASET}_train_with_stage1_gate.xlsx \
  --output_xlsx data/stage_b/${DATASET}_train_with_stage2_gate.xlsx \
  --hf_model_dir "${SMALL_MODEL}" \
  --best_ckpt    checkpoints/${DATASET}/best.pt

# --- 3.3 DeepSeek 细粒度增强 ---
#   需先 export DEEPSEEK_API_KEY=sk-xxxx
python src/augmentation/deepseek_augment/run_augmentation.py \
  --config configs/augmentation/constraint.yaml

# --- 最终检测训练（在增强训练集上重训小模型）---
python src/detection/train_final.py \
  --root "${ROOT}" --dataset "${DATASET}" --model "${SMALL_MODEL}" \
  --train_aug results/augmented/5-roberta-base+llama3-8b/${DATASET}_Augmented_Dataset_deepseek_R1.xlsx \
  --init_ckpt checkpoints/${DATASET}/best.pt
