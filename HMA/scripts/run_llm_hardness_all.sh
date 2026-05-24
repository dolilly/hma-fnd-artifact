#!/usr/bin/env bash
# Section 3.1.2 - run LLM hardness on all five datasets
set -e
cd "$(dirname "$0")/.."

for ds in constraint pheme twitter15 twitter16 weibo; do
  echo "===== LLM Hardness: ${ds} ====="
  python src/difficulty_modeling/llm_hardness/train_llm_hardness.py \
    --config configs/llm_hardness/${ds}.yaml
done
