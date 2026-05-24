#!/usr/bin/env bash
# Section 3.3 - run DeepSeek augmentation on all five datasets
# Requires: export DEEPSEEK_API_KEY=sk-xxxx  (or a local .env file)
set -e
cd "$(dirname "$0")/.."

for ds in constraint pheme twitter15 twitter16 weibo; do
  echo "===== Augmentation: ${ds} ====="
  python src/augmentation/deepseek_augment/run_augmentation.py \
    --config configs/augmentation/${ds}.yaml
done
