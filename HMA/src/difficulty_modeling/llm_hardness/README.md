# Section 3.1.2 — LLM-based Semantic Hardness (h^LLM)

Soft Prompt Tuning on top of a frozen LLM to estimate sample-level
semantic hardness. The five per-dataset scripts are unified into one
`train_llm_hardness.py` driven by a YAML config.

## Run

```bash
python train_llm_hardness.py --config ../../../configs/llm_hardness/pheme.yaml
```

## Per-dataset specifics (kept faithfully)

| Dataset    | padding | dropout | LayerNorm | dynamic pool | lr_prompt | pos_weight |
|------------|---------|---------|-----------|--------------|-----------|------------|
| Constraint | left    | 0.3     | no        | no           | 1e-3      | 1.2        |
| PHEME      | right   | 0.2     | yes       | yes          | 1e-2      | 2.0        |
| Twitter15  | left    | 0.3     | no        | no           | 1e-3      | 2.0        |
| Twitter16  | left    | 0.5     | no        | no           | 1e-3      | 2.0        |
| Weibo      | left    | 0.3     | no        | no           | 1e-3      | 1.5        |

The Soft Prompt initialization text (4 dimensions + prefix) is stored
verbatim in each YAML and was verified character-for-character against the
original scripts.

## Adding Qwen2.5 (or other LLMs)

The driver is model-agnostic. To add a Qwen2.5-7B / 1.5B variant, just copy a
YAML, change `model.name`, and (optionally) put it under a
`configs/llm_hardness/<model>/` sub-folder. No code change needed.
