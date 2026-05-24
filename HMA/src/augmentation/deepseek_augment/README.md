# Section 3.3 — Fine-grained Multi-style Augmentation

For samples selected by the Stage-2 gate, generate augmented text via
DeepSeek-R1 along 3 styles (Rewrite / Expand / Disguise) x 3 strengths
(Light / Medium / Strong). Strong-Disguise produces 2 samples.

## Security

The API key is **never** hardcoded. Provide it via environment variable or a
local `.env` file:

```bash
export DEEPSEEK_API_KEY=sk-xxxx
# or: cp .env.example .env  &&  edit .env
```

## Run

```bash
python run_augmentation.py --config ../../../configs/augmentation/pheme.yaml
```

## Prompt sets (stored as JSON, verified verbatim)

| File                       | Used by                         | Style notes                  |
|----------------------------|---------------------------------|------------------------------|
| `prompts_en_medical.json`  | Constraint                      | medical / CDC disguise role  |
| `prompts_en_generic.json`  | PHEME / Twitter15 / Twitter16   | journalist / police role     |
| `prompts_zh_weibo.json`    | Weibo                           | Chinese gov-style + refusal filter |

`expand` is length-capped (EN: 150 tokens, ZH: 200) to prevent runaway
generation; Weibo additionally drops outputs containing AI-disclaimer phrases.
