# Datasets

This folder holds the datasets used by HMA. **Raw data is git-ignored** and
must be prepared locally. Place each dataset under its own sub-directory:

```
data/
├── Constraint/   { train, val, test }.xlsx   # labels: real/fake or 0/1
├── PHEME/         { train, val, test }.xlsx   # labels: 0=Non-Rumor, 1=Rumor
├── Twitter15/     { train, val, test }.xlsx
├── Twitter16/     { train, val, test }.xlsx
└── Weibo/         { train, val, test }.xlsx   # labels already flipped: 1=Rumor
```

Each file needs at least a `text` column and a `label` column (an `id`
column is recommended; if absent it is generated automatically).

Intermediate artifacts produced by the pipeline (e.g.
`*_train_with_h.xlsx`, `*_train_with_hx_mlp.xlsx`,
`*_train_with_stage2_gate.xlsx`) are also written under `data/`.
