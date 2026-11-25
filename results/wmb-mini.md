WMB mini (experiments/wmb-mini.yaml: 311m token-embedding table, BiGRU head, features ABC) -- ROUGE-5 F1 on val732 (best epoch) and test545. test int8 is the same head with the table int8-quantised at inference, the 0.2.x mini's numerics.

Regenerate: `python tools/render_tables.py`

Config: `recipe=table`, `board=wmb`, `model=311m`, `emb=fp32`, `head=bigru`, `feats=ABC`, `hidden=256`, `epochs=4`, `batch_size=8`, `head_lr=0.002`, `train_limit=None`, `extra=None`, `extra_repeat=None`, `export_test=True`, `keep=True`

Test n: 544.

| seeds | val | test | test int8 |
|---|---|---|---|
| 3 | 0.8613±0.0026 | 0.9004±0.0007 | 0.9005±0.0006 |

## Runs

| run | seed | commit | val | test | test int8 |
|---|---|---|---|---|---|
| 2026-11-20T2138-493b | 0 | 115b03b | 0.8642 | 0.9011 | 0.9011 |
| 2026-11-20T2147-7f3c | 1 | 115b03b | 0.8591 | 0.9001 | 0.9003 |
| 2026-11-20T2155-556e | 2 | 115b03b | 0.8608 | 0.8999 | 0.8999 |
