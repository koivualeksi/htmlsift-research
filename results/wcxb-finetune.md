WCXB fine-tune -- word-F1 by their evaluate.py. WCXB carves no val fold, so the val columns are dev1497 (train == val == dev) and the test table is test511.

Regenerate: `python tools/download_results.py`

## val (epoch-mean)

### 311m

| layers | ep | bigru/none |
|---|---|---|
| 10 | 4 | 0.9357±0.0014 |

## val (best-epoch)

### 311m

| layers | ep | bigru/none |
|---|---|---|
| 10 | 4 | 0.9628±0.0007 |

## test

| model | layers | ep | head/feats | seeds | n | F1 | P | R |
|---|---|---|---|---|---|---|---|---|
| 311m | 10 | 4 | bigru/none | 3 | 511 | 0.9209±0.0007 | 0.9214±0.0021 | 0.9404±0.0029 |

