WMB fine-tune W8A8-QAT -- val732 ROUGE-5 F1 on both bases (§6); the test545 table is the arms re-run with --export-test, and its seed count is its own.

Regenerate: `python tools/download_results.py`

## val (epoch-mean)

### 97m

| layers | ep | bigru/none |
|---|---|---|
| 6 | 4 | 0.8779±0.0041 |
| 6 | 8 | 0.8820±0.0009 |

## val (best-epoch)

### 97m

| layers | ep | bigru/none |
|---|---|---|
| 6 | 4 | 0.8821±0.0030 |
| 6 | 8 | 0.8880±0.0026 |

## test

| model | layers | ep | head/feats | seeds | n | F1 | P | R |
|---|---|---|---|---|---|---|---|---|
| 97m | 6 | 4 | bigru/none | 3 | 544 | 0.9261±0.0046 | 0.9270±0.0088 | 0.9508±0.0048 |

## Training-set size -- val (epoch-mean)

### 97m

| pages | layers | ep | bigru/none |
|---|---|---|---|
| 125 | 6 | 4 | 0.7550±0.0118 |
| 250 | 6 | 4 | 0.8200±0.0061 |
| 500 | 6 | 4 | 0.8412±0.0076 |
| 1000 | 6 | 4 | 0.8498±0.0082 |
| 2000 | 6 | 4 | 0.8656±0.0072 |
| 4000 | 6 | 4 | 0.8762±0.0019 |

## Training-set size -- val (best-epoch)

### 97m

| pages | layers | ep | bigru/none |
|---|---|---|---|
| 125 | 6 | 4 | 0.8060±0.0074 |
| 250 | 6 | 4 | 0.8362±0.0037 |
| 500 | 6 | 4 | 0.8479±0.0070 |
| 1000 | 6 | 4 | 0.8620±0.0025 |
| 2000 | 6 | 4 | 0.8743±0.0014 |
| 4000 | 6 | 4 | 0.8799±0.0008 |

