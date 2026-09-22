WCXB fine-tune W8A8-QAT -- word-F1 by their evaluate.py, val columns dev1497 (WCXB carves no val fold), test table test511.

Regenerate: `python tools/download_results.py`

## val (epoch-mean)

### 97m

| layers | ep | bigru/none |
|---|---|---|
| 6 | 4 | 0.9178±0.0039 |

## val (best-epoch)

### 97m

| layers | ep | bigru/none |
|---|---|---|
| 6 | 4 | 0.9433±0.0009 |

## test

| model | layers | ep | head/feats | seeds | n | F1 | P | R |
|---|---|---|---|---|---|---|---|---|
| 97m | 6 | 4 | bigru/none | 3 | 511 | 0.9096±0.0009 | 0.9142±0.0010 | 0.9331±0.0014 |

