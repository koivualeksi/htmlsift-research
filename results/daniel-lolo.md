DAnIEL leave-one-language-out -- ROUGE-L F1, our reimplementation of the SIGIR 2025 metric (LIMITS.md). The result is the test table: train on four languages, score the fifth, so n is that language's page count. The val columns are the four training languages and are in-domain, not a board number.

Regenerate: `python tools/download_results.py`

## Held-out language -- val (epoch-mean)

### 311m

| holdout | layers | ep | bigru/none |
|---|---|---|---|
| Chinese | 10 | 4 | 0.9633±0.0025 |
| English | 10 | 4 | 0.9644±0.0007 |
| Greek | 10 | 4 | 0.9636±0.0010 |
| Polish | 10 | 4 | 0.9712±0.0011 |
| Russian | 10 | 4 | 0.9727±0.0014 |

## Held-out language -- val (best-epoch)

### 311m

| holdout | layers | ep | bigru/none |
|---|---|---|---|
| Chinese | 10 | 4 | 0.9743±0.0001 |
| English | 10 | 4 | 0.9727±0.0003 |
| Greek | 10 | 4 | 0.9748±0.0002 |
| Polish | 10 | 4 | 0.9810±0.0003 |
| Russian | 10 | 4 | 0.9828±0.0007 |

## Held-out language -- test

| holdout | model | layers | ep | head/feats | seeds | n | F1 | P | R |
|---|---|---|---|---|---|---|---|---|---|
| Chinese | 311m | 10 | 4 | bigru/none | 3 | 401 | 0.9595±0.0027 | 0.9716±0.0014 | 0.9568±0.0035 |
| English | 311m | 10 | 4 | bigru/none | 3 | 475 | 0.9507±0.0005 | 0.9381±0.0010 | 0.9782±0.0009 |
| Greek | 311m | 10 | 4 | bigru/none | 3 | 273 | 0.9555±0.0083 | 0.9663±0.0106 | 0.9558±0.0038 |
| Polish | 311m | 10 | 4 | bigru/none | 3 | 274 | 0.9174±0.0030 | 0.9323±0.0039 | 0.9403±0.0020 |
| Russian | 311m | 10 | 4 | bigru/none | 3 | 266 | 0.9021±0.0042 | 0.8947±0.0073 | 0.9290±0.0058 |

