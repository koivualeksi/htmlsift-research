WCXB fine-tune -- word-F1 by their evaluate.py. WCXB carves no val fold, so the val columns are dev1497 (train == val == dev) and the test table is test511. The page-type ladder is the exception: its val is that type's dev left after the top rung, and its test is that type's test511 pages.

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

## Training-set size by page type -- val (epoch-mean)

### 311m

| type | pages | layers | ep | bigru/none | bigru/ABC |
|---|---|---|---|---|---|
| article | 50 | 10 | 4 | 0.9041±0.0016 | — |
| article | 100 | 10 | 4 | 0.9320±0.0025 | — |
| article | 150 | 10 | 4 | 0.9386±0.0013 | — |
| article | 200 | 10 | 4 | 0.9368±0.0140 | — |
| article | 250 | 10 | 4 | 0.9404±0.0031 | — |
| article | 500 | 10 | 4 | 0.9572±0.0008 | — |
| forum | 50 | 10 | 4 | 0.7638±0.0138 | 0.7663±0.0168 |
| forum | 100 | 10 | 4 | 0.8058±0.0015 | 0.8077±0.0021 |
| service | 50 | 10 | 4 | 0.8095±0.0071 | 0.8161±0.0048 |
| service | 100 | 10 | 4 | 0.8289±0.0080 | 0.8384±0.0073 |
| service | 150 | 10 | 4 | 0.8517±0.0048 | 0.8460±0.0081 |

## Training-set size by page type -- val (best-epoch)

### 311m

| type | pages | layers | ep | bigru/none | bigru/ABC |
|---|---|---|---|---|---|
| article | 50 | 10 | 4 | 0.9117±0.0025 | — |
| article | 100 | 10 | 4 | 0.9429±0.0017 | — |
| article | 150 | 10 | 4 | 0.9492±0.0009 | — |
| article | 200 | 10 | 4 | 0.9503±0.0011 | — |
| article | 250 | 10 | 4 | 0.9513±0.0058 | — |
| article | 500 | 10 | 4 | 0.9619±0.0006 | — |
| forum | 50 | 10 | 4 | 0.7888±0.0215 | 0.7927±0.0233 |
| forum | 100 | 10 | 4 | 0.8255±0.0016 | 0.8274±0.0008 |
| service | 50 | 10 | 4 | 0.8438±0.0099 | 0.8509±0.0081 |
| service | 100 | 10 | 4 | 0.8541±0.0112 | 0.8616±0.0094 |
| service | 150 | 10 | 4 | 0.8649±0.0030 | 0.8660±0.0094 |

## Training-set size by page type -- test

| type | pages | model | layers | ep | head/feats | seeds | n | F1 | P | R |
|---|---|---|---|---|---|---|---|---|---|---|
| article | 50 | 311m | 10 | 4 | bigru/none | 3 | 257 | 0.9401±0.0007 | 0.9377±0.0020 | 0.9591±0.0023 |
| article | 100 | 311m | 10 | 4 | bigru/none | 3 | 257 | 0.9554±0.0011 | 0.9473±0.0025 | 0.9783±0.0012 |
| article | 150 | 311m | 10 | 4 | bigru/none | 3 | 257 | 0.9626±0.0014 | 0.9550±0.0038 | 0.9838±0.0017 |
| article | 200 | 311m | 10 | 4 | bigru/none | 3 | 257 | 0.9627±0.0009 | 0.9571±0.0009 | 0.9808±0.0024 |
| article | 250 | 311m | 10 | 4 | bigru/none | 3 | 257 | 0.9641±0.0007 | 0.9593±0.0043 | 0.9817±0.0043 |
| article | 500 | 311m | 10 | 4 | bigru/none | 3 | 257 | 0.9672±0.0026 | 0.9634±0.0025 | 0.9818±0.0026 |
| forum | 50 | 311m | 10 | 4 | bigru/ABC | 3 | 51 | 0.8343±0.0264 | 0.7810±0.0498 | 0.9403±0.0149 |
| forum | 50 | 311m | 10 | 4 | bigru/none | 3 | 51 | 0.8438±0.0074 | 0.7981±0.0163 | 0.9357±0.0163 |
| forum | 100 | 311m | 10 | 4 | bigru/ABC | 3 | 51 | 0.8829±0.0078 | 0.8589±0.0207 | 0.9359±0.0090 |
| forum | 100 | 311m | 10 | 4 | bigru/none | 3 | 51 | 0.8778±0.0046 | 0.8477±0.0128 | 0.9377±0.0063 |
| service | 50 | 311m | 10 | 4 | bigru/ABC | 3 | 59 | 0.8158±0.0087 | 0.7850±0.0068 | 0.8978±0.0110 |
| service | 50 | 311m | 10 | 4 | bigru/none | 3 | 59 | 0.8128±0.0141 | 0.7822±0.0085 | 0.8964±0.0241 |
| service | 100 | 311m | 10 | 4 | bigru/ABC | 3 | 59 | 0.8134±0.0107 | 0.7973±0.0068 | 0.8828±0.0220 |
| service | 100 | 311m | 10 | 4 | bigru/none | 3 | 59 | 0.8190±0.0035 | 0.7925±0.0078 | 0.9001±0.0094 |
| service | 150 | 311m | 10 | 4 | bigru/ABC | 3 | 59 | 0.8270±0.0100 | 0.7996±0.0166 | 0.9046±0.0226 |
| service | 150 | 311m | 10 | 4 | bigru/none | 3 | 59 | 0.8246±0.0021 | 0.8007±0.0113 | 0.8987±0.0097 |

