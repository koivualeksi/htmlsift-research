WCXB articles -- data-efficiency ladder. Word-F1 (vendored) at threshold 0.5, mean±std over seeds. Train pages are nested subsets of the 513-page pool (dev articles minus 79 byte-identical test twins and a 200-page val held out by domain); val is that held-out set; test is all 257 test articles; disjoint is the 155 of them whose domain no dev article has. Scratch starts from a random head (mini) or the pretrained backbone (base); arm A from the shipped model, whose zero-shot score is the 0-page row. Mini is scored at rank 128 with its int8 table (the shipped form); base as its pod scored it (GPU). Third-party extractors on the same pages: results/heuristics-wcxb-article.md.

Regenerate: `python bench/accuracy/ladder.py`

Seeds: mini, scratch 3, mini, arm A 3, base, scratch 1, base, arm A 1.

## val

| train pages | mini, scratch | mini, arm A | base, scratch | base, arm A |
|---|---|---|---|---|
| 0 | — | 0.9300 | — | 0.9437 |
| 25 | 0.9091±0.0026 | 0.9335±0.0008 | 0.9375 | 0.9479 |
| 50 | 0.9270±0.0029 | 0.9382±0.0019 | 0.9449 | 0.9477 |
| 100 | 0.9352±0.0014 | 0.9430±0.0028 | 0.9479 | 0.9542 |
| 200 | 0.9374±0.0004 | 0.9450±0.0010 | 0.9515 | 0.9560 |
| 400 | 0.9451±0.0035 | 0.9489±0.0010 | 0.9534 | 0.9578 |
| 513 | 0.9458±0.0015 | 0.9484±0.0036 | 0.9545 | 0.9595 |

## test

| train pages | mini, scratch | mini, arm A | base, scratch | base, arm A |
|---|---|---|---|---|
| 0 | — | 0.9176 | — | 0.9343 |
| 25 | 0.8974±0.0032 | 0.9275±0.0010 | 0.9481 | 0.9369 |
| 50 | 0.9147±0.0070 | 0.9401±0.0015 | 0.9557 | 0.9409 |
| 100 | 0.9168±0.0031 | 0.9444±0.0029 | 0.9557 | 0.9546 |
| 200 | 0.9291±0.0045 | 0.9483±0.0012 | 0.9564 | 0.9519 |
| 400 | 0.9479±0.0008 | 0.9537±0.0039 | 0.9587 | 0.9607 |
| 513 | 0.9513±0.0016 | 0.9562±0.0008 | 0.9666 | 0.9602 |

## disjoint

| train pages | mini, scratch | mini, arm A | base, scratch | base, arm A |
|---|---|---|---|---|
| 0 | — | 0.9209 | — | 0.9332 |
| 25 | 0.9155±0.0031 | 0.9297±0.0017 | 0.9517 | 0.9364 |
| 50 | 0.9316±0.0058 | 0.9461±0.0044 | 0.9612 | 0.9403 |
| 100 | 0.9359±0.0018 | 0.9489±0.0033 | 0.9620 | 0.9570 |
| 200 | 0.9505±0.0034 | 0.9530±0.0026 | 0.9592 | 0.9521 |
| 400 | 0.9545±0.0019 | 0.9539±0.0044 | 0.9621 | 0.9633 |
| 513 | 0.9577±0.0024 | 0.9580±0.0020 | 0.9731 | 0.9618 |
