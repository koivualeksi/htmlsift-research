WMB mini objective screens -- the shipped mini (wmb-mini) under changes to the loss, weight averaging and the training schedule; features and inputs fixed. ROUGE-5 F1 on val732 (best epoch) and test545. Run-to-run noise at one seed is ~0.006 val / ~0.009 test, so a row under three seeds is a screen, not a result. None is an option the run predates, so it ran at the default (off).

Regenerate: `python tools/render_tables.py`

Config: `recipe=table`, `board=wmb`, `model=311m`, `emb=fp32`, `head=bigru`, `feats=ABC`, `hidden=256`, `batch_size=8`, `train_limit=None`, `extra=None`, `extra_repeat=None`, `export_test=True`, `keep=True`

Test n: 544.

| epochs | head_lr | len_cap | len_floor | page_norm | len_side | fp_weight | fp_box | text_drop | feat_drop | focal_gamma | label_smooth | smooth_weight | box_mass | plain | ema | page_f1 | seeds | val | test | test int8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.0005 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 3 | 0.8615±0.0033 | 0.9030±0.0024 | 0.9030±0.0025 |
| 4 | 0.001 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 3 | 0.8679±0.0009 | 0.9087±0.0019 | 0.9087±0.0019 |
| 4 | 0.002 | 200 | 0.2 | False | neg | 1.0 | None | None | None | None | None | None | None | None | None | None | 3 | 0.8536±0.0068 | 0.8975±0.0024 | 0.8976±0.0024 |
| 4 | 0.002 | 200 | 0.2 | None | None | None | None | None | None | None | None | None | None | None | None | None | 3 | 0.8589±0.0035 | 0.8956±0.0059 | 0.8955±0.0059 |
| 4 | 0.002 | None | None | False | both | 1.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 1.0 | False | 0.0 | None | 3 | 0.8552±0.0039 | 0.8993±0.0022 | 0.8992±0.0023 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.995 | None | 3 | 0.8668±0.0017 | 0.9053±0.0035 | 0.9054±0.0034 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.999 | None | 5 | 0.8693±0.0023 | 0.9065±0.0031 | 0.9065±0.0032 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.0 | None | 3 | 0.8634±0.0020 | 0.9032±0.0020 | 0.9032±0.0019 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 0.5 | 3 | 0.8736±0.0017 | 0.9112±0.0024 | 0.9112±0.0024 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 1.0 | 3 | 0.8709±0.0008 | 0.9115±0.0016 | 0.9114±0.0017 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 2.0 | 3 | 0.8715±0.0032 | 0.9135±0.0019 | 0.9137±0.0020 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 9 | 0.8739±0.0023 | 0.9130±0.0020 | 0.9129±0.0020 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 8.0 | 3 | 0.8704±0.0039 | 0.9140±0.0019 | 0.9140±0.0019 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 6 | 0.8706±0.0026 | 0.9086±0.0016 | 0.9086±0.0016 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 10.0 | 0.0 | False | 0.999 | None | 3 | 0.8546±0.0024 | 0.9067±0.0070 | 0.9067±0.0070 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 3.0 | 0.0 | False | 0.999 | None | 4 | 0.8686±0.0027 | 0.9103±0.0021 | 0.9103±0.0022 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.05 | 0.0 | 0.0 | False | 0.0 | None | 3 | 0.8580±0.0013 | 0.9008±0.0022 | 0.9008±0.0022 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.05 | 1.0 | 0.0 | False | 0.999 | None | 3 | 0.8672±0.0033 | 0.9100±0.0014 | 0.9098±0.0014 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 2 | 0.0 | 0.0 | 0.0 | False | None | None | 3 | 0.8546±0.0051 | 0.8962±0.0052 | 0.8961±0.0051 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 9 | 0.8746±0.0015 | 0.9118±0.0025 | 0.9118±0.0025 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 3 | 0.8720±0.0024 | 0.9102±0.0031 | 0.9101±0.0032 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.05 | 1.0 | 0.0 | False | 0.999 | None | 3 | 0.8656±0.0025 | 0.9111±0.0017 | 0.9112±0.0017 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | None | None | None | None | None | None | None | 1 | 0.8613 | 0.9025 | 0.9026 |
| 4 | 0.002 | None | None | False | both | 1.0 | None | 0.3 | 0.0 | None | None | None | None | None | None | None | 1 | 0.8523 | 0.8970 | 0.8971 |
| 4 | 0.002 | None | None | False | both | 2.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 3 | 0.8610±0.0026 | 0.9005±0.0029 | 0.9006±0.0028 |
| 4 | 0.002 | None | None | False | both | 2.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 3 | 0.8617±0.0042 | 0.9019±0.0012 | 0.9019±0.0012 |
| 4 | 0.002 | None | None | False | both | 2.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 3 | 0.8732±0.0032 | 0.9133±0.0059 | 0.9136±0.0061 |
| 4 | 0.002 | None | None | False | both | 4.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 3 | 0.8550±0.0010 | 0.9025±0.0026 | 0.9025±0.0025 |
| 4 | 0.002 | None | None | False | both | 4.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 2 | 0.8564±0.0014 | 0.8978±0.0026 | 0.8979±0.0025 |
| 4 | 0.002 | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | 3 | 0.8613±0.0026 | 0.9004±0.0007 | 0.9005±0.0006 |
| 4 | 0.002 | None | None | True | None | None | None | None | None | None | None | None | None | None | None | None | 3 | 0.8573±0.0024 | 0.9000±0.0025 | 0.9000±0.0025 |
| 4 | 0.004 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 3 | 0.8627±0.0008 | 0.9011±0.0051 | 0.9012±0.0050 |
| 6 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 3 | 0.8701±0.0011 | 0.9093±0.0025 | 0.9092±0.0028 |
| 6 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.9995 | None | 3 | 0.8712±0.0013 | 0.9088±0.0026 | 0.9090±0.0028 |

## Runs

| run | epochs | head_lr | len_cap | len_floor | page_norm | len_side | fp_weight | fp_box | text_drop | feat_drop | focal_gamma | label_smooth | smooth_weight | box_mass | plain | ema | page_f1 | seed | commit | val | test | test int8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-21T2252-4aca | 4 | 0.0005 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8577 | 0.9057 | 0.9058 |
| 2026-11-21T2301-05cb | 4 | 0.0005 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8633 | 0.9009 | 0.9010 |
| 2026-11-21T2307-e4bc | 4 | 0.0005 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8636 | 0.9024 | 0.9023 |
| 2026-11-21T2314-cfa5 | 4 | 0.001 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8690 | 0.9108 | 0.9109 |
| 2026-11-21T2320-0378 | 4 | 0.001 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8676 | 0.9080 | 0.9080 |
| 2026-11-21T2327-fbf8 | 4 | 0.001 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8673 | 0.9073 | 0.9072 |
| 2026-11-21T1731-a952 | 4 | 0.002 | 200 | 0.2 | False | neg | 1.0 | None | None | None | None | None | None | None | None | None | None | 0 | 3280a71 | 0.8539 | 0.9003 | 0.9003 |
| 2026-11-21T1740-fb66 | 4 | 0.002 | 200 | 0.2 | False | neg | 1.0 | None | None | None | None | None | None | None | None | None | None | 1 | 3280a71 | 0.8603 | 0.8963 | 0.8964 |
| 2026-11-21T1748-0797 | 4 | 0.002 | 200 | 0.2 | False | neg | 1.0 | None | None | None | None | None | None | None | None | None | None | 2 | 3280a71 | 0.8467 | 0.8959 | 0.8960 |
| 2026-11-21T1559-f87b | 4 | 0.002 | 200 | 0.2 | None | None | None | None | None | None | None | None | None | None | None | None | None | 0 | 3bc6531 | 0.8597 | 0.8969 | 0.8967 |
| 2026-11-21T1603-33ae | 4 | 0.002 | 200 | 0.2 | None | None | None | None | None | None | None | None | None | None | None | None | None | 1 | 3bc6531 | 0.8618 | 0.9008 | 0.9007 |
| 2026-11-21T1606-4c28 | 4 | 0.002 | 200 | 0.2 | None | None | None | None | None | None | None | None | None | None | None | None | None | 2 | 3bc6531 | 0.8551 | 0.8892 | 0.8891 |
| 2026-11-21T2036-5e33 | 4 | 0.002 | None | None | False | both | 1.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 1.0 | False | 0.0 | None | 0 | 013e596 | 0.8514 | 0.9019 | 0.9019 |
| 2026-11-21T2053-bf4a | 4 | 0.002 | None | None | False | both | 1.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 1.0 | False | 0.0 | None | 1 | 013e596 | 0.8591 | 0.8980 | 0.8979 |
| 2026-11-21T2109-145f | 4 | 0.002 | None | None | False | both | 1.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 1.0 | False | 0.0 | None | 2 | 013e596 | 0.8551 | 0.8980 | 0.8979 |
| 2026-11-21T2043-a868 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.995 | None | 0 | 013e596 | 0.8655 | 0.9055 | 0.9055 |
| 2026-11-21T2049-1684 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.995 | None | 1 | 013e596 | 0.8687 | 0.9017 | 0.9019 |
| 2026-11-21T2054-b80b | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.995 | None | 2 | 013e596 | 0.8663 | 0.9088 | 0.9087 |
| 2026-11-21T2059-610a | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.999 | None | 0 | 013e596 | 0.8682 | 0.9088 | 0.9088 |
| 2026-11-21T2127-c941 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.999 | None | 0 | 5b308db | 0.8724 | 0.9080 | 0.9082 |
| 2026-11-21T2104-9c47 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.999 | None | 1 | 013e596 | 0.8705 | 0.9089 | 0.9089 |
| 2026-11-21T2135-7b95 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.999 | None | 1 | 5b308db | 0.8688 | 0.9016 | 0.9013 |
| 2026-11-21T2108-3d39 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | 0.999 | None | 2 | 013e596 | 0.8663 | 0.9052 | 0.9054 |
| 2026-11-21T2035-2d87 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.0 | None | 0 | 013e596 | 0.8634 | 0.9010 | 0.9010 |
| 2026-11-21T2050-ac27 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.0 | None | 1 | 013e596 | 0.8654 | 0.9040 | 0.9039 |
| 2026-11-21T2105-9d0a | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.0 | None | 2 | 013e596 | 0.8615 | 0.9046 | 0.9046 |
| 2026-11-21T2349-470d | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 0.5 | 0 | 83169f2 | 0.8737 | 0.9088 | 0.9087 |
| 2026-11-21T2357-c3f5 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 0.5 | 1 | 83169f2 | 0.8753 | 0.9113 | 0.9112 |
| 2026-11-22T0002-f22d | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 0.5 | 2 | 83169f2 | 0.8719 | 0.9136 | 0.9135 |
| 2026-11-22T0251-dfa0 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 1.0 | 0 | 0d9c0bf | 0.8706 | 0.9097 | 0.9095 |
| 2026-11-22T0300-e385 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 1.0 | 1 | 0d9c0bf | 0.8703 | 0.9128 | 0.9128 |
| 2026-11-22T0307-769f | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 1.0 | 2 | 0d9c0bf | 0.8718 | 0.9119 | 0.9120 |
| 2026-11-22T0007-31fe | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 2.0 | 0 | 83169f2 | 0.8688 | 0.9115 | 0.9116 |
| 2026-11-22T0013-ce10 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 2.0 | 1 | 83169f2 | 0.8750 | 0.9154 | 0.9155 |
| 2026-11-22T0018-825b | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 2.0 | 2 | 83169f2 | 0.8707 | 0.9137 | 0.9141 |
| 2026-11-22T0314-4564 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 0 | 0d9c0bf | 0.8738 | 0.9148 | 0.9148 |
| 2026-11-22T0437-67a4 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 0 | 2ccdc57 | 0.8761 | 0.9146 | 0.9148 |
| 2026-11-22T0321-be1a | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 1 | 0d9c0bf | 0.8758 | 0.9125 | 0.9124 |
| 2026-11-22T0446-9942 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 1 | 2ccdc57 | 0.8779 | 0.9126 | 0.9125 |
| 2026-11-22T0328-7190 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 2 | 0d9c0bf | 0.8708 | 0.9134 | 0.9134 |
| 2026-11-22T0453-5656 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 2 | 2ccdc57 | 0.8718 | 0.9129 | 0.9130 |
| 2026-11-22T0500-2713 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 3 | 2ccdc57 | 0.8729 | 0.9121 | 0.9120 |
| 2026-11-22T0507-1c25 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 4 | 2ccdc57 | 0.8731 | 0.9152 | 0.9150 |
| 2026-11-22T0514-22ba | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 5 | 2ccdc57 | 0.8727 | 0.9086 | 0.9084 |
| 2026-11-22T0352-5a24 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 8.0 | 0 | b2be6aa | 0.8659 | 0.9130 | 0.9128 |
| 2026-11-22T0401-ec3f | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 8.0 | 1 | b2be6aa | 0.8726 | 0.9129 | 0.9129 |
| 2026-11-22T0408-d903 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 8.0 | 2 | b2be6aa | 0.8728 | 0.9162 | 0.9162 |
| 2026-11-21T2127-2f31 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 0 | 23ff3a3 | 0.8726 | 0.9104 | 0.9105 |
| 2026-11-21T2258-6778 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8722 | 0.9088 | 0.9088 |
| 2026-11-21T2131-0927 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 1 | 23ff3a3 | 0.8715 | 0.9064 | 0.9065 |
| 2026-11-21T2306-4132 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8668 | 0.9078 | 0.9079 |
| 2026-11-21T2135-b49f | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 2 | 23ff3a3 | 0.8726 | 0.9076 | 0.9076 |
| 2026-11-21T2313-51a0 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8678 | 0.9105 | 0.9106 |
| 2026-11-21T2319-2af7 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 10.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8569 | 0.9065 | 0.9064 |
| 2026-11-21T2326-eee3 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 10.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8522 | 0.8998 | 0.8998 |
| 2026-11-21T2333-44ad | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 10.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8547 | 0.9139 | 0.9139 |
| 2026-11-21T2138-d73e | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 3.0 | 0.0 | False | 0.999 | None | 0 | 23ff3a3 | 0.8712 | 0.9115 | 0.9114 |
| 2026-11-21T2257-140d | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 3.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8684 | 0.9073 | 0.9072 |
| 2026-11-21T2306-a10e | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 3.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8649 | 0.9121 | 0.9121 |
| 2026-11-21T2313-1284 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 3.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8698 | 0.9104 | 0.9103 |
| 2026-11-21T2029-716e | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.05 | 0.0 | 0.0 | False | 0.0 | None | 0 | 013e596 | 0.8572 | 0.8986 | 0.8984 |
| 2026-11-21T2040-48a1 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.05 | 0.0 | 0.0 | False | 0.0 | None | 1 | 013e596 | 0.8595 | 0.9011 | 0.9011 |
| 2026-11-21T2050-9091 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.05 | 0.0 | 0.0 | False | 0.0 | None | 2 | 013e596 | 0.8572 | 0.9029 | 0.9029 |
| 2026-11-21T2340-e445 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.05 | 1.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8708 | 0.9088 | 0.9087 |
| 2026-11-21T2347-4727 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.05 | 1.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8665 | 0.9115 | 0.9114 |
| 2026-11-21T2354-8ba6 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.05 | 1.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8644 | 0.9096 | 0.9093 |
| 2026-11-21T2003-297f | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 2 | 0.0 | 0.0 | 0.0 | False | None | None | 0 | 9a61c17 | 0.8558 | 0.9019 | 0.9018 |
| 2026-11-21T2021-f2f6 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 2 | 0.0 | 0.0 | 0.0 | False | None | None | 1 | 9a61c17 | 0.8590 | 0.8949 | 0.8949 |
| 2026-11-21T2037-2e0d | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 2 | 0.0 | 0.0 | 0.0 | False | None | None | 2 | 9a61c17 | 0.8491 | 0.8917 | 0.8917 |
| 2026-11-22T0355-960f | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 0 | b2be6aa | 0.8750 | 0.9096 | 0.9095 |
| 2026-11-22T0521-0cbc | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 0 | 2ccdc57 | 0.8726 | 0.9165 | 0.9165 |
| 2026-11-22T0405-243a | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 1 | b2be6aa | 0.8768 | 0.9107 | 0.9109 |
| 2026-11-22T0529-fe03 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 1 | 2ccdc57 | 0.8740 | 0.9095 | 0.9097 |
| 2026-11-22T0412-891b | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 2 | b2be6aa | 0.8768 | 0.9141 | 0.9141 |
| 2026-11-22T0536-e832 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 2 | 2ccdc57 | 0.8756 | 0.9098 | 0.9101 |
| 2026-11-22T0543-f267 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 3 | 2ccdc57 | 0.8738 | 0.9108 | 0.9109 |
| 2026-11-22T0550-1a19 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 4 | 2ccdc57 | 0.8739 | 0.9112 | 0.9109 |
| 2026-11-22T0557-9fbb | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 5 | 2ccdc57 | 0.8730 | 0.9143 | 0.9141 |
| 2026-11-21T2320-5cab | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8741 | 0.9137 | 0.9137 |
| 2026-11-21T2327-3257 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8726 | 0.9089 | 0.9088 |
| 2026-11-21T2334-3143 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8694 | 0.9079 | 0.9079 |
| 2026-11-22T0001-a092 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.05 | 1.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8685 | 0.9107 | 0.9108 |
| 2026-11-22T0008-06d2 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.05 | 1.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8645 | 0.9130 | 0.9131 |
| 2026-11-22T0015-aaf2 | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | 0 | 0.05 | 1.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8639 | 0.9096 | 0.9098 |
| 2026-11-21T1901-0dba | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.3 | None | None | None | None | None | None | None | 0 | 2bdbfc7 | 0.8613 | 0.9025 | 0.9026 |
| 2026-11-21T1902-4aee | 4 | 0.002 | None | None | False | both | 1.0 | None | 0.3 | 0.0 | None | None | None | None | None | None | None | 0 | 2bdbfc7 | 0.8523 | 0.8970 | 0.8971 |
| 2026-11-21T2001-19ae | 4 | 0.002 | None | None | False | both | 2.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 0 | 9a61c17 | 0.8623 | 0.8996 | 0.8995 |
| 2026-11-21T2011-08e4 | 4 | 0.002 | None | None | False | both | 2.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 1 | 9a61c17 | 0.8580 | 0.8982 | 0.8986 |
| 2026-11-21T2019-6fc3 | 4 | 0.002 | None | None | False | both | 2.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 2 | 9a61c17 | 0.8627 | 0.9038 | 0.9038 |
| 2026-11-21T2027-aee1 | 4 | 0.002 | None | None | False | both | 2.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 0 | 9a61c17 | 0.8665 | 0.9022 | 0.9020 |
| 2026-11-21T2036-f6cc | 4 | 0.002 | None | None | False | both | 2.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 1 | 9a61c17 | 0.8597 | 0.9006 | 0.9006 |
| 2026-11-21T2044-189d | 4 | 0.002 | None | None | False | both | 2.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 2 | 9a61c17 | 0.8589 | 0.9029 | 0.9031 |
| 2026-11-22T0356-e891 | 4 | 0.002 | None | None | False | both | 2.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 0 | b2be6aa | 0.8706 | 0.9076 | 0.9076 |
| 2026-11-22T0405-dbfc | 4 | 0.002 | None | None | False | both | 2.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 1 | b2be6aa | 0.8767 | 0.9128 | 0.9133 |
| 2026-11-22T0412-3a8a | 4 | 0.002 | None | None | False | both | 2.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | 4.0 | 2 | b2be6aa | 0.8723 | 0.9194 | 0.9198 |
| 2026-11-21T2052-5972 | 4 | 0.002 | None | None | False | both | 4.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 0 | 9a61c17 | 0.8553 | 0.9054 | 0.9053 |
| 2026-11-21T2101-0843 | 4 | 0.002 | None | None | False | both | 4.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 1 | 9a61c17 | 0.8558 | 0.9017 | 0.9017 |
| 2026-11-21T2109-a3bc | 4 | 0.002 | None | None | False | both | 4.0 | 3 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 2 | 9a61c17 | 0.8539 | 0.9003 | 0.9005 |
| 2026-11-21T2118-3d90 | 4 | 0.002 | None | None | False | both | 4.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 0 | 9a61c17 | 0.8574 | 0.8960 | 0.8961 |
| 2026-11-21T2126-288c | 4 | 0.002 | None | None | False | both | 4.0 | 5 | 0.0 | 0.0 | 0 | 0.0 | 0.0 | 0.0 | False | None | None | 1 | 9a61c17 | 0.8554 | 0.8996 | 0.8997 |
| 2026-11-20T2138-493b | 4 | 0.002 | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | 0 | 115b03b | 0.8642 | 0.9011 | 0.9011 |
| 2026-11-20T2147-7f3c | 4 | 0.002 | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | 1 | 115b03b | 0.8591 | 0.9001 | 0.9003 |
| 2026-11-20T2155-556e | 4 | 0.002 | None | None | None | None | None | None | None | None | None | None | None | None | None | None | None | 2 | 115b03b | 0.8608 | 0.8999 | 0.8999 |
| 2026-11-21T1712-ee28 | 4 | 0.002 | None | None | True | None | None | None | None | None | None | None | None | None | None | None | None | 0 | 0bc1122 | 0.8584 | 0.8992 | 0.8992 |
| 2026-11-21T1720-370b | 4 | 0.002 | None | None | True | None | None | None | None | None | None | None | None | None | None | None | None | 1 | 0bc1122 | 0.8546 | 0.8980 | 0.8979 |
| 2026-11-21T1727-50d6 | 4 | 0.002 | None | None | True | None | None | None | None | None | None | None | None | None | None | None | None | 2 | 0bc1122 | 0.8590 | 0.9027 | 0.9028 |
| 2026-11-21T2334-e151 | 4 | 0.004 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8621 | 0.9017 | 0.9016 |
| 2026-11-21T2341-56e2 | 4 | 0.004 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8624 | 0.8957 | 0.8960 |
| 2026-11-21T2347-17aa | 4 | 0.004 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8636 | 0.9059 | 0.9059 |
| 2026-11-21T2258-a495 | 6 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 0 | 7faa0e5 | 0.8713 | 0.9116 | 0.9119 |
| 2026-11-21T2310-ec06 | 6 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 1 | 7faa0e5 | 0.8699 | 0.9096 | 0.9095 |
| 2026-11-21T2320-72b3 | 6 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.999 | None | 2 | 7faa0e5 | 0.8691 | 0.9066 | 0.9062 |
| 2026-11-21T2330-1fad | 6 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.9995 | None | 0 | 7faa0e5 | 0.8725 | 0.9107 | 0.9107 |
| 2026-11-21T2341-0e5a | 6 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.9995 | None | 1 | 7faa0e5 | 0.8712 | 0.9100 | 0.9104 |
| 2026-11-21T2352-3fc1 | 6 | 0.002 | None | None | False | both | 1.0 | None | 0.0 | 0.0 | 0 | 0.0 | 1.0 | 0.0 | False | 0.9995 | None | 2 | 7faa0e5 | 0.8699 | 0.9058 | 0.9057 |
