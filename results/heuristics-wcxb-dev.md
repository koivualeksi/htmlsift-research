WCXB val -- CPU accuracy vs third-party extractors.

CPU: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD.
Pins: onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.
mini: shipped int8 ONNX (data/onnx), seed s1.

One population: the 1497 block-scored pages. Metric = word-F1 over each extractor's text output (trafilatura/resiliparse native, readability flattened via html_to_text); empty output scores 0. mini is its canonical selection score; heuristics via score_text. Regenerate: `python bench/accuracy/heuristics.py --board wcxb`.

| method | F1 | prec | rec | n |
|---|---|---|---|---|
| htmlsift (mini) | 0.8346 | 0.8251 | 0.9035 | 1497 |
| trafilatura* | 0.8132 | 0.8464 | 0.8412 | 1497 |
| readability | 0.7020 | 0.8248 | 0.6961 | 1497 |
| resiliparse* | 0.7711 | 0.7474 | 0.8757 | 1497 |

Gate: mini via score_text 0.8346 vs canonical selection 0.8346, Δ=0.0000 (PASS at 1e-3) -- score_text does not move mini's scoring.

trafilatura*: empty on 10 (scored 0) of 1497.
resiliparse*: empty on 29 (scored 0) of 1497.
