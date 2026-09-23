WCXB test -- CPU accuracy vs third-party extractors.

CPU: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD.
Pins: onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.
mini: shipped int8 ONNX (data/onnx), seed s1.

One population: the 511 block-scored pages. Metric = word-F1 over each extractor's text output (trafilatura/resiliparse native, readability flattened via html_to_text); empty output scores 0. mini is its canonical selection score; heuristics via score_text. Regenerate: `python bench/accuracy/heuristics.py --board wcxb`.

| method | F1 | prec | rec | n |
|---|---|---|---|---|
| htmlsift (mini) | 0.8512 | 0.8437 | 0.9100 | 511 |
| trafilatura* | 0.8584 | 0.8787 | 0.8786 | 511 |
| readability | 0.7653 | 0.8627 | 0.7550 | 511 |
| resiliparse* | 0.7909 | 0.7658 | 0.8926 | 511 |

Gate: mini via score_text 0.8512 vs canonical selection 0.8512, Δ=0.0000 (PASS at 1e-3) -- score_text does not move mini's scoring.

trafilatura*: empty on 3 (scored 0) of 511.
resiliparse*: empty on 10 (scored 0) of 511.
