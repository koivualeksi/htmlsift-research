DANIEL test -- CPU accuracy vs third-party extractors.

CPU: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD.
Pins: onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.
mini: shipped int8 ONNX (data/onnx), seed s1.

One population: the 1689 block-scored pages. Metric = ROUGE-L (macro over 5 languages) over each extractor's text output (trafilatura/resiliparse native, readability flattened via html_to_text); empty output scores 0. mini is its canonical selection score; heuristics via score_text. Regenerate: `python bench/accuracy/heuristics.py --board daniel`.

| method | F1 | prec | rec | n |
|---|---|---|---|---|
| htmlsift (mini) | 0.8825 | 0.8479 | 0.9640 | 1689 |
| trafilatura | 0.8265 | 0.7866 | 0.9339 | 1689 |
| readability | 0.8925 | 0.8967 | 0.9072 | 1689 |
| resiliparse* | 0.7094 | 0.6018 | 0.9709 | 1689 |

Gate: mini via score_text 0.8825 vs canonical selection 0.8825, Δ=0.0000 (PASS at 1e-3) -- score_text does not move mini's scoring.

resiliparse*: empty on 1 (scored 0) of 1689.
