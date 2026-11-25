WCXB test -- CPU accuracy vs third-party extractors.

CPU: x86_64.
Pins: htmlsift==0.2.1 onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.

One population: the 511 pages of the fold. Metric = word-F1 over each extractor's text output (trafilatura/resiliparse native, readability flattened via html_to_text, htmlsift the package's output="text"); empty output scores 0. Regenerate: `python bench/accuracy/heuristics.py --board wcxb --bundle <dir> ...`.

| method | F1 | prec | rec | n |
|---|---|---|---|---|
| htmlsift 311m-table-bigru-fABC-nb1-pf4-ema0.999-r128_s2* | 0.8529 | 0.8679 | 0.8913 | 511 |
| htmlsift 311m-table-bigru-fABC_s1* | 0.8528 | 0.8474 | 0.9096 | 511 |
| trafilatura* | 0.8584 | 0.8787 | 0.8786 | 511 |
| readability | 0.7653 | 0.8627 | 0.7550 | 511 |
| resiliparse* | 0.7909 | 0.7658 | 0.8926 | 511 |

htmlsift 311m-table-bigru-fABC-nb1-pf4-ema0.999-r128_s2*: empty on 9 (scored 0) of 511.
htmlsift 311m-table-bigru-fABC_s1*: empty on 8 (scored 0) of 511.
trafilatura*: empty on 3 (scored 0) of 511.
resiliparse*: empty on 10 (scored 0) of 511.
