WCXB-ARTICLE val -- CPU accuracy vs third-party extractors.

CPU: x86_64.
Pins: htmlsift==0.2.1 onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.

One population: the 200 pages of the fold. Metric = word-F1 over each extractor's text output (trafilatura/resiliparse native, readability flattened via html_to_text, htmlsift the package's output="text"); empty output scores 0. Regenerate: `python bench/accuracy/heuristics.py --board wcxb-article --bundle <dir> ...`.

| method | F1 | prec | rec | n |
|---|---|---|---|---|
| htmlsift 311m-table-bigru-fABC-nb1-pf4-ema0.999-r128_s2* | 0.9279 | 0.9139 | 0.9650 | 200 |
| trafilatura | 0.9284 | 0.9388 | 0.9400 | 200 |
| readability | 0.9046 | 0.9408 | 0.9156 | 200 |
| resiliparse | 0.8875 | 0.8444 | 0.9627 | 200 |

htmlsift 311m-table-bigru-fABC-nb1-pf4-ema0.999-r128_s2*: empty on 1 (scored 0) of 200.
