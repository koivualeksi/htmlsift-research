WCXB-ARTICLE test -- CPU accuracy vs third-party extractors.

CPU: x86_64.
Pins: htmlsift==0.2.1 onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.

One population: the 257 pages of the fold. Metric = word-F1 over each extractor's text output (trafilatura/resiliparse native, readability flattened via html_to_text, htmlsift the package's output="text"); empty output scores 0. Regenerate: `python bench/accuracy/heuristics.py --board wcxb-article --bundle <dir> ...`.

| method | F1 | prec | rec | n | F1 domain_disjoint (n) |
|---|---|---|---|---|---|
| htmlsift 311m-table-bigru-fABC-nb1-pf4-ema0.999-r128_s2 | 0.9175 | 0.9118 | 0.9550 | 257 | 0.9212 (155) |
| trafilatura | 0.9370 | 0.9278 | 0.9670 | 257 | 0.9413 (155) |
| readability | 0.9280 | 0.9500 | 0.9312 | 257 | 0.9532 (155) |
| resiliparse | 0.8147 | 0.8061 | 0.8987 | 257 | 0.8819 (155) |

