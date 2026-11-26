DANIEL test -- CPU accuracy vs third-party extractors.

CPU: x86_64.
Pins: htmlsift==0.2.1 onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.

One population: the 1689 pages of the fold. Metric = ROUGE-L (macro over 5 languages) over each extractor's text output (trafilatura/resiliparse native, readability flattened via html_to_text, htmlsift the package's output="text"); empty output scores 0. Regenerate: `python bench/accuracy/heuristics.py --board daniel --bundle <dir> ...`.

| method | F1 | prec | rec | n |
|---|---|---|---|---|
| htmlsift 311m-table-bigru-fABC-nb1-pf4-ema0.999-r128_s2* | 0.8925 | 0.8715 | 0.9527 | 1689 |
| htmlsift 311m-table-bigru-fABC_s1* | 0.8818 | 0.8475 | 0.9639 | 1689 |
| trafilatura | 0.8255 | 0.7855 | 0.9340 | 1689 |
| readability | 0.8904 | 0.8942 | 0.9059 | 1689 |
| resiliparse* | 0.7094 | 0.6018 | 0.9709 | 1689 |

htmlsift 311m-table-bigru-fABC-nb1-pf4-ema0.999-r128_s2*: empty on 2 (scored 0) of 1689.
htmlsift 311m-table-bigru-fABC_s1*: empty on 3 (scored 0) of 1689.
resiliparse*: empty on 1 (scored 0) of 1689.
