WMB test545 -- CPU speed + F1 vs third-party extractors.

CPU: AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD | 12 logical visible | onnxruntime intra=1.
Pins: torch==2.11.0+cu126 onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.
Model: mini ONNX (311m-table-bigru-fABC s1, int8 table + BiGRU head).

Speed = extraction only (html2text not timed). F1 = the board's metric over each extractor's standard output (WMB html2texts html-kind). One population -- the 544 cmc-scored pages -- on both axes: an empty return is timed and scored (F1 0 vs non-empty gold), only a crash is excluded (asterisk). Regenerate: `python bench/speed/cpu_compare.py --onnx --model 311m-table-bigru-fABC --seed 1 --fold test --threads 1`.

| method | median ms | mean ms | p90 ms | pages/s | ×ours p50 | ×ours mean | F1 | F1 n |
|---|---|---|---|---|---|---|---|---|
| ours | 36.1 | 79.7 | 125.6 | 27.7 | 1.0× | 1.0× | 0.8994 | 544 |
| trafilatura | 31.2 | 70.8 | 121.9 | 32.0 | 0.9× | 0.9× | 0.7525 | 544 |
| readability | 19.3 | 33.7 | 62.2 | 51.7 | 0.5× | 0.4× | 0.8016 | 544 |
| resiliparse* | 1.9 | 4.5 | 9.5 | 522.5 | 0.1× | 0.1× | 0.7135 | 544 |

×ours = how many times faster ours is (method time / our time), over the pages both timed.
resiliparse*: empty on 4 (scored 0) of 544.

Size dependence (×ours on the mean, by page tokens):

| tokens | n | trafilatura | readability | resiliparse |
|---|---|---|---|---|
| <1024 | 140 | 1.3× | 0.7× | 0.1× |
| 1024-4096 | 248 | 1.2× | 0.6× | 0.1× |
| 4096-16384 | 120 | 1.0× | 0.5× | 0.1× |
| 16384-32768 | 21 | 0.9× | 0.4× | 0.1× |
| >32768 | 15 | 0.6× | 0.2× | 0.0× |

