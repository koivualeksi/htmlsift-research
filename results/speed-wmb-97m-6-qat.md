WMB test545 -- CPU speed + F1 vs third-party extractors.

CPU: AMD Ryzen 5 3600 6-Core Processor | 2 logical visible | torch intra=2 inter=1.
Pins: torch==2.11.0+cpu onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.
Model: torch keeper 97m-6-qat s1.

Speed = extraction only (html2text not timed). F1 = the board's metric over each extractor's standard output (WMB html2texts html-kind). One population -- the 544 cmc-scored pages -- on both axes: an empty return is timed and scored (F1 0 vs non-empty gold), only a crash is excluded (asterisk). Regenerate: `python bench/speed/cpu_compare.py --model 97m-6-qat --seed 1 --fold test --threads 2`.

| method | median ms | mean ms | p90 ms | pages/s | ×htmlsift p50 | ×htmlsift mean | F1 | F1 n |
|---|---|---|---|---|---|---|---|---|
| htmlsift | 1120.0 | 11883.6 | 25136.2 | 0.9 | 1.0× | 1.0× | 0.9289 | 544 |
| trafilatura | 40.3 | 89.7 | 163.8 | 24.8 | 0.0× | 0.0× | 0.7525 | 544 |
| readability | 25.9 | 45.0 | 90.8 | 38.7 | 0.0× | 0.0× | 0.8016 | 544 |
| resiliparse* | 1.7 | 3.8 | 6.1 | 596.5 | 0.0× | 0.0× | 0.7135 | 544 |

×htmlsift = how many times faster htmlsift is (method time / our time), over the pages both timed.
resiliparse*: empty on 4 (scored 0) of 544.

Size dependence (×htmlsift on the mean, by page tokens):

| tokens | n | trafilatura | readability | resiliparse |
|---|---|---|---|---|
| <1024 | 147 | 0.1× | 0.0× | 0.0× |
| 1024-4096 | 246 | 0.0× | 0.0× | 0.0× |
| 4096-16384 | 119 | 0.0× | 0.0× | 0.0× |
| 16384-32768 | 18 | 0.0× | 0.0× | 0.0× |
| >32768 | 14 | 0.0× | 0.0× | 0.0× |

