WMB test545 -- CPU speed + F1 vs third-party extractors.

CPU: Intel(R) Core(TM) i7-6820HQ CPU @ 2.70GHz | 2 logical visible | cgroup 2 vCPU, 4 GiB | onnxruntime intra=1.
Python 3.13.18. Pins: htmlsift==0.2.1@7e3d05a torch==2.11.0+cpu onnxruntime==1.22.0 lxml==6.1.1 trafilatura==2.2.0 resiliparse==1.0.9 readability-lxml==0.9.
Model: htmlsift package on bundle bundle (wmb/ckpt/311m-table-bigru-fABC-nb1-pf4-ema0.999-r128_s2.pt).

Speed = extraction only (html2text not timed), methods timed page by page in rotating order. F1 = the board's metric over each extractor's standard output (WMB html2texts html-kind). One population -- the 544 cmc-scored pages -- on both axes: an empty return is timed and scored (F1 0 vs non-empty gold), only a crash is excluded (asterisk). Regenerate: `python bench/speed/cpu_compare.py --bundle /bundle --fold test --threads 1 --memory`.

| method | median ms | mean ms | p90 ms | pages/s | ×htmlsift p50 | ×htmlsift mean | F1 | F1 n | RSS loaded MB | worst page MB |
|---|---|---|---|---|---|---|---|---|---|---|
| htmlsift* | 32.9 | 69.8 | 122.7 | 30.4 | 1.0× | 1.0× | 0.9114 | 544 | 401 | 576 |
| trafilatura | 44.0 | 99.6 | 189.4 | 22.7 | 1.3× | 1.4× | 0.7525 | 544 | 48 | 285 |
| readability | 31.1 | 57.1 | 113.9 | 32.2 | 0.9× | 0.8× | 0.8016 | 544 | 34 | 134 |
| resiliparse* | 2.2 | 5.5 | 8.1 | 453.6 | 0.1× | 0.1× | 0.7135 | 544 | 26 | 87 |

×htmlsift = how many times faster htmlsift is (method time / our time), over the pages both timed.
RSS loaded = resident memory of a process holding only that method, after loading it and running one page; worst page = the largest single-page peak RSS over the fold, in the same process.
htmlsift*: empty on 1 (scored 0) of 544.
resiliparse*: empty on 4 (scored 0) of 544.

Size dependence (×htmlsift on the mean, by page tokens):

| tokens | n | trafilatura | readability | resiliparse |
|---|---|---|---|---|
| <1024 | 140 | 2.2× | 1.3× | 0.1× |
| 1024-4096 | 248 | 1.8× | 1.1× | 0.1× |
| 4096-16384 | 120 | 1.5× | 1.0× | 0.1× |
| 16384-32768 | 21 | 1.4× | 0.8× | 0.1× |
| >32768 | 15 | 0.9× | 0.4× | 0.1× |

