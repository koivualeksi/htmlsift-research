WMB test -- MinerU-HTML v1.1 (HunYuan 0.5B, the Dripper lineage), NVIDIA GeForce RTX 4090.

545 pages, 0 misses. Their markdown conversion off the clock (main_html, output_format=none). Regenerate: `python bench/speed/gpu_dripper.py --device cuda`.

| metric | value |
|---|---|
| batched throughput | 1.16 pages/s |
| batch-1 latency | p50 844 ms, p90 3343 ms |
| F1 (WMB ROUGE-5) | 0.9306 (n=544) |

Full pipeline only: HTML->DOM->decode->HTML is one call (not split). Compare the batched row against bench/speed/gpu_throughput.py's full-pipeline row on the same card type.

