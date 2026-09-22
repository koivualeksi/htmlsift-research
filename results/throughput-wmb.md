WMB test -- GPU throughput, NVIDIA GeForce RTX 4090.

545 pages, batch<= 64, B*T cap 8192, seed-0 keeper, fp32 weights under bf16 autocast, one run per arm. Full pipeline = render+prep+model serial; model only = encoder+pool+head batched; batch-1 latency is model only.

| arm | full pipeline pg/s | model only pg/s | batch-1 p50 ms | batch-1 p90 ms | CPU render ms/pg | regenerate |
|---|---|---|---|---|---|---|
| 311m-10 | 18.4 | 23.6 | 6.7 | 73.6 | 4.5 | `python bench/speed/gpu_throughput.py --model 311m-10 --device cuda` |
| 311m-10 +band | 24.6 | 36.7 | 7.4 | 45.2 | 5.1 | `python bench/speed/gpu_throughput.py --model 311m-10 --device cuda --band` |
| 97m-6-qat | 30.2 | 51.3 | 4.2 | 37.8 | 4.4 | `python bench/speed/gpu_throughput.py --model 97m-6-qat --device cuda` |
