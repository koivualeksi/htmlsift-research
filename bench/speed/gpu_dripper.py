"""MinerU-HTML v1.1 (the Dripper lineage) as a WMB competitor on the GPU (§7 phase).

MinerU-HTML is the WMB team's own extractor -- a small LM decoding labels over a simplified DOM,
served by vLLM, emitting main-content HTML natively. We run their current board entry, v1.1
(HunYuan 0.5B, 0.9001 on their full-set board), not the original Dripper 0.6B (Qwen3, 0.8779):
Dripper 0.6B's constrained decode needs vLLM's V0 state machine, and under the V1 engine that
vLLM 0.11.1 boots it never closes -- ~12k output tokens per page against a 16k cap, a 4090 run
that would not finish inside the watchdog. We run it under OUR harness on OUR fold with OUR
scorer: their published F1 is a different metric and population and is NOT comparable (root
CLAUDE.md §6/§7). Its main_html reaches WMB.score_text as the same (text, "html") every
extractor does, so it is judged exactly like trafilatura and like us.

The package is `mineru_html` (PyPI mineru-html); `MinerUHTML` is its vLLM implementation -- the
HF model card's `dripper.api.Dripper` does not exist in the shipped package. output_format=
"none" makes process() return the extracted main_html and SKIP Dripper's own markdown
conversion (its html2text analogue) -- the like-for-like html-mode, that step off the clock,
symmetric with ours. use_fall_back="trafilatura" is Dripper's shipped behaviour for pages it
can't handle.

Its own pod (deploy/pod/pod_dripper.sh, route B): vLLM's image owns torch 2.9 + vLLM 0.11.1 + CUDA;
this imports only the torch-free WMB scorer, never our model. Batched throughput (process()
takes a list; vLLM batches internally) is the headline vs our GPU throughput.

    python bench/speed/gpu_dripper.py --device cuda
    python bench/speed/gpu_dripper.py --device cuda --fold val --limit 50
"""
import argparse
import time
from pathlib import Path

from vendors.wmb.adapter import eval as wmb_eval
from vendors.wmb.adapter.benchmark import WMBBenchmark

ROOT = Path(__file__).resolve().parents[2]

MODEL = "opendatalab/MinerU-HTML-v1.1-hunyuan0.5B-compact"
# Their MinerUHTML wrapper hardcodes 262,144, which is 24 GiB of KV cache for one sequence and
# does not fit a 4090. Exactly 2 of the 545 test pages exceed either limit (343k and 690k prompt
# tokens), so the trafilatura-fallback set is identical to their default.
MAX_CTX = 131072


def resolve_model(model_path):
    if model_path:
        return model_path
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL)


def main_html(r):
    """The extracted main-content HTML of one result, or None on a miss/error."""
    od = getattr(r, "output_data", None)
    return getattr(od, "main_html", None) if od else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", help=f"local weights dir (default: pull {MODEL})")
    ap.add_argument("--fold", default="test", choices=["test", "val"])
    ap.add_argument("--device", default="cuda", help="cuda; vLLM has no CPU path here")
    ap.add_argument("--limit", type=int, default=None, help="cap pages (smoke)")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    import torch
    from mineru_html import MinerUHTMLConfig, MinerUHTMLGeneric, create_vllm_backend
    card = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # Factory route, not the MinerUHTML wrapper, only so max_context_window can be MAX_CTX (above);
    # every other setting is the wrapper's own default. Pages still over it after their
    # DOM-simplification take the trafilatura fallback (their shipped path).
    config = MinerUHTMLConfig(use_fall_back="trafilatura", early_load=True, output_format="none")
    llm = create_vllm_backend(
        model_path=resolve_model(args.model_path),
        response_format=config.response_format,
        max_context_window=MAX_CTX,
        model_init_kwargs={"tensor_parallel_size": 1},
    )
    extractor = MinerUHTMLGeneric(llm=llm, config=config)

    pages = wmb_eval.load_pages(wmb_eval.fold_ids(args.fold))
    items = [(t, p["html"]) for t, p in pages.items() if p.get("html")]
    if args.limit:
        items = items[:args.limit]
    tids = [t for t, _ in items]
    htmls = [h for _, h in items]
    n = len(items)

    extractor.process(htmls[:args.warmup])                     # warmup: weights hot, cuda graphs

    t = time.perf_counter()
    results = extractor.process(htmls)                         # vLLM batches internally
    wall = time.perf_counter() - t
    produced = {tid: mh for tid, r in zip(tids, results) if (mh := main_html(r))}

    # Batch-1 latency: single-request response time, one process() call per page.
    lat = []
    for h in htmls:
        s = time.perf_counter()
        extractor.process([h])
        lat.append((time.perf_counter() - s) * 1e3)
    lat.sort()

    extractor.llm.cleanup()

    sc = WMBBenchmark().score_text({tid: (mh, "html") for tid, mh in produced.items()}, args.fold)

    p50, p90 = lat[n // 2], lat[int(0.9 * n)]
    md = "\n".join([
        f"WMB {args.fold} -- MinerU-HTML v1.1 (HunYuan 0.5B, the Dripper lineage), {card}.", "",
        f"{n} pages, {n - len(produced)} misses. Their markdown conversion off the clock "
        f"(main_html, output_format=none). Regenerate: `python bench/speed/gpu_dripper.py --device cuda`.", "",
        "| metric | value |", "|---|---|",
        f"| batched throughput | {n / wall:.2f} pages/s |",
        f"| batch-1 latency | p50 {p50:.0f} ms, p90 {p90:.0f} ms |",
        f"| F1 (WMB ROUGE-5) | {sc['f1']:.4f} (n={sc['n']}) |", "",
        "Full pipeline only: HTML->DOM->decode->HTML is one call (not split). Compare the "
        "batched row against bench/speed/gpu_throughput.py's full-pipeline row on the same card type.", ""])
    print("\n" + md)
    dest = ROOT / "results" / "dripper-wmb.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md + "\n", encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
