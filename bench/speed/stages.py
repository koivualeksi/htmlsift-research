"""WMB per-stage latency for one keeper on a chosen device: time the real inference path
html -> render -> model -> text over a fold's pages, and report per-stage latency and
pages/s. The full-path number is what the writeup compares against trafilatura / MinerU-HTML v1.1;
the model stage alone is the CPU-vs-GPU figure. Not F1 -- accuracy is
bench/accuracy/predict.py; here the output text is produced but never scored.

Render is the INFERENCE render (frontend.render_blocks: parse -> sanitize -> render_tree,
+ features when the ckpt carries them), never the labeling render: compute_main_set and the
html2text/ROUGE step are F1-only and would inflate a speed number.

    python bench/speed/stages.py --model 311m-10 --device cpu
    python bench/speed/stages.py --model 97m-6-qat --device cuda
    python bench/speed/stages.py --model 97m-table-bigru-fABC --device cpu --threads 8
"""
import argparse
import platform
import statistics
import time
from pathlib import Path

import torch

from bench.speed.frontend import build_feats_fn, render_blocks
from core.loader import build_infer, resolve_ckpt
from core.prep_page import prep_page
from vendors.shared.arms import WINDOW
from vendors.wmb.adapter import eval as wmb_eval

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", help="keeper arm id under wmb/ckpt/, e.g. 311m-10 or 97m-6-qat")
    grp.add_argument("--ckpt", help="a single ckpt (local path or repo-relative)")
    ap.add_argument("--seed", type=int, default=0, help="which seed's ckpt (--model mode)")
    ap.add_argument("--fold", default="test", choices=["test", "val", "train"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--limit", type=int, default=None, help="cap pages timed")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (cpu only)")
    ap.add_argument("--warmup", type=int, default=5, help="untimed pages first (first-call overhead)")
    ap.add_argument("--band", action="store_true",
                    help="chunked band attention (core.band, bit-identical, speed-only; encoder arms)")
    args = ap.parse_args()

    dev = torch.device(args.device)
    if args.threads and dev.type == "cpu":
        torch.set_num_threads(args.threads)
    ref = args.ckpt or f"wmb/ckpt/{args.model}_s{args.seed}.pt"
    ck = resolve_ckpt(ref, dev)
    tok, infer_fn, kind, cap = build_infer(ck, dev, band=args.band)
    feats_fn = build_feats_fn(ck)
    card = torch.cuda.get_device_name(0) if dev.type == "cuda" else (platform.processor() or "cpu")
    print(f"{ref}: kind={kind} feats={ck.get('feats') or 'none'} cap={cap} | "
          f"device={dev.type} card={card} threads={torch.get_num_threads()}", flush=True)

    pages = wmb_eval.load_pages(wmb_eval.fold_ids(args.fold))
    items = [(tid, p["html"]) for tid, p in pages.items() if p.get("html")]
    if args.limit:
        items = items[:args.limit]

    cuda = dev.type == "cuda"

    def run(tid, html):
        t0 = time.perf_counter()
        _root, blocks, _src, feats = render_blocks(html, feats_fn)
        if not blocks:
            return None
        t1 = time.perf_counter()
        page = prep_page(tok, tid, blocks, None, WINDOW, cap=cap, feats=feats)
        if cuda: torch.cuda.synchronize()
        t2 = time.perf_counter()
        infer_fn(page)
        if cuda: torch.cuda.synchronize()
        t3 = time.perf_counter()
        return len(page["members"]), (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3

    for tid, html in items[:args.warmup]:          # first-call overhead (lazy init, cuda kernels)
        run(tid, html)

    render_ms, prep_ms, model_ms, n_blocks = [], [], [], 0
    for tid, html in items:
        r = run(tid, html)
        if r is None:
            continue
        nb, rms, pms, mms = r
        render_ms.append(rms); prep_ms.append(pms); model_ms.append(mms); n_blocks += nb
    full = [r + p + m for r, p, m in zip(render_ms, prep_ms, model_ms)]
    n = len(full)

    def row(name, xs):
        return (f"| {name} | {statistics.median(xs):.1f} | {statistics.mean(xs):.1f} | "
                f"{sorted(xs)[int(0.9 * n)]:.1f} |")
    md = "\n".join([
        f"WMB {args.fold} -- per-stage latency, {card}.", "",
        f"{n} pages, {n_blocks:,} blocks ({n_blocks / n:.0f}/page), device={dev.type}, "
        f"threads={torch.get_num_threads()}. Regenerate: "
        f"`python bench/speed/stages.py --model {args.model or args.ckpt} --device {dev.type}`.", "",
        "| stage | median ms | mean ms | p90 ms |", "|---|---|---|---|",
        row("render", render_ms), row("prep", prep_ms), row("model", model_ms), row("full", full), "",
        f"full: {n / (sum(full) / 1000):.1f} pages/s sustained, "
        f"{1000 / statistics.median(full):.1f} pages/s median.",
        f"model-only: {n / (sum(model_ms) / 1000):.1f} pages/s sustained, "
        f"{n_blocks / (sum(model_ms) / 1000):,.0f} blocks/s.", ""])
    print("\n" + md)
    dest = ROOT / "results" / "stages-wmb.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md + "\n", encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
