"""Batched GPU throughput for one encoder keeper on WMB (§7 GPU phase).

bench/speed/stages.py runs batch-1: one short page barely occupies a GPU, so its pages/s
understates the card. This batches within-window pages (length-bucketed, padded,
masked) to keep the GPU busy and reports SUSTAINED pages/s -- the headline axis
against MinerU-HTML v1.1 (bench/speed/gpu_dripper.py, same card type). Two numbers side by side:
full-pipeline (CPU render + model) and model-only (encoder + pool + head), plus
batch-1 latency p50/p90.

Speed only. The masked batched forward is per-block identical to encode_window on
each page alone (up to GPU fp rounding), and any page over the window keeps the
exact per-page stitch -- but the published F1 still comes from bench/accuracy/predict.py,
never from here. fp32 encoder arms only (311m, 97m); table/QAT are the CPU story
(bench/speed/cpu_compare.py), and the encoder keepers are text-only.

    python bench/speed/gpu_throughput.py --model 311m-10 --device cuda
    python bench/speed/gpu_throughput.py --model 311m-10 --device cuda --batch 64 --max-tokens 8192
"""
import argparse
import platform
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from core.loader import resolve_ckpt
from bench.speed.frontend import render_blocks       # inference render: html -> (root, blocks, src, feats)
from core.model import build_encoder, build_head, pool, pool_page
from core.prep_page import prep_page
from vendors.shared.arms import WINDOW
from vendors.wmb.adapter import eval as wmb_eval

ROOT = Path(__file__).resolve().parents[2]


def build_encoder_head(ck, dev):
    """Raw encoder + head modules (not a per-page infer closure -- the batch needs the
    modules). Encoder arms only, run fp32: a QAT keeper's fp32 masters load unpatched (on a
    GPU int8 is no speedup -- a CPU/packaging lever, §7), so the honest GPU number is the fp32
    forward. The keepers carry no features."""
    if ck.get("kind") == "table":
        raise SystemExit("throughput is the encoder story; the table is bench/speed/cpu_compare.py")
    if ck.get("feats"):
        raise SystemExit("encoder keepers are text-only; no features on this path")
    enc = build_encoder(ck["model"], ck["layers"], device=dev)   # prints resolved dtype (§4)
    # A QAT keeper's fake-quant is in the forward, not the state -- QATW8A8Linear reuses
    # nn.Linear's params, so its fp32 masters load into a plain (unpatched) encoder. We do NOT
    # patch: this measures the deployable fp32 forward, the meaningful GPU number.
    if ck.get("qat"):
        print("note: qat keeper run as fp32 (int8 is a CPU lever; GPU measures the fp32 forward)",
              flush=True)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    head = build_head(ck.get("head", "bigru"), d_in=enc.config.hidden_size,
                      hidden=ck.get("hidden", 256)).to(dev)
    head.load_state_dict(ck["head_state"]); head.eval()
    return enc, head


def encode_batch(enc, ids_list, dev, autocast):
    """One padded, masked forward over within-window pages -> per-page hidden [n_i, H]
    fp32, pad rows dropped. attention_mask stops a real token attending to padding, so
    each page's states match core.model.encode_window run alone."""
    T = max(len(x) for x in ids_list)
    ids = torch.zeros((len(ids_list), T), dtype=torch.long, device=dev)
    mask = torch.zeros((len(ids_list), T), dtype=torch.long, device=dev)
    for r, x in enumerate(ids_list):
        ids[r, :len(x)] = torch.from_numpy(x)
        mask[r, :len(x)] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        h = enc(input_ids=ids, attention_mask=mask).last_hidden_state
    return [h[r, :len(x)].float() for r, x in enumerate(ids_list)]


def batches(pages, order, batch, max_tokens):
    """Length-sorted `order` -> batches of <= `batch` pages, capped so pages x padded-len
    (B x T) <= max_tokens. Same guard as core/frozen._batches, at the token-id level."""
    i = 0
    while i < len(order):
        idx = [order[i]]; i += 1
        while (i < len(order) and len(idx) < batch
               and (len(idx) + 1) * len(pages[order[i]]["ids"]) <= max_tokens):
            idx.append(order[i]); i += 1
        yield idx


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="encoder keeper arm under wmb/ckpt/, e.g. 311m-10")
    g.add_argument("--ckpt", help="explicit ckpt (local path or repo-relative)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", default="test", choices=["test", "val"])
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--batch", type=int, default=64, help="max pages per batch")
    ap.add_argument("--max-tokens", type=int, default=8192, help="B x T cap per batch")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--band", action="store_true",
                    help="chunked band attention on the sliding layers (core.band; bit-identical, "
                         "targets the long windowed pages that dominate sustained throughput)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the encoder with dynamic shapes")
    args = ap.parse_args()

    dev = torch.device(args.device)
    autocast = dev.type == "cuda"
    ref = args.ckpt or f"wmb/ckpt/{args.model}_s{args.seed}.pt"
    ck = resolve_ckpt(ref, dev)
    tok = AutoTokenizer.from_pretrained(ck["model"])
    enc, head = build_encoder_head(ck, dev)
    if args.band:
        import core.band
        core.band.enable(enc)                                  # bit-identical (tests/gate/test_band)
        print("band: chunked band attention enabled", flush=True)
    if args.compile:
        enc = torch.compile(enc, dynamic=True)
        print("compile: torch.compile(dynamic=True) enabled", flush=True)
    cap = ck.get("cap", 0)
    card = torch.cuda.get_device_name(0) if dev.type == "cuda" else (platform.processor() or "cpu")
    print(f"{ref}: layers={ck['layers']} head={ck.get('head', 'bigru')} cap={cap} | "
          f"device={dev.type} card={card} batch={args.batch} max_tokens={args.max_tokens}",
          flush=True)

    items = wmb_eval.load_pages(wmb_eval.fold_ids(args.fold))
    items = [(t, p["html"]) for t, p in items.items() if p.get("html")]
    if args.limit:
        items = items[:args.limit]

    # Stage 1 render (CPU, lxml) then stage 2 prep -- timed apart so both full-pipeline
    # and model-only fall out of one run. A page that renders to nothing is dropped.
    t = time.perf_counter()
    rendered = [(tid, render_blocks(h, None)[1]) for tid, h in items]
    render_ms = (time.perf_counter() - t) * 1e3
    rendered = [(tid, b) for tid, b in rendered if b]
    t = time.perf_counter()
    pages = [prep_page(tok, tid, b, None, WINDOW, cap=cap) for tid, b in rendered]
    prep_ms = (time.perf_counter() - t) * 1e3

    long_ = [i for i, p in enumerate(pages) if len(p["ids"]) > WINDOW]
    order = sorted((i for i, p in enumerate(pages) if len(p["ids"]) <= WINDOW),
                   key=lambda i: len(pages[i]["ids"]))
    plan = list(batches(pages, order, args.batch, args.max_tokens))

    @torch.no_grad()
    def model_pass():
        for idx in plan:
            for k, h in zip(idx, encode_batch(enc, [pages[k]["ids"] for k in idx], dev, autocast)):
                p = pages[k]
                head(pool(h, p["members"], 0, len(p["members"]), 0)[None])   # discarded
        for k in long_:                                        # over-window pages: per-page stitch
            head(pool_page(enc, pages[k], dev, autocast)[None])

    model_pass()                                               # warmup: cuda kernels, allocator
    if dev.type == "cuda": torch.cuda.synchronize()
    t = time.perf_counter()
    model_pass()
    if dev.type == "cuda": torch.cuda.synchronize()
    model_ms = (time.perf_counter() - t) * 1e3

    # Batch-1 latency: single-page response time (batching is a throughput lever, not a
    # latency one), model-only, over the same pages.
    @torch.no_grad()
    def one(p):
        if dev.type == "cuda": torch.cuda.synchronize()
        s = time.perf_counter()
        head(pool_page(enc, p, dev, autocast)[None])
        if dev.type == "cuda": torch.cuda.synchronize()
        return (time.perf_counter() - s) * 1e3
    for p in pages[:args.warmup]:
        one(p)
    lat = sorted(one(p) for p in pages)

    n = len(pages)
    full_pps = n / ((render_ms + prep_ms + model_ms) / 1e3)
    model_pps = n / (model_ms / 1e3)
    p50, p90 = lat[n // 2], lat[int(0.9 * n)]
    arm = f"{args.model or args.ckpt}{' +band' if args.band else ''}"
    cmd = (f"`python bench/speed/gpu_throughput.py --model {args.model or args.ckpt} "
           f"--device {dev.type}{' --band' if args.band else ''}`")
    row = (f"| {arm} | {full_pps:.1f} | {model_pps:.1f} | {p50:.1f} | {p90:.1f} | "
           f"{render_ms / n:.1f} | {cmd} |")
    dest = ROOT / "results" / "throughput-wmb.md"
    head = [f"WMB {args.fold} -- GPU throughput, {card}.", "",
            f"{n} pages, batch<= {args.batch}, B*T cap {args.max_tokens}, seed-0 keeper, fp32 "
            "weights under bf16 autocast, one run per arm. Full pipeline = render+prep+model serial; model only = "
            "encoder+pool+head batched; batch-1 latency is model only.", "",
            "| arm | full pipeline pg/s | model only pg/s | batch-1 p50 ms | batch-1 p90 ms "
            "| CPU render ms/pg | regenerate |",
            "|---|---|---|---|---|---|---|"]
    rows = {}
    if dest.exists():
        for l in dest.read_text(encoding="utf-8").splitlines():
            if l.startswith("| ") and not l.startswith("| arm"):
                rows[l.split(" | ")[0][2:]] = l
    rows[arm] = row
    md = "\n".join(head + [rows[k] for k in sorted(rows)]) + "\n"
    print("\n" + md)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
