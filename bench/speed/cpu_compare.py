"""CPU speed + F1 for our model vs third-party extractors on WMB (§7).

Every extractor -- ours and the baselines -- emits a STANDARD output (main-content HTML, or
text for resiliparse); the board's score_text() converts and judges it. So html2text lives in
one place (WMB.score_text), applied to everyone the same way, and it is never on the speed
clock: speed is extraction only. Our model's markdown and html output differ only by that
untimed html2text, so on WMB our speed is one number.

All numbers are over the SHARED pages every method produced output for (refuses an empty
intersection). CPU only -- a GPU-ours-vs-CPU-theirs ratio is the §7 landmine; the GPU
comparison (MinerU-HTML v1.1, html-native) is a separate phase.

    python bench/speed/cpu_compare.py --model 311m-10 --fold test
    python bench/speed/cpu_compare.py --model 97m-6-qat --fold val --limit 50 --threads 8
"""
import argparse
import os
import platform
import statistics
from pathlib import Path

import torch
from dotenv import load_dotenv

from bench.speed import competitors as ex
from bench.speed.htmlsift import HtmlsiftExtractor, HtmlsiftOnnxExtractor
from export.paths import mini_paths
from vendors.wmb.adapter import eval as wmb_eval
from vendors.wmb.adapter.benchmark import WMBBenchmark

ROOT = Path(__file__).resolve().parents[2]
BUCKETS = [1024, 4096, 16384, 32768]                  # upper edges; last is > 32768
BUCKET_LABELS = ["<1024", "1024-4096", "4096-16384", "16384-32768", ">32768"]


def _bucket(n):
    for edge, label in zip(BUCKETS, BUCKET_LABELS):
        if n < edge:
            return label
    return BUCKET_LABELS[-1]


def _speed(outputs, tids):
    xs = sorted(outputs[t].extract_ms for t in tids)
    n = len(xs)
    return {"median": statistics.median(xs), "mean": statistics.mean(xs),
            "p90": xs[int(0.9 * n)], "pps": 1000 / statistics.median(xs)}


def _env(threads_desc):
    """Environment banner: an absolute CPU ms is only reproducible against a named CPU, a
    pinned thread count and a pinned render/extractor stack (§4, §6), so the report records
    all three. threads_desc names the backend that governs our timed path (onnxruntime for
    --onnx, torch otherwise) and its thread count. The published number is still the ratio."""
    from importlib.metadata import PackageNotFoundError, version

    cpu = platform.processor() or "unknown"
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass                                               # not Linux (e.g. a Windows smoke)
    visible = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

    def ver(pkg):
        for name in (pkg, pkg + "-gpu"):        # onnxruntime ships as onnxruntime or onnxruntime-gpu
            try:
                return version(name)
            except PackageNotFoundError:
                continue
        return "?"
    pins = " ".join(f"{p}=={ver(p)}" for p in
                    ("torch", "onnxruntime", "lxml", "trafilatura", "resiliparse", "readability-lxml"))
    return [f"CPU: {cpu} | {visible} logical visible | {threads_desc}.",
            f"Pins: {pins}.", ""]


def report(methods, board, fold, tokens, pop, env, scorable, notes, regen=""):
    """methods: name -> {tid: Output} (empties included). Every method is scored over one
    population -- the cmc-scored pages in `scorable` -- on both speed and F1: an empty return is
    timed and scores as an empty prediction, only a crash is excluded. notes[name] carries the
    empty/crash counts; a method with either is asterisked. 'htmlsift' is the ratio baseline."""
    names = ["htmlsift"] + [n for n in methods if n != "htmlsift"]
    covered = {n: set(methods[n]) & scorable for n in names}
    if not covered["htmlsift"]:
        raise SystemExit("htmlsift produced no scorable page -- nothing to compare")
    flagged = {n for n in names if notes[n]["empty"] or notes[n]["raised"]}
    ocov = covered["htmlsift"]
    # §7/§4: a speed ratio is only meaningful over pages both methods timed. A method
    # that produced output but shares no page with ours (disjoint, not merely empty)
    # would ratio over an empty set -- refuse rather than emit a cross-population number.
    for n in names:
        if n != "htmlsift" and covered[n] and not (covered[n] & ocov):
            raise SystemExit(f"{n} shares no timed page with htmlsift -- "
                             "refusing a cross-population speed ratio")

    out = [f"WMB {pop} -- CPU speed + F1 vs third-party extractors.", "", *env,
           "Speed = extraction only (html2text not timed). F1 = the board's metric over each "
           "extractor's standard output (WMB html2texts html-kind). One population -- the "
           f"{len(scorable)} cmc-scored pages -- on both axes: an empty return is timed and scored "
           "(F1 0 vs non-empty gold), only a crash is excluded (asterisk). Regenerate: "
           f"{regen}.", ""]

    header = ["method", "median ms", "mean ms", "p90 ms", "pages/s", "×htmlsift p50", "×htmlsift mean", "F1", "F1 n"]
    out += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for name in names:
        outputs, cov = methods[name], covered[name]
        label = name + ("*" if name in flagged else "")
        if not cov:
            out.append("| " + " | ".join([label] + ["-"] * 6 + ["0.0000", "0"]) + " |")
            continue
        s = _speed(outputs, cov)
        sc = board.score_text({t: (outputs[t].text, outputs[t].kind) for t in cov}, fold)
        both = ocov & cov
        rp = statistics.median(outputs[t].extract_ms for t in both) / \
            statistics.median(methods["htmlsift"][t].extract_ms for t in both)
        rm = statistics.mean(outputs[t].extract_ms for t in both) / \
            statistics.mean(methods["htmlsift"][t].extract_ms for t in both)
        out.append("| " + " | ".join([label, f"{s['median']:.1f}", f"{s['mean']:.1f}",
                    f"{s['p90']:.1f}", f"{s['pps']:.1f}", f"{rp:.1f}×", f"{rm:.1f}×",
                    f"{sc['f1']:.4f}", str(sc['n'])]) + " |")
    out += ["", "×htmlsift = how many times faster htmlsift is (method time / our time), over the pages "
            "both timed."]
    for name in names:
        e, r = notes[name]["empty"], notes[name]["raised"]
        if e or r:
            bits = ([f"empty on {e} (scored 0)"] if e else []) + ([f"crashed on {r} (excluded)"] if r else [])
            out.append(f"{name}*: {', '.join(bits)} of {len(scorable)}.")
    out.append("")

    # size dependence: ratio htmlsift-vs-each on the mean, per token bucket (htmlsift's scored pages)
    by_bucket = {lab: [t for t in ocov if _bucket(tokens[t]) == lab] for lab in BUCKET_LABELS}
    third = [n for n in names if n != "htmlsift"]
    out += ["Size dependence (×htmlsift on the mean, by page tokens):", "",
            "| tokens | n | " + " | ".join(third) + " |",
            "|" + "|".join(["---"] * (len(third) + 2)) + "|"]
    for lab in BUCKET_LABELS:
        tids = by_bucket[lab]
        if not tids:
            continue
        cells = []
        for n in third:
            pair = [t for t in tids if t in covered[n]]
            om = statistics.mean(methods["htmlsift"][t].extract_ms for t in pair) if pair else 0
            cells.append(f"{statistics.mean(methods[n][t].extract_ms for t in pair) / om:.1f}×" if pair else "-")
        out.append("| " + " | ".join([lab, str(len(tids))] + cells) + " |")
    out.append("")
    return "\n".join(out)


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="311m-10", help="keeper arm under wmb/ckpt/")
    ap.add_argument("--ckpt", help="explicit ckpt (local path or repo-relative), overrides --model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", default="test", choices=["test", "val"])
    ap.add_argument("--extractors", default=",".join(ex.EXTRACTORS),
                    help="comma list; default all registered")
    ap.add_argument("--limit", type=int, default=None, help="cap pages (smoke)")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--onnx", action="store_true",
                    help="bench the mini ONNX graphs in data/onnx/ instead of a torch keeper")
    ap.add_argument("--device", default="cpu", help="cpu only; a GPU ratio is the §7 landmine")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("cpu_compare.py is CPU-only; the GPU comparison (MinerU-HTML v1.1) is a separate phase")

    pages = wmb_eval.load_pages(wmb_eval.fold_ids(args.fold))
    if args.limit:
        pages = dict(list(pages.items())[:args.limit])

    if args.threads:
        torch.set_num_interop_threads(1)               # pin inter-op too, before any torch work
    ref = args.ckpt or f"wmb/ckpt/{args.model}_s{args.seed}.pt"
    if args.onnx:
        table_p, head_p = mini_paths(ROOT / "data" / "onnx")
        model = HtmlsiftOnnxExtractor(ref, table_p, head_p, args.threads)
        artifact = f"mini ONNX ({args.model} s{args.seed}, int8 table + BiGRU head)"
    else:
        model = HtmlsiftExtractor(ref, args.device, args.threads)
        artifact = f"torch keeper {args.model} s{args.seed}"
    ours, tokens, ours_notes = model.run(pages, args.warmup)
    methods, notes = {"htmlsift": ours}, {"htmlsift": ours_notes}
    for name in args.extractors.split(","):
        methods[name], notes[name] = ex.run(name, pages, args.warmup)

    scorable = {tid for tid, p in pages.items() if p.get("cmc") is not None}
    threads_desc = (f"onnxruntime intra={args.threads or 'default'}" if args.onnx
                    else f"torch intra={torch.get_num_threads()} inter={torch.get_num_interop_threads()}")
    env = _env(threads_desc)
    env.insert(2, f"Model: {artifact}.")
    regen = "`python bench/speed/cpu_compare.py" + "".join(f" {p}" for p in (
        "--onnx" if args.onnx else "",
        f"--ckpt {args.ckpt}" if args.ckpt else f"--model {args.model}",
        f"--seed {args.seed}" if args.seed else "",
        f"--fold {args.fold}",
        f"--threads {args.threads}" if args.threads else "") if p) + "`"
    md = report(methods, WMBBenchmark(), args.fold, tokens, wmb_eval.POP[args.fold], env, scorable, notes, regen)
    print("\n" + md)
    dest = ROOT / "results" / ("speed-wmb.md" if args.onnx else f"speed-wmb-{args.model}.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md + "\n", encoding="utf-8")
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
