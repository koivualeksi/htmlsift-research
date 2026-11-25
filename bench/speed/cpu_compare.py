"""CPU speed + F1 for our model vs third-party extractors on WMB (§7).

Every extractor -- ours and the baselines -- emits a STANDARD output (main-content HTML, or
text for resiliparse); the board's score_text() converts and judges it. So html2text lives in
one place (WMB.score_text), applied to everyone the same way, and it is never on the speed
clock: speed is extraction only. Our model's markdown and html output differ only by that
untimed html2text, so on WMB our speed is one number.

The mini is timed through the installed htmlsift package on a bundle directory (--bundle),
the code a user runs; a torch keeper (--model/--ckpt) through this repo's render and serialize.
Methods are timed page by page, the order rotating per page, so load drift inside a run lands
on every method alike. --memory adds each method's RAM, measured in a process of its own
(bench/speed/memory.py). The published runs are the two containers in deploy/docker/cpu_speed.sh.

All numbers are over the SHARED pages every method produced output for (refuses an empty
intersection). CPU only -- a GPU-ours-vs-CPU-theirs ratio is the §7 landmine; the GPU
comparison (MinerU-HTML v1.1, html-native) is a separate phase.

    python bench/speed/cpu_compare.py --bundle data/bundles/mini --fold test --threads 1 --memory
    python bench/speed/cpu_compare.py --model 97m-6-qat --fold val --limit 50 --threads 8
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from functools import partial
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path

from dotenv import load_dotenv

from bench.speed import competitors as ex
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


def _ver(pkg):
    for name in (pkg, pkg + "-gpu"):        # onnxruntime ships as onnxruntime or onnxruntime-gpu
        try:
            v = version(name)
        except PackageNotFoundError:
            continue
        raw = distribution(name).read_text("direct_url.json")
        vcs = json.loads(raw).get("vcs_info") if raw else None
        return f"{v}@{vcs['commit_id'][:7]}" if vcs else v
    return "?"


def _limits():
    """The cgroup v2 CPU quota and memory cap, '' outside a limited container."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        mem = Path("/sys/fs/cgroup/memory.max").read_text().strip()
    except OSError:
        return ""
    cpus = "no CPU cap" if quota == "max" else f"{int(quota) / int(period):g} vCPU"
    ram = "no memory cap" if mem == "max" else f"{int(mem) / 2**30:g} GiB"
    return f" | cgroup {cpus}, {ram}"


def _env(threads_desc):
    """Environment banner: an absolute CPU ms is only reproducible against a named CPU, a
    pinned thread count, the container limits and a pinned render/extractor stack (§4, §6), so
    the report records all of them. threads_desc names the backend that governs our timed path
    (onnxruntime for the package, torch for a keeper) and its thread count. The published
    number is still the ratio."""
    cpu = platform.processor() or "unknown"
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass                                               # not Linux (e.g. a Windows smoke)
    visible = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    pins = " ".join(f"{p}=={_ver(p)}" for p in
                    ("htmlsift", "torch", "onnxruntime", "lxml", "trafilatura", "resiliparse",
                     "readability-lxml"))
    return [f"CPU: {cpu} | {visible} logical visible{_limits()} | {threads_desc}.",
            f"Python {platform.python_version()}. Pins: {pins}.", ""]


def time_interleaved(fns, pages, warmup):
    """{name: html -> Output} over pages, page by page, the method order rotating each page.
    Warmup pages run untimed first. A raise drops that page for that method only (no valid
    output or time); an empty return is kept. Returns ({name: {tid: Output}}, notes)."""
    names = list(fns)
    items = [(tid, p["html"]) for tid, p in pages.items() if p.get("html")]
    for _, html in items[:warmup]:
        for n in names:
            try:
                fns[n](html)
            except Exception:  # noqa: BLE001
                pass
    out, raised = {n: {} for n in names}, dict.fromkeys(names, 0)
    for k, (tid, html) in enumerate(items):
        for n in names[k % len(names):] + names[:k % len(names)]:
            try:
                out[n][tid] = fns[n](html)
            except Exception:  # noqa: BLE001
                raised[n] += 1
    notes = {n: {"empty": sum(not o.text for o in out[n].values()), "raised": raised[n]}
             for n in names}
    for n in names:
        print(f"{n}: {len(out[n])}/{len(items)} timed ({notes[n]['empty']} empty, "
              f"{raised[n]} raised)", flush=True)
    return out, notes


def measure_memory(name, args):
    """One method's RAM from bench/speed/memory.py in a fresh interpreter; None if it died
    (e.g. killed at the container's memory cap)."""
    cmd = [sys.executable, "-m", "bench.speed.memory", name, "--fold", args.fold]
    if name == "htmlsift":
        cmd += ["--bundle", str(args.bundle)] + (["--threads", str(args.threads)] if args.threads else [])
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        print(f"{name}: memory probe exited {r.returncode}\n{r.stderr[-2000:]}", flush=True)
        return None
    res = json.loads(r.stdout.strip().splitlines()[-1])
    print(f"{name}: RSS after load {res['after_load_mb']:.0f} MB, worst page {res['peak_mb']:.0f} MB",
          flush=True)
    return res


def report(methods, board, fold, tokens, pop, env, scorable, notes, regen="", ram=None):
    """methods: name -> {tid: Output} (empties included). Every method is scored over one
    population -- the cmc-scored pages in `scorable` -- on both speed and F1: an empty return is
    timed and scores as an empty prediction, only a crash is excluded. notes[name] carries the
    empty/crash counts; a method with either is asterisked. 'htmlsift' is the ratio baseline.
    ram: name -> memory.py result (None where the probe died), or None for no RAM columns."""
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
           "Speed = extraction only (html2text not timed), methods timed page by page in rotating "
           "order. F1 = the board's metric over each "
           "extractor's standard output (WMB html2texts html-kind). One population -- the "
           f"{len(scorable)} cmc-scored pages -- on both axes: an empty return is timed and scored "
           "(F1 0 vs non-empty gold), only a crash is excluded (asterisk). Regenerate: "
           f"{regen}.", ""]

    header = ["method", "median ms", "mean ms", "p90 ms", "pages/s", "×htmlsift p50", "×htmlsift mean", "F1", "F1 n"]
    if ram is not None:
        header += ["RSS loaded MB", "worst page MB"]
    out += ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for name in names:
        outputs, cov = methods[name], covered[name]
        label = name + ("*" if name in flagged else "")
        mem = []
        if ram is not None:
            r = ram.get(name)
            mem = [f"{r['after_load_mb']:.0f}", f"{r['peak_mb']:.0f}"] if r else ["died", "died"]
        if not cov:
            out.append("| " + " | ".join([label] + ["-"] * 6 + ["0.0000", "0"] + mem) + " |")
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
                    f"{sc['f1']:.4f}", str(sc['n'])] + mem) + " |")
    out += ["", "×htmlsift = how many times faster htmlsift is (method time / our time), over the pages "
            "both timed."]
    if ram is not None:
        out.append("RSS loaded = resident memory of a process holding only that method, after "
                   "loading it and running one page; worst page = the largest single-page peak "
                   "RSS over the fold, in the same process.")
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
    ap.add_argument("--bundle", type=Path,
                    help="mini bundle dir: time the installed htmlsift package on it (the shipped path)")
    ap.add_argument("--model", default="311m-10", help="keeper arm under wmb/ckpt/ (no --bundle)")
    ap.add_argument("--ckpt", help="explicit ckpt (local path or repo-relative), overrides --model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", default="test", choices=["test", "val"])
    ap.add_argument("--extractors", default=",".join(ex.EXTRACTORS),
                    help="comma list; default all registered")
    ap.add_argument("--limit", type=int, default=None, help="cap pages (smoke)")
    ap.add_argument("--threads", type=int, default=None,
                    help="onnxruntime intra-op threads (--bundle) or torch CPU threads (keeper)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--memory", action="store_true",
                    help="add each method's RAM, one process per method (Linux)")
    ap.add_argument("--out", type=Path, help="report path; default results/speed-wmb[-<model>].md")
    ap.add_argument("--device", default="cpu", help="cpu only; a GPU ratio is the §7 landmine")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("cpu_compare.py is CPU-only; the GPU comparison (MinerU-HTML v1.1) is a separate phase")
    if args.memory and not args.bundle:
        raise SystemExit("--memory measures the package mini (--bundle)")

    pages = wmb_eval.load_pages(wmb_eval.fold_ids(args.fold))
    if args.limit:
        pages = dict(list(pages.items())[:args.limit])

    if args.bundle:
        from bench.speed.package import PackageExtractor
        model = PackageExtractor(str(args.bundle), args.threads)
        src = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8")).get("source", {})
        artifact = f"htmlsift package on bundle {args.bundle.name} ({src.get('keeper', 'keeper unknown')})"
        threads_desc = f"onnxruntime intra={args.threads or 'default'}"
    else:
        import torch

        from bench.speed.keeper import HtmlsiftExtractor
        if args.threads:
            torch.set_num_interop_threads(1)           # pin inter-op too, before any torch work
        ref = args.ckpt or f"wmb/ckpt/{args.model}_s{args.seed}.pt"
        model = HtmlsiftExtractor(ref, args.device, args.threads)
        artifact = f"torch keeper {args.model} s{args.seed}"
        threads_desc = f"torch intra={torch.get_num_threads()} inter={torch.get_num_interop_threads()}"
    fns = {"htmlsift": model.extract}
    for name in args.extractors.split(","):
        fns[name] = partial(ex.time_one, ex.EXTRACTORS[name])
    methods, notes = time_interleaved(fns, pages, args.warmup)
    tokens = {tid: model.tokens(pages[tid]["html"]) for tid in methods["htmlsift"]}
    ram = {n: measure_memory(n, args) for n in methods} if args.memory else None

    scorable = {tid for tid, p in pages.items() if p.get("cmc") is not None}
    env = _env(threads_desc)
    env.insert(2, f"Model: {artifact}.")
    regen = "`python bench/speed/cpu_compare.py" + "".join(f" {p}" for p in (
        f"--bundle {args.bundle.as_posix()}" if args.bundle else
        (f"--ckpt {args.ckpt}" if args.ckpt else f"--model {args.model}"),
        f"--seed {args.seed}" if args.seed and not args.bundle else "",
        f"--fold {args.fold}",
        f"--threads {args.threads}" if args.threads else "",
        "--memory" if args.memory else "") if p) + "`"
    md = report(methods, WMBBenchmark(), args.fold, tokens, wmb_eval.POP[args.fold], env, scorable,
                notes, regen, ram)
    print("\n" + md)
    dest = args.out or ROOT / "results" / ("speed-wmb.md" if args.bundle else f"speed-wmb-{args.model}.md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md + "\n", encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
