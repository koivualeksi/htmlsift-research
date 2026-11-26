"""RAM of one extractor, in a process that holds nothing else: RSS once it is loaded and has
run one page, and the largest single-page peak over the fold. Linux only (/proc). cpu_compare.py
--memory runs it once per method, so no extractor's figure carries another's libraries.

Pages stream from wmb.jsonl one at a time, read without the WMB scorer's imports, so the
process holds one page and the extractor. The peak is reset before each page
(/proc/self/clear_refs 5), so each page's peak is its own, not the run's high-water mark.

    python -m bench.speed.memory trafilatura --fold test
    python -m bench.speed.memory htmlsift --bundle data/bundles/mini --threads 1
"""
import argparse
import json
from pathlib import Path

from bench.speed import competitors as ex

DATA = Path(__file__).resolve().parents[2] / "vendors" / "wmb" / "data"


def _mb(key):
    with open("/proc/self/status", encoding="ascii") as f:
        for line in f:
            if line.startswith(key):
                return int(line.split()[1]) / 1024
    raise KeyError(key)


def _pages(fold, limit):
    ids = {t for t, v in json.loads((DATA / "splits.json").read_text("utf-8")).items()
           if v["fold"] == fold}
    n = 0
    with open(DATA / "wmb.jsonl", encoding="utf-8") as f:
        for line in f:
            if limit is not None and n == limit:
                return
            r = json.loads(line)
            if r["track_id"] in ids and r.get("html"):
                n += 1
                yield r["html"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", help="htmlsift or a competitors.EXTRACTORS name")
    ap.add_argument("--bundle", help="mini bundle dir (htmlsift)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--fold", default="test")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.method == "htmlsift":
        from bench.speed.package import PackageExtractor
        fn = PackageExtractor(args.bundle, args.threads).extract
    else:
        fn = ex.EXTRACTORS[args.method].fn

    fn(next(_pages(args.fold, 1)))
    after_load = _mb("VmRSS")
    peak, n = after_load, 0
    for html in _pages(args.fold, args.limit):
        Path("/proc/self/clear_refs").write_text("5")
        try:
            fn(html)
        except Exception:  # noqa: BLE001 -- a raise is excluded from timing too
            continue
        peak = max(peak, _mb("VmHWM"))
        n += 1
    print(json.dumps({"method": args.method, "after_load_mb": after_load, "peak_mb": peak,
                      "pages": n}))


if __name__ == "__main__":
    main()
