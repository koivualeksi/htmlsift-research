"""CPU accuracy: our mini vs third-party extractors on WCXB / DAnIEL (the boards outside
the WMB training policy). Accuracy only -- no speed columns: §7 keeps speed claims to WMB
(bench/speed/), and a cross-board speed ratio would reopen the population landmine.

Every extractor is judged by the board's score_text() on its best TEXT output: trafilatura
and resiliparse emit native text (as the WCXB leaderboard scores them); readability has no
text mode, so its html is flattened with html_to_text. htmlsift is the installed package
on a bundle directory, Extractor(bundle_dir).extract(output="text"), the code a user runs;
one row per --bundle, labelled by the keeper its manifest names.

Population = every page of the fold; every method is scored over that one set, an empty
return scoring 0.

    python bench/accuracy/heuristics.py --board wcxb --bundle data/bundles/mini --bundle <0.2.x bundle>
    python bench/accuracy/heuristics.py --board daniel --limit 20      # smoke
"""
import argparse
import json
import platform
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from htmlsift import Extractor

from bench.speed import competitors as ex
from vendors.shared.benchmark import load_benchmark

ROOT = Path(__file__).resolve().parents[2]
METRIC = {"wcxb": "word-F1", "daniel": "ROUGE-L (macro over 5 languages)"}
GOLD = {"wcxb": "main_content", "daniel": "reference"}


def _headline(bench, board, outputs, fold):
    """The board's headline over an extractor's outputs. WCXB: per-record word-F1. DAnIEL: the
    macro over its five languages -- the repo's published DAnIEL lens (matches the generalization
    ledger) -- by grouping outputs by language and meaning the per-language score_text. Returns
    {n, prec, rec, f1}; the vendored metric inside score_text is untouched, this only aggregates."""
    if board != "daniel":
        return bench.score_text(outputs, fold)
    rows = bench._fold_rows(fold)
    by = defaultdict(dict)
    for tid, out in outputs.items():
        by[rows[tid][3]][tid] = out               # rows[tid] = (blocks, labels, ref, language)
    per = [bench.score_text(sub, fold) for sub in by.values()]
    return {"n": sum(s["n"] for s in per), "prec": mean(s["prec"] for s in per),
            "rec": mean(s["rec"] for s in per), "f1": mean(s["f1"] for s in per)}


def _pages(board, fold):
    """{tid: {"html": ...}} for the fold. WCXB reads the split from wcxb.jsonl; DAnIEL is
    eval-only, all 1,689."""
    board = board.partition("-")[0]                  # wcxb-article reads wcxb's pages
    path = ROOT / "vendors" / board / "data" / f"{board}.jsonl"
    split = "test" if fold == "test" else "dev"
    pages = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if board == "wcxb" and r["split"] != split:
                continue
            pages[r["track_id"]] = {"html": r["html"]}
    return pages


def _env():
    pins = " ".join(f"{p}=={_ver(p)}" for p in
                    ("htmlsift", "onnxruntime", "lxml", "trafilatura", "resiliparse", "readability-lxml"))
    return [f"CPU: {platform.processor() or 'unknown'}.", f"Pins: {pins}.", ""]


def _keeper(bundle):
    """The keeper a bundle was baked from, as its manifest records it."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    return Path(manifest["source"]["keeper"]).stem


def _ver(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "?"


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True, choices=["wcxb", "wcxb-article", "daniel"])
    ap.add_argument("--fold", default="test", choices=["test", "val"])
    ap.add_argument("--bundle", type=Path, action="append", required=True,
                    help="mini bundle directory; repeat for one row each")
    ap.add_argument("--extractors", default=",".join(ex.EXTRACTORS))
    ap.add_argument("--limit", type=int, default=None, help="cap pages (smoke)")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.board == "daniel" and args.fold != "test":
        raise SystemExit("DAnIEL is eval-only: fold must be test")

    bench = load_benchmark(args.board)
    scorable = set(list(bench._fold_blocks(args.fold))[:args.limit])
    pages = {t: p for t, p in _pages(args.board, args.fold).items() if t in scorable}
    assert set(pages) == scorable, scorable - set(pages)

    subs = bench.subsets(args.fold) if hasattr(bench, "subsets") else {}
    base = args.board.partition("-")[0]

    def scores(out):
        return (_headline(bench, args.board, out, args.fold),
                {n: _headline(bench, args.board, {t: o for t, o in out.items() if t in ids},
                              args.fold) for n, ids in subs.items()})

    rows = []
    for bundle in args.bundle:
        extractor = Extractor(bundle_dir=bundle, threads=args.threads)
        out = {t: (extractor.extract(pages[t]["html"], output="text"), "text") for t in scorable}
        empty = sum(not text for text, _ in out.values())
        rows.append((f"htmlsift {_keeper(bundle)}", *scores(out), {"empty": empty, "raised": 0}))
    for name in args.extractors.split(","):
        out, notes = ex.run(name, pages, warmup=0, text=True)   # native text where a tool has it
        cov = set(out) & scorable
        rows.append((name, *scores({t: (out[t].text, out[t].kind) for t in cov}), notes))

    lines = [f"{args.board.upper()} {args.fold} -- CPU accuracy vs third-party extractors.", "",
             *_env(),
             f"One population: the {len(scorable)} pages of the fold. Metric = {METRIC[base]} "
             f"over each extractor's text output (trafilatura/resiliparse native, readability flattened "
             f"via html_to_text, htmlsift the package's output=\"text\"); empty output scores 0. "
             f"Regenerate: `python bench/accuracy/heuristics.py --board {args.board} --bundle <dir> ...`.", "",
             "| method | F1 | prec | rec | n |" + "".join(f" F1 {n} (n) |" for n in subs),
             "|---|---|---|---|---|" + "---|" * len(subs)]
    for name, sc, ssc, notes in rows:
        flag = "*" if notes["empty"] or notes["raised"] else ""
        lines.append(f"| {name}{flag} | {sc['f1']:.4f} | {sc['prec']:.4f} | {sc['rec']:.4f} | {sc['n']} |"
                     + "".join(f" {s['f1']:.4f} ({s['n']}) |" for s in ssc.values()))
    lines.append("")
    for name, _, _, notes in rows:
        if notes["empty"] or notes["raised"]:
            bits = ([f"empty on {notes['empty']} (scored 0)"] if notes["empty"] else []) + \
                   ([f"crashed on {notes['raised']} (excluded)"] if notes["raised"] else [])
            lines.append(f"{name}*: {', '.join(bits)} of {len(scorable)}.")

    md = "\n".join(lines)
    print("\n" + md)
    suffix = "" if args.fold == "test" else "-dev"      # test is the published fold; dev is the cross-check
    dest = ROOT / "results" / f"heuristics-{args.board}{suffix}.md"
    dest.write_text(md + "\n", encoding="utf-8")
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
