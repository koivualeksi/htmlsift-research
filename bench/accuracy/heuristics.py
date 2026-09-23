"""CPU accuracy: our mini vs third-party extractors on WCXB / DAnIEL (the boards outside
the WMB training policy). Accuracy only -- no speed columns: §7 keeps speed claims to WMB
(bench/speed/), and a cross-board speed ratio would reopen the population landmine.

Every extractor is judged by the board's score_text() on its best TEXT output: trafilatura
and resiliparse emit native text (as the WCXB leaderboard scores them); readability has no
text mode, so its html is flattened with html_to_text. mini is the shipped
int8 ONNX artifact (seed s1); on a text-metric board its standard output is its selected
TEXT (join_selected, the assembly score_fold uses), so score_text returns its canonical
number. Flattening mini's html instead loses ~1pt (the round-trip is lossy) and would move
our scoring, so we do not; the gate asserts score_text and score_fold agree for mini.

Population = the block-scored pages of the fold (score_fold's own set, = the published mini
population); every method is scored over that one set, an empty return scoring 0.

    python bench/accuracy/heuristics.py --board wcxb
    python bench/accuracy/heuristics.py --board daniel --limit 20      # smoke
"""
import argparse
import json
import platform
import tempfile
from collections import defaultdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

from bench.accuracy.predict import _load_feats
from bench.speed import competitors as ex
from core.loader import resolve_ckpt
from export.infer import load_mini
from export.paths import mini_paths
from vendors.shared.arms import WINDOW
from vendors.shared.benchmark import load_benchmark
from vendors.shared.labeling import join_selected

ROOT = Path(__file__).resolve().parents[2]
KEEPER = "wmb/ckpt/311m-table-bigru-fABC_s1.pt"     # shipped mini arm/seed (tools/release.py)
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
                    ("onnxruntime", "lxml", "trafilatura", "resiliparse", "readability-lxml"))
    return [f"CPU: {platform.processor() or 'unknown'}.", f"Pins: {pins}.",
            "mini: shipped int8 ONNX (data/onnx), seed s1.", ""]


def _ver(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "?"


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", required=True, choices=["wcxb", "daniel"])
    ap.add_argument("--fold", default="test", choices=["test", "val"])
    ap.add_argument("--keeper", default=KEEPER, help="table keeper (tokenizer + z-score stats)")
    ap.add_argument("--extractors", default=",".join(ex.EXTRACTORS))
    ap.add_argument("--limit", type=int, default=None, help="cap pages (smoke)")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    if args.board == "daniel" and args.fold != "test":
        raise SystemExit("DAnIEL is eval-only: fold must be test")
    dev = torch.device("cpu")

    bench = load_benchmark(args.board)
    pages = _pages(args.board, args.fold)
    if args.limit:
        pages = dict(list(pages.items())[:args.limit])

    # mini (shipped ONNX): one infer_fn drives both the canonical score_fold number and the
    # score_text gate, so the gate isolates the html round-trip, not a second model.
    ck = resolve_ckpt(args.keeper, dev)
    tok = AutoTokenizer.from_pretrained(ck["model"])
    table_p, head_p = mini_paths(ROOT / "data" / "onnx")
    infer_fn = load_mini(table_p, head_p, args.threads)
    feats = _load_feats(bench, ck)
    scratch = Path(tempfile.gettempdir()) / f"heuristics_{args.board}_probs.jsonl"
    probs = bench.export_probs(args.fold, tok, infer_fn, scratch, WINDOW, 0, feats, args.limit)

    scorable = set(probs)                              # the block-scored population
    pages = {t: pages[t] for t in scorable if t in pages}
    blocks = bench._fold_blocks(args.fold)

    # mini's standard output on a text-metric board is its selected text, not html: the same
    # join_selected assembly score_fold/score_by_language use, so score_text (text-kind, no
    # flattening) returns its canonical number. Flattening our html instead loses ~1pt (the html
    # round-trip is lossy) -- that would move our scoring, so we do not. Gate asserts the two agree.
    mini_out = {t: (join_selected(blocks[t], [v > 0.5 for v in probs[t]]), "text") for t in scorable}
    mini_sc = _headline(bench, args.board, mini_out, args.fold)
    canon = (bench.score_by_language(probs, args.fold, 0.5)["macro"] if args.board == "daniel"
             else bench.score_fold(probs, args.fold, 0.5)["f1"])
    gate = abs(mini_sc["f1"] - canon)

    rows = [("htmlsift (mini)", mini_sc, {"empty": 0, "raised": 0})]
    for name in args.extractors.split(","):
        out, notes = ex.run(name, pages, warmup=0, text=True)   # native text where a tool has it
        cov = set(out) & scorable
        sc = _headline(bench, args.board, {t: (out[t].text, out[t].kind) for t in cov}, args.fold)
        rows.append((name, sc, notes))

    lines = [f"{args.board.upper()} {args.fold} -- CPU accuracy vs third-party extractors.", "",
             *_env(),
             f"One population: the {len(scorable)} block-scored pages. Metric = {METRIC[args.board]} "
             f"over each extractor's text output (trafilatura/resiliparse native, readability flattened "
             f"via html_to_text); empty output scores 0. mini is its canonical selection score; heuristics "
             f"via score_text. Regenerate: `python bench/accuracy/heuristics.py --board {args.board}`.", "",
             "| method | F1 | prec | rec | n |", "|---|---|---|---|---|"]
    for name, sc, notes in rows:
        flag = "*" if notes["empty"] or notes["raised"] else ""
        lines.append(f"| {name}{flag} | {sc['f1']:.4f} | {sc['prec']:.4f} | {sc['rec']:.4f} | {sc['n']} |")
    lines += ["", f"Gate: mini via score_text {mini_sc['f1']:.4f} vs canonical selection {canon:.4f}, "
              f"Δ={gate:.4f} ({'PASS' if gate < 1e-3 else 'FAIL'} at 1e-3) -- score_text does not move "
              f"mini's scoring.", ""]
    for name, _, notes in rows:
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
