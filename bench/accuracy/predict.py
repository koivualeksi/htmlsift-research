"""Cross-board inference: run a trained keeper checkpoint over a target board it was
never trained on, emit that board's {track_id, probs} jsonl -> the board's eval --probs.
The zero-shot path (WMB-trained -> DAnIEL / WCXB). Board-agnostic: dispatch encoder vs
table on the ckpt's kind, resolve the target Benchmark by name, hand export_probs an
infer_fn -- no per-board branching. Our-models path only; the third-party extractor
comparison is bench/speed/ (competitors.py).

--model writes one eval run per (model, board, seed) to the run store, experiment
`generalization`; --push ships each as it finishes. --limit makes them smoke runs, which
stay in data/runs/smoke/ and are never pushed.

    python bench/accuracy/predict.py --model wmb-311m-10 --push                     # all boards, all seeds
    python bench/accuracy/predict.py --ckpt ckpt.pt --benchmark daniel --out probs.jsonl  # one ckpt, one board
"""
import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

from core.loader import build_infer, resolve_ckpt
from trainers import runs
from vendors.shared import bench_ops
from vendors.shared.arms import WINDOW
from vendors.shared.benchmark import load_benchmark

BOARDS = ["wmb", "wcxb", "daniel"]
TARGETS = [*BOARDS, "wcxb-article"]                 # --ckpt mode: single-type boards too
ROOT = Path(__file__).resolve().parents[2]


def _run_board(ck, tok, infer_fn, cap, board, out, fold, limit):
    """One keeper over one board: export probs, score, print. Returns the score dict."""
    bench = load_benchmark(board)
    feats = bench_ops.carried_features(bench, ck)
    out.parent.mkdir(parents=True, exist_ok=True)
    probs = bench.export_probs(fold, tok, infer_fn, out, WINDOW, cap, feats, limit)
    s = bench.score_fold(probs, fold, 0.5)
    if subs := bench_ops.score_subsets(bench, probs, fold, 0.5):
        s["subsets"] = {k: {"n": v["n"], "f1": v["f1"]} for k, v in subs.items()}
    if hasattr(bench, "score_by_language"):
        s["by_lang"] = bench.score_by_language(probs, fold, 0.5)
    if hasattr(bench, "score_by_type"):
        s["by_type"] = bench.score_by_type(probs, fold, 0.5)
    tail = ""
    if "by_lang" in s:
        tail = "  [" + " ".join(f"{k} {v:.4f}" for k, v in s["by_lang"].items()) + "]"
    if "by_type" in s:
        tail += "  [" + " ".join(f"{k} {v:.4f}" for k, v in s["by_type"].items()) + "]"
    tail += "".join(f"  [{k} {v['f1']:.4f} n={v['n']}]" for k, v in s.get("subsets", {}).items())
    print(f"{board} {fold}: f1 {s['f1']:.4f}  (n={s['n']} P={s['prec']:.4f} R={s['rec']:.4f}){tail}  -> {out}")
    return s


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", help="board-prefixed keeper id, e.g. wmb-311m-10 (runs all boards)")
    grp.add_argument("--ckpt", help="a single ckpt (local path or repo-relative), one board")
    ap.add_argument("--benchmark", choices=TARGETS, help="target board (--ckpt mode)")
    ap.add_argument("--out", help="probs jsonl (--ckpt mode; auto-derived for --model)")
    ap.add_argument("--boards", default=",".join(BOARDS), help="boards to run (--model mode)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds to run (--model mode)")
    ap.add_argument("--fold", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--push", action="store_true",
                    help="push each run to $HF_RESULTS_DATASET (--model mode)")
    ap.add_argument("--table-int8", action="store_true",
                    help="force per-row int8 on a table keeper's frozen embedding table at inference "
                         "(numerically the shipped mini -- core.quant.emb_int8_fakequant). Runs are "
                         "tagged <model>-int8 so they never land in the fp32 generalization cells")
    args = ap.parse_args()
    dev = torch.device(args.device)

    if args.model:
        parts = args.model.split("-", 1)
        if len(parts) != 2 or parts[0] not in BOARDS:
            raise SystemExit(f"model id must be <board>-<arm>, board in {BOARDS}: {args.model}")
        trained_on, arm = parts
        mid = args.model + ("-int8" if args.table_int8 else "")   # cell identity; the ckpt still loads from arm
        boards = args.boards.split(",")
        smoke = args.limit is not None
        if args.push:
            if smoke:
                raise SystemExit("--limit runs are smoke runs and are never pushed")
            runs.preflight()
        launch = runs.new_id()
        for seed in range(args.seeds):
            ck = resolve_ckpt(f"{trained_on}/ckpt/{arm}_s{seed}.pt", dev)
            if args.table_int8:
                if ck.get("kind") != "table":
                    raise SystemExit("--table-int8 applies to a table keeper only")
                ck["int8"] = True                       # quantize the frozen table; head weights unchanged
            tok, infer_fn, _kind, cap = build_infer(ck, dev)
            for board in boards:
                config = {"recipe": "predict", "board": board, "model": mid,
                          "trained_on": trained_on, "eval_board": board, "seed": seed,
                          "fold": args.fold}
                folder = runs.start("generalization", launch, config, kind="eval", smoke=smoke)
                out = ROOT / "data" / "runs" / board / "probs" / f"{mid}_s{seed}.jsonl"
                s = _run_board(ck, tok, infer_fn, cap, board, out, args.fold, args.limit)
                runs.finish(folder, {k: s[k] for k in ("n", "prec", "rec", "f1", "by_lang", "by_type", "subsets")
                                     if k in s})
                if args.push:
                    runs.push(folder.name)
    else:
        if not (args.benchmark and args.out):
            raise SystemExit("--ckpt needs --benchmark and --out")
        ck = resolve_ckpt(args.ckpt, dev)
        if args.table_int8:
            if ck.get("kind") != "table":
                raise SystemExit("--table-int8 applies to a table keeper only")
            ck["int8"] = True
        tok, infer_fn, _kind, cap = build_infer(ck, dev)
        _run_board(ck, tok, infer_fn, cap, args.benchmark, Path(args.out), args.fold, args.limit)


if __name__ == "__main__":
    main()
