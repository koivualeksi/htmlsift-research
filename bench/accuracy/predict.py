"""Cross-board inference: run a trained keeper checkpoint over a target board it was
never trained on, emit that board's {track_id, probs} jsonl -> the board's eval --probs.
The zero-shot path (WMB-trained -> DAnIEL / WCXB). Board-agnostic: dispatch encoder vs
table on the ckpt's kind, resolve the target Benchmark by name, hand export_probs an
infer_fn -- no per-board branching. Our-models path only; the third-party extractor
comparison is bench/speed/ (competitors.py).

    python bench/accuracy/predict.py --model wmb-311m-10                            # all boards, all seeds
    python bench/accuracy/predict.py --ckpt ckpt.pt --benchmark daniel --out probs.jsonl  # one ckpt, one board
"""
import argparse
from pathlib import Path

import torch
from dotenv import load_dotenv

from core.loader import build_infer, resolve_ckpt
from core.features import apply_zscore, group_columns
from trainers import ledger
from vendors.shared.arms import WINDOW
from vendors.shared.benchmark import load_benchmark

BOARDS = ["wmb", "wcxb", "daniel"]
ROOT = Path(__file__).resolve().parents[2]


def _load_feats(bench, ck):
    """Target board's features for a ckpt trained with them, z-scored by the ckpt's
    CARRIED (training-board) stats then sliced to the group -- a foreign board normalized
    by the training yardstick, never its own (as bench/speed/frontend.py:build_feats_fn).
    apply_zscore runs on the FULL layout before the slice, since ck['zscore'] indexes it.
    Only table ckpts carry stats; the encoder keeper is text-only, so an encoder ckpt with
    features cross-board is unsupported."""
    group = ck.get("feats")
    if not group:
        return None
    stats = ck.get("zscore")
    if stats is None:
        raise SystemExit("ckpt has features but no carried z-score stats "
                         "(encoder+features cross-board unsupported; keepers are text-only)")
    cols = group_columns(group)
    return {tid: apply_zscore(a, stats)[:, cols] for tid, a in bench.raw_features().items()}


def _run_board(ck, tok, infer_fn, cap, board, out, fold, limit):
    """One keeper over one board: export probs, score, print. Returns the score dict
    (piece 4 records it)."""
    bench = load_benchmark(board)
    feats = _load_feats(bench, ck)
    out.parent.mkdir(parents=True, exist_ok=True)
    probs = bench.export_probs(fold, tok, infer_fn, out, WINDOW, cap, feats, limit)
    s = bench.score_fold(probs, fold, 0.5)
    if hasattr(bench, "score_by_language"):
        s["by_lang"] = bench.score_by_language(probs, fold, 0.5)
    if hasattr(bench, "score_by_type"):
        s["by_type"] = bench.score_by_type(probs, fold, 0.5)
    tail = ""
    if "by_lang" in s:
        tail = "  [" + " ".join(f"{k} {v:.4f}" for k, v in s["by_lang"].items()) + "]"
    if "by_type" in s:
        tail += "  [" + " ".join(f"{k} {v:.4f}" for k, v in s["by_type"].items()) + "]"
    print(f"{board} {fold}: f1 {s['f1']:.4f}  (n={s['n']} P={s['prec']:.4f} R={s['rec']:.4f}){tail}  -> {out}")
    return s


def _gen_key(r):
    return (r["model"], r["eval_board"], r["seed"])


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--model", help="board-prefixed keeper id, e.g. wmb-311m-10 (runs all boards)")
    grp.add_argument("--ckpt", help="a single ckpt (local path or repo-relative), one board")
    ap.add_argument("--benchmark", choices=BOARDS, help="target board (--ckpt mode)")
    ap.add_argument("--out", help="probs jsonl (--ckpt mode; auto-derived for --model)")
    ap.add_argument("--boards", default=",".join(BOARDS), help="boards to run (--model mode)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds to run (--model mode)")
    ap.add_argument("--fold", default="test")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--push", nargs="?", const="", default=None,
                    help="push the generalization ledger to $HF_RESULTS_DATASET (--model mode)")
    ap.add_argument("--shard", default="", help="ledger shard for parallel runs (--model mode)")
    ap.add_argument("--table-int8", action="store_true",
                    help="force per-row int8 on a table keeper's frozen embedding table at inference "
                         "(numerically the shipped mini -- core.quant.emb_int8_fakequant). Ledger rows are "
                         "tagged <model>-int8 so they never overwrite the fp32 generalization cells")
    args = ap.parse_args()
    dev = torch.device(args.device)

    if args.model:
        parts = args.model.split("-", 1)
        if len(parts) != 2 or parts[0] not in BOARDS:
            raise SystemExit(f"model id must be <board>-<arm>, board in {BOARDS}: {args.model}")
        trained_on, arm = parts
        mid = args.model + ("-int8" if args.table_int8 else "")   # ledger identity; the ckpt still loads from arm
        boards = args.boards.split(",")
        led = ledger.Ledger(ROOT / "data" / "runs" / "generalization_results.jsonl",
                            ledger.resolve_repo(args.push),
                            ledger.sharded_path("generalization_results.jsonl", args.shard),
                            key_fn=_gen_key)
        led.resume()
        for seed in range(args.seeds):
            if led.all_done(_gen_key({"model": mid, "eval_board": b, "seed": seed})
                            for b in boards):
                print(f"[skip {mid} s{seed}: all boards done]", flush=True)
                continue
            ck = resolve_ckpt(f"{trained_on}/ckpt/{arm}_s{seed}.pt", dev)
            if args.table_int8:
                if ck.get("kind") != "table":
                    raise SystemExit("--table-int8 applies to a table keeper only")
                ck["int8"] = True                       # quantize the frozen table; head weights unchanged
            tok, infer_fn, _kind, cap = build_infer(ck, dev)
            for board in boards:
                if led.is_done(_gen_key({"model": mid, "eval_board": board, "seed": seed})):
                    print(f"[skip {mid} {board} s{seed}: done]", flush=True)
                    continue
                out = ROOT / "data" / "runs" / board / "probs" / f"{mid}_s{seed}.jsonl"
                s = _run_board(ck, tok, infer_fn, cap, board, out, args.fold, args.limit)
                rec = {"model": mid, "trained_on": trained_on, "eval_board": board,
                       "seed": seed, "n": s["n"], "prec": s["prec"], "rec": s["rec"], "f1": s["f1"]}
                if "by_lang" in s:
                    rec["by_lang"] = s["by_lang"]
                if "by_type" in s:
                    rec["by_type"] = s["by_type"]
                led.append(rec)
            led.merge_push(f"benchmark {mid} s{seed}")
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
