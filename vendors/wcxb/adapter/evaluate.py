"""
Score WCXB predictions through the benchmark's own evaluate.py.

Prediction text in -> their word-F1 out. Nothing here reimplements the metric:
it calls the vendored upstream/evaluate.py verbatim (CLAUDE.md 8/9). Predictions
are keyed by bare file_id, the key their scorer uses.

    python -X utf8 vendors/wcxb/adapter/evaluate.py --split test --oracle
    python -X utf8 vendors/wcxb/adapter/evaluate.py --split test --preds model_out.json
    python -X utf8 vendors/wcxb/adapter/evaluate.py --split test --probs data/runs/wcxb/probs/97m-6_test_s0.jsonl
    python -X utf8 vendors/wcxb/adapter/evaluate.py --split test --selectall

--selectall scores every block: the floor a model must clear, not a gate.

--oracle is the best block selection against the gold reference: a ceiling
diagnostic, not training -- test is never labeled. --preds scores a model's
{file_id: text} output. Writes data/eval/wcxb_<split>_<tag>_preds.json, which
re-scores with their shipped scorer (the copy beside the data tree, byte-identical
to upstream/):
    python vendors/wcxb/data/evaluate.py --split S --results <preds.json> --per-type
"""
import argparse
import json
from pathlib import Path

from core.constants import THRESHOLD
from vendors.wcxb.adapter.labels import (
    join_selected,
    label_page,
    wcxb_eval
)

DATA = Path(__file__).resolve().parents[1] / "data"
N = {"dev": 1497, "test": 511}
# oracle board ceiling, frozen from this repo's run (render_hidden=True), asserted
# on every oracle run over all 511 test pages via their evaluate_results.
ORACLE_CEILING = {"test": 0.9933}


def oracle_preds(split: str) -> dict:
    """{file_id: text} from the labeler's best selection against the gold refs."""
    refs = {}
    with open(DATA / "wcxb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split:
                refs[r["track_id"]] = r
    preds = {}
    with open(DATA / "blocks.jsonl", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            r = refs.get(b["track_id"])
            if r is None:
                continue
            labels, *_ = label_page(b["blocks"], r["main_content"] or "", r["with"], r["without"])
            preds[r["file_id"]] = join_selected(b["blocks"], labels)
    assert len(preds) == N[split], f"{len(preds)} preds, expected {N[split]}"
    return preds


def selectall_preds(split: str) -> dict:
    """{file_id: text} with every block selected -- the floor. Same
    adapt(unescape(block)) assembly as oracle_preds, labels all 1."""
    meta = {}  # track_id -> file_id, this split only
    with open(DATA / "wcxb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split:
                meta[r["track_id"]] = r["file_id"]
    preds = {}
    with open(DATA / "blocks.jsonl", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            fid = meta.get(b["track_id"])
            if fid is not None:
                preds[fid] = join_selected(b["blocks"])
    assert len(preds) == N[split], f"{len(preds)} preds, expected {N[split]}"
    return preds


def probs_preds(split: str, probs_path: Path, threshold: float) -> dict:
    """{file_id: text} from a {track_id, probs} export (finetune --export-test), labels at
    threshold. Same adapt(unescape(block)) assembly as oracle_preds."""
    meta = {}  # track_id -> file_id, this split only
    with open(DATA / "wcxb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == split:
                meta[r["track_id"]] = r["file_id"]
    blocks = {}
    with open(DATA / "blocks.jsonl", encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if b["track_id"] in meta:
                blocks[b["track_id"]] = b["blocks"]
    probs = {}
    with open(probs_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            probs[r["track_id"]] = r["probs"]
    assert meta.keys() <= probs.keys(), \
        f"{len(meta.keys() - probs.keys())} {split} pages missing from probs"
    preds = {}
    for tid, fid in meta.items():
        blk, pr = blocks[tid], probs[tid]
        assert len(pr) == len(blk), f"probs/blocks mismatch {tid}: {len(pr)} vs {len(blk)}"
        preds[fid] = join_selected(blk, [v > threshold for v in pr])
    assert len(preds) == N[split], f"{len(preds)} preds, expected {N[split]}"
    return preds


def load_gt(split: str) -> dict:
    """Ground truth in the shape evaluate_results consumes, from our collapsed
    corpus. Their load_ground_truth reads a tree relative to the vendored file;
    wcxb.jsonl carries the same fields, keyed by the same file_id."""
    gt = {}
    with open(DATA / "wcxb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] != split:
                continue
            gt[r["file_id"]] = {
                "main_content": r["main_content"] or "",
                "with": r["with"] or [],
                "without": r["without"] or [],
                "title": r.get("title", ""),
                "page_type": r["page_type"],
            }
    return gt


def score(preds: dict, split: str, tag: str):
    out = DATA / "eval" / f"wcxb_{split}_{tag}_preds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(preds), encoding="utf-8")
    results = wcxb_eval.evaluate_results(load_gt(split), preds, per_type=True)
    print(f"\n-> {out}")
    return sum(r["f1"] for r in results) / len(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--oracle", action="store_true")
    grp.add_argument("--preds", type=str, help="{file_id: text} json from a model")
    grp.add_argument("--probs", type=str,
                     help="{track_id, probs} jsonl from finetune --export-test")
    grp.add_argument("--selectall", action="store_true",
                     help="score every block selected -- the floor")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="decision threshold for --probs (tuned on dev, never test)")
    args = ap.parse_args()

    if args.oracle:
        f1 = score(oracle_preds(args.split), args.split, "oracle")
        exp = ORACLE_CEILING.get(args.split)
        assert exp is None or round(f1, 4) == exp, f"{args.split} oracle {f1:.4f} != {exp}"
    elif args.selectall:
        score(selectall_preds(args.split), args.split, "selectall")
    elif args.probs:
        stem = Path(args.probs).stem
        thr = "" if args.threshold == 0.5 else f"_thr{args.threshold:g}"
        score(probs_preds(args.split, Path(args.probs), args.threshold),
              args.split, f"{stem}{thr}")
    else:
        preds = json.loads(Path(args.preds).read_text(encoding="utf-8"))
        score(preds, args.split, Path(args.preds).stem)


if __name__ == "__main__":
    main()
