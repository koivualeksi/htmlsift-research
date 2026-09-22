"""ROUGE-5 scoring of block-label predictions against the WMB references.

The 545 board metric is ROUGE-5 F1 against convert_main_content (their
calc_rouge_n_score, vendored in upstream/rouge_utils.py). Predictions are
manufactured through serialize.py (labels -> prune-then-render), so we score the
same kind of object the references are. Torch-free: the model forward lives in
its own module and reaches this scorer only as a per-block-probs file.

Folds:
  test (default) -- the 545 board (both 529 + test-only 16). cmc n=544;
    2fe1c202 has no convert_main_content, so it drops out of the cmc mean.
    groundtruth_content is scored as a secondary table.
  val            -- the 732 pool val pages. cmc n=732 (all labeled); the pool
    carries no groundtruth_content, so there is no secondary table.

Prediction sources:
  oracle (default) -- gold labels from blocks.jsonl. Bounds every model result
    and is this module's gate: it reproduces the DOM serializer's ceiling.
  --probs PATH     -- {track_id, probs} exported by the model module, labels at
    --threshold (default 0.5). The headline test number runs through here.
  --selectall      -- every block selected: the floor a model must clear. A
    baseline, not a gate -- no assertion.

  python vendors/wmb/adapter/eval.py [--fold test|val]
  python vendors/wmb/adapter/eval.py --fold val --probs data/eval/val_probs.jsonl
"""

import argparse
import json
from pathlib import Path

from core.constants import THRESHOLD
from vendors.wmb.adapter import serialize
from vendors.shared.upstream import load

WMB = Path(__file__).resolve().parents[1]
DATA = WMB / "data"
COMBINED = DATA / "wmb.jsonl"
SPLITS = DATA / "splits.json"
BLOCKS = DATA / "blocks.jsonl"
EVAL = DATA / "eval"

POP = {"test": "test545", "val": "val732"}

# Oracle DOM ceiling, ROUGE-5 F1 vs convert_main_content, frozen from this repo's
# own run of the lxml provenance pipeline (§4, 2026-08-31). The oracle asserts it
# and reports the fresh value. (html2text-pipeline baseline was test 0.994356 /
# val 0.982753; the lxml renderer costs -0.26% test / -0.44% val.)
CEILING = {"test": 0.991769, "val": 0.978383}


calc_rouge_n_score = load(WMB / "upstream" / "rouge_utils.py", "wmb_rouge").calc_rouge_n_score


def fold_ids(fold: str) -> set[str]:
    with open(SPLITS, encoding="utf-8") as f:
        return {tid for tid, v in json.load(f).items() if v["fold"] == fold}


def load_pages(ids: set[str]) -> dict:
    """track_id -> {html, url, cmc, gt} for the fold, from wmb.jsonl."""
    pages = {}
    with open(COMBINED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["track_id"] not in ids:
                continue
            pages[r["track_id"]] = {
                "html": r["html"],
                "url": r.get("url", ""),
                "cmc": r.get("convert_main_content"),
                "gt": r.get("groundtruth_content"),
            }
    return pages


def load_blocks(ids: set[str]) -> dict:
    """track_id -> {blocks, labels} for the fold, from blocks.jsonl."""
    blocks = {}
    with open(BLOCKS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["track_id"] not in ids:
                continue
            blocks[r["track_id"]] = {"blocks": r["blocks"], "labels": r["labels"]}
    return blocks


def predict(pages: dict, blocks: dict, labels_of):
    """labels_of(tid, b) -> per-block labels -> serialize each page. No fallback:
    provenance serialization is exact for every page."""
    preds = {}
    for tid, b in blocks.items():
        labels = labels_of(tid, b)
        page = pages[tid]
        prep = serialize.prepare(page["html"], page["url"], len(labels))
        preds[tid] = serialize.apply(prep, labels)
    return preds


def score_predictions(preds: dict, pages: dict, tag: str, out_path: Path) -> dict:
    """Score each prediction vs cmc (primary) and gt (secondary, where present),
    arithmetic mean over records. Writes per-record scores; the cmc/gt keys are
    present only where a reference existed, so downstream reads use r.get(key)."""
    rows = []
    for tid, pred in preds.items():
        row = {"track_id": tid}
        if pages[tid]["cmc"] is not None:
            row["cmc"] = calc_rouge_n_score(pages[tid]["cmc"], pred)
        if pages[tid]["gt"] is not None:
            row["gt"] = calc_rouge_n_score(pages[tid]["gt"], pred)
        rows.append(row)

    summary = {"tag": tag, "n_predictions": len(preds)}
    for key, label in (("cmc", "convert_main_content"), ("gt", "groundtruth_content")):
        scored = [r[key] for r in rows if key in r]
        if not scored:
            continue
        summary[key] = {
            "n": len(scored),
            "prec": sum(s["prec"] for s in scored) / len(scored),
            "rec": sum(s["rec"] for s in scored) / len(scored),
            "f1": sum(s["f1"] for s in scored) / len(scored),
        }
        m = summary[key]
        print(f"{tag} vs {label}: n={m['n']}  "
              f"P {m['prec']:.4f}  R {m['rec']:.4f}  F1 {m['f1']:.4f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"per-record scores: {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", choices=["test", "val"], default="test")
    ap.add_argument("--probs", type=str, default=None,
                    help="{track_id, probs} jsonl from the model module "
                         "(oracle mode when omitted)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="decision threshold for --probs (tuned on val, never test)")
    ap.add_argument("--selectall", action="store_true",
                    help="score every block selected -- the floor")
    args = ap.parse_args()

    ids = fold_ids(args.fold)
    pages = load_pages(ids)
    blocks = load_blocks(ids)
    assert set(pages) == ids and set(blocks) == ids, "fold ids missing from a source"

    if args.fold == "test":
        assert len(ids) == 545, f"test fold {len(ids)} != 545"
    else:
        assert len(ids) == 732, f"val fold {len(ids)} != 732"

    if args.probs:
        probs_path = Path(args.probs)
        probs = {}
        with open(probs_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                probs[r["track_id"]] = r["probs"]
        assert ids <= set(probs), f"{len(ids - set(probs))} fold pages missing from probs"

        def labels_of(tid, b):
            p = probs[tid]
            assert len(p) == len(b["blocks"]), \
                f"probs/blocks mismatch {tid}: {len(p)} vs {len(b['blocks'])}"
            return [int(v > args.threshold) for v in p]

        preds = predict(pages, blocks, labels_of)
        stem = probs_path.stem
        thr = "" if args.threshold == 0.5 else f"_thr{args.threshold:g}"
        print(f"probs [{stem}] {POP[args.fold]} thr {args.threshold:g}: {len(preds)} pages")
        score_predictions(preds, pages, f"MODEL/{stem}{thr}/{POP[args.fold]}",
                          EVAL / f"model_rouge_{stem}{thr}_{args.fold}.jsonl")
        return

    if args.selectall:
        preds = predict(pages, blocks, lambda tid, b: [1] * len(b["blocks"]))
        print(f"select-all {POP[args.fold]}: {len(preds)} pages")
        score_predictions(preds, pages, f"SELECTALL/{POP[args.fold]}",
                          EVAL / f"selectall_rouge_{args.fold}.jsonl")
        return

    preds = predict(pages, blocks, lambda tid, b: b["labels"])
    print(f"oracle {POP[args.fold]}: {len(preds)} pages")
    summary = score_predictions(preds, pages, f"ORACLE/{POP[args.fold]}",
                                EVAL / f"oracle_rouge_{args.fold}.jsonl")

    f1 = summary["cmc"]["f1"]
    frozen = CEILING[args.fold]
    if frozen is None:
        print(f"oracle {POP[args.fold]} ceiling F1 {f1:.6f} "
              f"(not yet frozen; pin in CEILING)")
    else:
        assert round(f1, 6) == frozen, f"oracle ceiling {f1:.6f} != frozen {frozen}"
        print(f"ceiling gate OK: {f1:.6f} == {frozen}")


if __name__ == "__main__":
    main()
