"""Per-run error breakdown on a WMB fold: which rows a run gets wrong, how many chars they
carry, and what fixing each kind would be worth in board F1. Reads {track_id, probs}
files (a run's val.jsonl, or predict.py --ckpt output). With several files each is
compared against the first, and their averaged probs are broken down as one more run:
if the average beats every input, the runs' errors are complementary.

Rows are labelled at the threshold and grouped into runs of consecutive wrong rows.
  FN (missed gold) by predicted neighbours: interior (both sides), edge (one), detached (none).
  FP (extra row) by gold neighbours: hole (both), edge (one), isolated (none). Every
    extra row on a page with no gold is nogold.
  Both also by row length and by prob band.
Columns: rows; chars as % of gold chars; pages touched; gain, the board F1 gained by setting
that class's rows to gold (touched pages rescored with the vendored ROUGE-5, mean over all
pages). Gains overlap, so they do not add up. fixed/broken: chars of the class the first run
got wrong and this run gets right, and the reverse. "room" is oracle F1 minus this run's F1.

    python vendors/wmb/adapter/errors.py --fold val base/val.jsonl arm/val.jsonl
"""
import argparse
import json
import multiprocessing as mp

import jieba
import numpy as np
from htmlsift.extractor import THRESHOLD

from vendors.wmb.adapter import eval as wmb_eval, serialize

LEN = ((0, 40, "<40"), (40, 200, "40-200"), (200, 1 << 30, ">=200"))
BAND = {"FN": ((0.4, 0.5, "0.4-0.5"), (0.2, 0.4, "0.2-0.4"), (-1, 0.2, "<=0.2")),
        "FP": ((0.5, 0.6, "0.5-0.6"), (0.6, 0.8, "0.6-0.8"), (0.8, 1, "0.8-1"))}
KIND = {"FN": ("detached", "edge", "interior"), "FP": ("isolated", "edge", "hole", "nogold")}

_SNAP = None                     # (prepared, cmc) per page; fork workers inherit it


def _f1(job):
    tid, labels = job
    prepared, cmc = _SNAP
    return wmb_eval.calc_rouge_n_score(cmc[tid], serialize.apply(prepared[tid], labels))["f1"]


def _runs(mask):
    """(start, end) of each maximal run of True, end exclusive."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))


def classify(p, gold, chars):
    """class name -> row mask for one page."""
    pred = p > THRESHOLD
    wrong = {"FN": gold & ~pred, "FP": pred & ~gold}
    out = {"all": wrong["FN"] | wrong["FP"]}
    for err, side in (("FN", pred), ("FP", gold)):
        out[err] = wrong[err]
        kinds = {k: np.zeros(len(p), bool) for k in KIND[err]}
        if err == "FP" and not gold.any():
            kinds["nogold"] = wrong[err]
        else:
            for s, e in _runs(wrong[err]):
                kinds[KIND[err][int(s > 0 and side[s - 1]) + int(e < len(p) and side[e])]][s:e] = True
        out.update({f"{err} {k}": m for k, m in kinds.items()})
        out.update({f"{err} len{n}": wrong[err] & (chars >= lo) & (chars < hi) for lo, hi, n in LEN})
        out.update({f"{err} p{n}": wrong[err] & (p > lo) & (p <= hi) for lo, hi, n in BAND[err]})
    return out


def breakdown(probs, data, pool, first=None):
    """Per-class stats for one run; first = the reference run's classes (compare mode)."""
    base = dict(zip(probs, pool.map(_f1, [(t, (probs[t] > THRESHOLD).astype(int).tolist())
                                          for t in probs])))
    cls = {t: classify(probs[t], data[t]["gold"], data[t]["chars"]) for t in probs}
    gold_ch = sum(int(d["chars"][d["gold"]].sum()) for d in data.values())
    rows = {}
    for name in cls[next(iter(cls))]:
        hit = {t: c[name] for t, c in cls.items() if c[name].any()}
        jobs = [(t, np.where(m, data[t]["gold"], probs[t] > THRESHOLD).astype(int).tolist())
                for t, m in hit.items()]
        gain = sum(f - base[t] for (t, _), f in zip(jobs, pool.map(_f1, jobs))) / len(base)
        r = {"rows": int(sum(m.sum() for m in hit.values())),
             "ch": sum(int(data[t]["chars"][m].sum()) for t, m in hit.items()) / gold_ch,
             "pages": len(hit), "gain": gain}
        if first is not None:
            ok = {t: ~cls[t]["all"] for t in cls}
            ref_ok = {t: ~first["cls"][t]["all"] for t in cls}
            r["fixed"] = sum(int(data[t]["chars"][first["cls"][t][name] & ok[t]].sum())
                             for t in cls) / gold_ch
            r["broken"] = sum(int(data[t]["chars"][cls[t][name] & ref_ok[t]].sum())
                              for t in cls) / gold_ch
        rows[name] = r
    f1 = sum(base.values()) / len(base)
    return {"f1": f1, "base": base, "cls": cls, "rows": rows}


def report(name, b, oracle):
    room = {t: oracle[t] - f for t, f in b["base"].items()}
    worst = sorted(room.values(), reverse=True)[:max(1, len(room) // 10)]
    total = sum(room.values())
    print(f"\n## {name}\n\nF1 {b['f1']:.4f}  oracle {sum(oracle.values()) / len(oracle):.4f}  "
          f"room {total / len(room):.4f}  worst 10% of pages hold {sum(worst) / total:.0%} of room\n")
    cmp_ = "fixed" in b["rows"]["all"]
    print("| class | rows | % gold ch | pages | gain |" + (" fixed | broken |" if cmp_ else ""))
    print("|---|--:|--:|--:|--:|" + ("--:|--:|" if cmp_ else ""))
    for k, r in b["rows"].items():
        tail = f" {r['fixed']:.2%} | {r['broken']:.2%} |" if cmp_ else ""
        print(f"| {k} | {r['rows']} | {r['ch']:.2%} | {r['pages']} | {r['gain']:+.4f} |{tail}")


def main():
    global _SNAP
    ap = argparse.ArgumentParser()
    ap.add_argument("probs", nargs="+", help="{track_id, probs} jsonl, first is the reference")
    ap.add_argument("--fold", choices=["val", "test"], default="val")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    runs = []
    for path in args.probs:
        with open(path, encoding="utf-8") as f:
            runs.append({r["track_id"]: np.asarray(r["probs"], float) for r in map(json.loads, f)})
    ids = set(runs[0])
    assert all(set(r) == ids for r in runs), "probs files cover different pages"
    pages = wmb_eval.load_pages(ids)
    blocks = wmb_eval.load_blocks(ids)
    data = {t: {"gold": np.asarray(b["labels"], bool),
                "chars": np.asarray([len(s) for s in b["blocks"]])} for t, b in blocks.items()}
    for r in runs:
        for t, p in r.items():
            assert len(p) == len(data[t]["gold"]), f"probs/blocks mismatch {t}"
    if len(runs) > 1:
        runs.append({t: np.mean([r[t] for r in runs], 0) for t in ids})
        args.probs.append("average of all")

    keep = [t for t in ids if pages[t]["cmc"] is not None]          # as score_fold
    runs = [{t: r[t] for t in keep} for r in runs]
    _SNAP = ({t: serialize.prepare(pages[t]["html"], pages[t]["url"], len(data[t]["gold"]))
              for t in keep}, {t: pages[t]["cmc"] for t in keep})
    jieba.initialize()           # warm in the parent so forked workers inherit the dict
    with mp.get_context("fork").Pool(args.workers) as pool:
        oracle = dict(zip(keep, pool.map(_f1, [(t, data[t]["gold"].astype(int).tolist())
                                               for t in keep])))
        first = None
        for path, r in zip(args.probs, runs):
            b = breakdown(r, data, pool, first)
            report(path, b, oracle)
            first = first or b


if __name__ == "__main__":
    main()
