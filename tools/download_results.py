"""
Results ledgers -> committed ablation markdown. Pulls each ledger from the HF dataset
($HF_RESULTS_DATASET), collapses each arm to its latest ts, aggregates over seeds, and
writes one markdown table per ledger under results/ -- so an external reader sees the
numbers without re-running anything. Frozen and fine-tuned are separate ledgers ->
separate files (never one table, per the basis rule, §6).

    python tools/download_results.py                                             # all ledgers -> results/*.md
    python tools/download_results.py --local data/runs/wmb/frozen_results.jsonl  # one ledger -> stdout (dev)
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from trainers.ledger import record_key

ROOT = Path(__file__).resolve().parents[1]

HEAD_ORDER = ["bigru", "linear", "transformer", "xgboost"]
FEAT_ORDER = ["none", "A", "AB", "ABC", "BC", "C"]


def _rank(order, v):
    return (order.index(v), "") if v in order else (len(order), v)


def _epochs(r):
    """Epoch budget, or None for a record with no per-epoch curves (the frozen, table
    and feature ledgers keep only best_val). trainers.train never stops early, so
    len(val_by_epoch) is the budget that was configured."""
    ve = r.get("val_by_epoch")
    return len(ve) if ve else None


def group_key(r):
    """The reader's cell identity: the writer's arm key -- which already separates cap,
    train_limit and holdout -- plus the epoch budget, which no key carries and which no
    cell may average over. Derived from record_key so the two cannot drift."""
    return record_key(r) + (_epochs(r),)


def _warn_test_cell_clashes(records):
    """A decision-grade test cell must resolve to one run, not a latest-ts tiebreak
    between two different ones (§6). Shards are arm-key-disjoint by construction, so a
    same-cell test_f1 clash across shards means that invariant broke -- warn loudly with
    the runs to reconcile, rather than let collapse() pick one on a timestamp alone."""
    by_cell = defaultdict(list)
    for r in records:
        if "test_f1" in r:
            by_cell[group_key(r)].append(r)
    for k, rs in by_cell.items():
        if len({round(r["test_f1"], 6) for r in rs}) > 1:
            runs = ", ".join(f"{r['test_f1']:.6f}@{r['ts']}"
                             for r in sorted(rs, key=lambda r: r["ts"]))
            print(f"WARNING: {k[0]}-{k[1]} {k[2]}/{k[3]} s{k[5]} ep{k[-1]}: "
                  f"{len(rs)} differing test runs on one cell ({runs}) -- latest-ts wins; "
                  "scrub the stale record so no published number rides on a timestamp.",
                  file=sys.stderr)


def collapse(records):
    """group_key -> latest-ts record, then the survivors grouped over everything but the
    seed into their per-seed record lists."""
    _warn_test_cell_clashes(records)
    latest = {}
    for r in records:
        k = group_key(r)
        if k not in latest or r["ts"] > latest[k]["ts"]:
            latest[k] = r
    groups = defaultdict(list)
    for k, r in latest.items():
        groups[k[:5] + k[6:]].append(r)          # k[5] is arm_key's seed
    return list(groups.values())


def cell(vals):
    m = statistics.mean(vals)
    return f"{m:.4f}±{statistics.stdev(vals):.4f}" if len(vals) > 1 else f"{m:.4f}"


def _arm(r):
    return f"{r['head']}/{r['feats'] or 'none'}" + (f"/cap{r['cap']}" if r.get("cap") else "")


# The two val bases (§6). epoch-mean is the selection basis and comes first; best-epoch is
# what the kept checkpoint realizes, so it is what a test number pairs against. Both are
# read off one run -- the per-epoch curves exist so the basis is never a reason to re-run.
# epoch-mean is None where a ledger keeps no curves (frozen, table, features).
BASES = (("epoch-mean", lambda r: statistics.mean(r["val_by_epoch"]) if r.get("val_by_epoch")
          else None),
         ("best-epoch", lambda r: r["best_val"]))


def _table(cells, row_of, row_head, value_of):
    cols = sorted({(g[0]["head"], g[0]["feats"] or "none", g[0].get("cap", 0)) for g in cells},
                  key=lambda c: (_rank(HEAD_ORDER, c[0]), _rank(FEAT_ORDER, c[1]), c[2]))
    at = {(row_of(g[0]), g[0]["head"], g[0]["feats"] or "none", g[0].get("cap", 0)): g
          for g in cells}
    header = row_head + [f"{h}/{f}" + (f"/cap{c}" if c else "") for h, f, c in cols]
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in sorted({row_of(g[0]) for g in cells}):
        vals = [[v for v in map(value_of, at.get((row, h, f, c), ())) if v is not None]
                for h, f, c in cols]
        out.append("| " + " | ".join([str(x) for x in row]
                                     + [cell(v) if v else "—" for v in vals]) + " |")
    return out + [""]


def _test_table(cells, title, extra, head):
    """The arms carrying an --export-test fold. Sparse -- one run each -- and its seed count
    is its own: an arm can have three val seeds and one exported test seed, so this cannot
    ride in the val grid without implying seeds it does not have. n is a column rather than
    a heading because it varies within a ledger: DAnIEL LOLO's test population is whichever
    language was held out (el 273 .. en 475)."""
    rows = [(g, [r for r in g if "test_f1" in r]) for g in cells]
    rows = [(g, t) for g, t in rows if t]
    if not rows:
        return []
    header = head + ["model", "layers", "ep", "head/feats", "seeds", "n", "F1", "P", "R"]
    out = [f"## {title}", "",
           "| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for g, t in sorted(rows, key=lambda gt: (tuple(f(gt[0][0]) for f in extra),
                                             gt[0][0]["model"], gt[0][0]["layers"],
                                             _epochs(gt[0][0]) or 0, _arm(gt[0][0]))):
        r = g[0]
        out.append("| " + " | ".join(
            [str(f(r)) for f in extra]
            + [r["model"], str(r["layers"]), str(_epochs(r)), _arm(r),
               str(len(t)), str(t[0]["test_n"])]
            + [cell([x[k] for x in t]) for k in ("test_f1", "test_prec", "test_rec")]) + " |")
    return out + [""]


def render(records, caption):
    """The full pool, the train-limit ladder and the LOLO holdouts are separate tables --
    one cell may never average across them (§6). Epoch budget is a row dimension wherever
    the ledger records curves, and each val section renders once per available basis."""
    sections = defaultdict(list)
    for g in collapse(records):
        r = g[0]
        sections["ladder" if "train_limit" in r else
                 "holdout" if "holdout" in r else "full"].append(g)
    out = [caption, "", "Regenerate: `python tools/download_results.py`", ""]
    for name, title, extra, head in (
            ("full", "", [], []),
            ("ladder", "Training-set size", [lambda r: r["train_limit"]], ["pages"]),
            ("holdout", "Held-out language", [lambda r: r["holdout"]], ["holdout"])):
        cells = sections.get(name)
        if not cells:
            continue
        bases = [b for b in BASES if any(b[1](g[0]) is not None for g in cells)]
        for label, value_of in bases:
            # a heading only where it disambiguates: a single-basis ledger renders bare
            heading = " -- ".join(t for t in (title, f"val ({label})" * (len(bases) > 1)) if t)
            if heading:
                out += [f"## {heading}", ""]
            for model in sorted({g[0]["model"] for g in cells}):
                sub = [g for g in cells if g[0]["model"] == model]
                ep = [_epochs] if any(_epochs(g[0]) for g in sub) else []
                fns = extra + [lambda r: r["layers"]] + ep
                out += [f"### {model}", ""]
                out += _table(sub, lambda r: tuple(f(r) for f in fns),
                              head + ["layers"] + (["ep"] if ep else []), value_of)
        # every section may carry an exported fold -- LOLO's whole result is its test fold
        out += _test_table(cells, " -- ".join(t for t in (title, "test") if t), extra, head)
    return "\n".join(out)


def collapse_matrix(records):
    """(model, eval_board, seed) -> latest-ts, then (model, eval_board) -> per-seed values
    (DAnIEL's value is the by_lang macro -- its board headline; others' is f1). langs:
    model -> lang -> per-seed values from by_lang, for the DAnIEL per-language table.

    n is not part of the key, so a short run would overwrite a full one on ts alone. Every
    genuine run of a board covers all of it, so the largest n seen for a board is its
    population and anything under it is a smoke that never belonged on the dataset."""
    full = defaultdict(int)
    for r in records:
        full[r["eval_board"]] = max(full[r["eval_board"]], r["n"])
    short = [r for r in records if r["n"] < full[r["eval_board"]]]
    if short:
        raise ValueError(
            "smoke records in the generalization ledger -- scrub them from the dataset:\n"
            + "\n".join(f"  {r['model']} on {r['eval_board']} s{r['seed']}: "
                        f"n={r['n']} of {full[r['eval_board']]} ({r['ts']})" for r in short))
    latest = {}
    for r in records:
        k = (r["model"], r["eval_board"], r["seed"])
        if k not in latest or r["ts"] > latest[k]["ts"]:
            latest[k] = r
    cells, langs, types = defaultdict(list), defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    for (model, board, _), r in latest.items():
        bl = r.get("by_lang")
        cells[(model, board)].append(bl["macro"] if board == "daniel" and bl else r["f1"])
        if bl:
            for lang, v in bl.items():
                langs[model][lang].append(v)
        bt = r.get("by_type")
        if bt:
            for t, v in bt.items():
                types[model][t].append(v)
    return cells, langs, types


TYPE_ORDER = ["article", "forum", "product", "collection", "listing", "documentation", "service"]


def render_matrix(records, caption):
    cells, langs, types = collapse_matrix(records)
    models = sorted({m for m, _ in cells})
    boards = [b for b in ("wmb", "wcxb", "daniel") if any(bb == b for _, bb in cells)]
    out = [caption, "", "Regenerate: `python tools/download_results.py`", "",
           "| model | " + " | ".join(boards) + " |",
           "|" + "|".join(["---"] * (len(boards) + 1)) + "|"]
    trained = {r["model"]: r["trained_on"] for r in records}
    for m in models:
        trained_on = trained[m]
        row = [m]
        for b in boards:
            if (m, b) not in cells:
                row.append("—")
            else:
                c = cell(cells[(m, b)])
                row.append(f"**{c}**" if b == trained_on else c)
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    if langs:
        cols = ["macro", "el", "pl", "ru", "en", "zh"]
        out += ["### DAnIEL per language (ROUGE-L F1, mean±std)", "",
                "| model | " + " | ".join(cols) + " |",
                "|" + "|".join(["---"] * (len(cols) + 1)) + "|"]
        for m in sorted(langs):
            out.append("| " + " | ".join([m] + [cell(langs[m][c]) if langs[m].get(c) else "—"
                                                 for c in cols]) + " |")
        out.append("")
    if types:
        out += ["### WCXB per type (word-F1, mean±std)", "",
                "| model | " + " | ".join(TYPE_ORDER) + " |",
                "|" + "|".join(["---"] * (len(TYPE_ORDER) + 1)) + "|"]
        for m in sorted(types):
            out.append("| " + " | ".join([m] + [cell(types[m][t]) if types[m].get(t) else "—"
                                                 for t in TYPE_ORDER]) + " |")
        out.append("")
    return "\n".join(out)


# (source paths, output file, caption, renderer) -- each table merges every shard of every
# source (arm-key-disjoint). Frozen/fine-tune/qat are separate bases -> separate tables
# (§6); the generalization matrix is its own renderer. A source not in the repo is skipped.
LEDGERS = [
    (["wmb/frozen_results.jsonl", "wmb/feature_results.jsonl",
      "wmb/table_results.jsonl"], "results/wmb-frozen.md",
     "WMB frozen screen -- val732 ROUGE-5 F1, best-epoch val.", render),
    (["wmb/finetune_results.jsonl"], "results/wmb-finetune.md",
     "WMB fine-tune -- val732 ROUGE-5 F1 on both bases (§6); the test545 table is the arms "
     "re-run with --export-test, and its seed count is its own.", render),
    (["wmb/finetune_qat_results.jsonl"], "results/wmb-finetune-qat.md",
     "WMB fine-tune W8A8-QAT -- val732 ROUGE-5 F1 on both bases (§6); the test545 table is "
     "the arms re-run with --export-test, and its seed count is its own.", render),
    (["wcxb/finetune_results.jsonl"], "results/wcxb-finetune.md",
     "WCXB fine-tune -- word-F1 by their evaluate.py. WCXB carves no val fold, so the val "
     "columns are dev1497 (train == val == dev) and the test table is test511.", render),
    (["wcxb/finetune_qat_results.jsonl"], "results/wcxb-finetune-qat.md",
     "WCXB fine-tune W8A8-QAT -- word-F1 by their evaluate.py, val columns dev1497 (WCXB "
     "carves no val fold), test table test511.", render),
    (["wcxb/table_results.jsonl"], "results/wcxb-table.md",
     "WCXB embedding-table arms (no encoder) -- word-F1 by their evaluate.py, dev1497. "
     "embedding_table has no test path, so there is no test511 table here.", render),
    ([f"daniel-{iso}/finetune_results.jsonl" for iso in ("el", "pl", "ru", "en", "zh")],
     "results/daniel-lolo.md",
     "DAnIEL leave-one-language-out -- ROUGE-L F1, our reimplementation of the SIGIR 2025 "
     "metric (LIMITS.md). The result is the test table: train on four languages, score the "
     "fifth, so n is that language's page count. The val columns are the four training "
     "languages and are in-domain, not a board number.", render),
    (["generalization_results.jsonl"], "results/generalization.md",
     "Generalization matrix -- test F1 per board (mean±std, 3 seeds). Each column is that "
     "board's OWN metric (WMB ROUGE-5 / WCXB word-F1 / DAnIEL ROUGE-L macro) -- comparable "
     "down a column, never across a row. **Bold** = in-domain (the model's training board), "
     "a reproduction, not a zero-shot number.", render_matrix),
]


def load_hf(repo, paths):
    """Load every shard of every source in `paths` (each a base like
    wmb/frozen_results.jsonl matching frozen_results.<models>.jsonl, or
    wmb/feature_results.jsonl), merged into one ledger. Shards are written per pod
    / per mode so parallel runs don't clobber; arm-keys are disjoint across them,
    so collapse() dedups without conflict. None if the repo has none of them yet."""
    from huggingface_hub import hf_hub_download, list_repo_files
    try:
        files = list_repo_files(repo, repo_type="dataset")
    except Exception:
        return None
    shards = []
    for path in paths:
        base = path[:-len(".jsonl")]
        shards += [f for f in files
                   if f == path or (f.startswith(base + ".") and f.endswith(".jsonl"))]
    if not shards:
        return None
    recs = []
    for f in sorted(set(shards)):
        p = hf_hub_download(repo, f, repo_type="dataset")
        recs += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    return recs


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("HF_RESULTS_DATASET"),
                    help="HF dataset repo (default $HF_RESULTS_DATASET)")
    ap.add_argument("--local", help="render one local ledger to stdout (dev)")
    args = ap.parse_args()
    if args.local:
        recs = [json.loads(l) for l in open(args.local, encoding="utf-8") if l.strip()]
        renderer = render_matrix if recs and "eval_board" in recs[0] else render
        print(renderer(recs, f"local: {args.local}"))
        return
    if not args.repo:
        raise SystemExit("set HF_RESULTS_DATASET or pass --local")
    for paths, out, caption, renderer in LEDGERS:
        recs = load_hf(args.repo, paths)
        if recs is None:
            print(f"skip {out}: none of {paths} in {args.repo} yet")
            continue
        dest = ROOT / out
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(renderer(recs, caption) + "\n", encoding="utf-8")
        print(f"wrote {out} ({len(recs)} records)")


if __name__ == "__main__":
    main()
