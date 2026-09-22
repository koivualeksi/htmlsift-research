"""
ROUGE-L scoring of DAnIEL predictions against the <p>-derived gold.

The board metric is ROUGE-L (P/R/F1), the metric of the SIGIR 2025 paper that
benchmarks extractors on this corpus. Nothing ships with the corpus to vendor, so
ROUGE-L is computed with rouge_score (the pinned dep WMB uses) over text tokenized
per language: jieba.lcut for Chinese, lowercased \\w+ for el/pl/ru/en -- the
both-sides tokenization WMB applies (CLAUDE.md 4). Without Chinese segmentation the
zh column collapses and the one claim this board carries dies.

Torch-free: predictions are text keyed by track_id. Two modes:

  oracle (default) -- the gold-labeled block-text join from labels.jsonl: the
    ceiling our block system can reach, assembled exactly as a model's prediction
    is, so it bounds every model result and is this module's gate.
  --preds PATH     -- {track_id: text} from a baseline runner or the model; the
    published board numbers run through here.
  --probs PATH     -- {track_id, probs} jsonl (what export_probs / infer_board
    write); blocks are selected at --threshold and joined exactly as oracle_preds,
    then scored. The cross-board zero-shot path feeds this.
  --selectall      -- every block selected: the floor a model must clear. A
    baseline, not a gate -- no assertion.

  python vendors/daniel/adapter/eval.py                 # oracle ceiling
  python vendors/daniel/adapter/eval.py --preds out.json
  python vendors/daniel/adapter/eval.py --probs probs.jsonl
  python vendors/daniel/adapter/eval.py --selectall
"""

import argparse
import json
import re
from pathlib import Path

import jieba
from rouge_score import rouge_scorer

from core.constants import THRESHOLD
from vendors.shared.labeling import join_selected

DATA = Path(__file__).resolve().parents[1] / "data"
COMBINED = DATA / "daniel.jsonl"
BLOCKS = DATA / "blocks.jsonl"
LABELS = DATA / "labels.jsonl"
EVAL = DATA / "eval"

LANGS = ["Greek", "Polish", "Russian", "English", "Chinese"]
ISO = {"Greek": "el", "Polish": "pl", "Russian": "ru", "English": "en", "Chinese": "zh"}

# ROUGE-L macro F1 of the oracle (word-F1-optimal selection) block-text join,
# frozen from this repo's run; the board ceiling and this module's gate.
ORACLE_CEILING = 0.9816

_WORD = re.compile(r"\w+", re.UNICODE)


class _PretokenizedTokenizer:
    def tokenize(self, text):
        return text.split(" ") if text else []


_scorer = rouge_scorer.RougeScorer(["rougeL"], tokenizer=_PretokenizedTokenizer())


def _tokens(text: str, language: str) -> list[str]:
    if language == "Chinese":
        return [t for t in jieba.lcut(text) if t.strip()]
    return _WORD.findall(text.lower())


def score(reference: str, prediction: str, language: str) -> dict:
    ref = " ".join(_tokens(reference, language))
    pred = " ".join(_tokens(prediction, language))
    s = _scorer.score(ref, pred)["rougeL"]
    return {"prec": s.precision, "rec": s.recall, "f1": s.fmeasure}


def load_records() -> dict:
    recs = {}
    with open(COMBINED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs[r["track_id"]] = {"language": r["language"], "reference": r["reference"]}
    return recs


def oracle_preds() -> dict:
    """{track_id: text} from the gold-labeled block-text join -- the same assembly
    a model's prediction uses, so the oracle bounds the model."""
    labels = {}
    with open(LABELS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            labels[r["track_id"]] = r["labels"]
    preds = {}
    with open(BLOCKS, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            lab = labels[b["track_id"]]
            assert len(lab) == len(b["blocks"]), f"labels/blocks length mismatch {b['track_id']}"
            preds[b["track_id"]] = join_selected(b["blocks"], lab)
    return preds


def selectall_preds() -> dict:
    """{track_id: text} with every block selected -- the floor. Same block-text
    join as oracle_preds, labels all 1."""
    preds = {}
    with open(BLOCKS, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            preds[b["track_id"]] = join_selected(b["blocks"])
    return preds


def probs_preds(probs_path: Path, threshold: float) -> dict:
    """{track_id: text} from a {track_id, probs} export (infer_board / finetune
    --export-test), blocks selected at threshold. Same adapt(unescape(block))
    assembly as oracle_preds, so a model result is scored exactly as the ceiling."""
    blocks = {}
    with open(BLOCKS, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            blocks[b["track_id"]] = b["blocks"]
    probs = {}
    with open(probs_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            probs[r["track_id"]] = r["probs"]
    missing = set(blocks) - set(probs)
    assert not missing, f"{len(missing)} pages missing from probs"
    preds = {}
    for tid, blk in blocks.items():
        pr = probs[tid]
        assert len(pr) == len(blk), f"probs/blocks mismatch {tid}: {len(pr)} vs {len(blk)}"
        preds[tid] = join_selected(blk, [v > threshold for v in pr])
    return preds


def aggregate(rows: list[dict]) -> tuple[dict, dict, dict]:
    by_lang = {l: [r for r in rows if r["language"] == l] for l in LANGS}
    table = {}
    for l in LANGS:
        rs = by_lang[l]
        n = len(rs)
        table[l] = {"n": n} | {k: sum(r[k] for r in rs) / n for k in ("prec", "rec", "f1")}
    macro = {k: sum(table[l][k] for l in LANGS) / len(LANGS) for k in ("prec", "rec", "f1")}
    micro = {k: sum(r[k] for r in rows) / len(rows) for k in ("prec", "rec", "f1")}
    return table, macro, micro


def print_table(tag: str, table: dict, macro: dict, micro: dict) -> None:
    print(f"\n{tag}")
    print(f"  {'':5} {'n':>5}   {'P':>6} {'R':>6} {'F1':>6}")
    for l in LANGS:
        m = table[l]
        print(f"  {ISO[l]:5} {m['n']:>5}   {m['prec']:.4f} {m['rec']:.4f} {m['f1']:.4f}")
    n_all = sum(table[l]["n"] for l in LANGS)
    print(f"  {'macro':5} {'':>5}   {macro['prec']:.4f} {macro['rec']:.4f} {macro['f1']:.4f}")
    print(f"  {'micro':5} {n_all:>5}   {micro['prec']:.4f} {micro['rec']:.4f} {micro['f1']:.4f}")


def score_all(preds: dict, recs: dict) -> list[dict]:
    return [{"track_id": tid, "language": recs[tid]["language"],
             **score(recs[tid]["reference"], preds[tid], recs[tid]["language"])}
            for tid in recs]


def report(tag: str, rows: list[dict], out_path: Path) -> dict:
    table, macro, micro = aggregate(rows)
    print_table(tag, table, macro, micro)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"per-record scores: {out_path}")
    return macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, default=None,
                    help="{track_id: text} json (oracle ceiling when omitted)")
    ap.add_argument("--probs", type=str, default=None,
                    help="{track_id, probs} jsonl from infer_board / finetune export")
    ap.add_argument("--selectall", action="store_true",
                    help="score every block selected -- the floor")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="decision threshold for --probs (set on WMB val, never on DAnIEL)")
    args = ap.parse_args()

    recs = load_records()
    assert len(recs) == 1689, f"{len(recs)} records != 1689"

    if args.selectall:
        preds = selectall_preds()
        assert set(preds) == set(recs), "blocks track_ids != records"
        report("SELECTALL/daniel1689", score_all(preds, recs),
               EVAL / "daniel_rougeL_selectall.jsonl")
        return

    if args.probs:
        preds = probs_preds(Path(args.probs), args.threshold)
        assert set(preds) == set(recs), "blocks track_ids != records"
        stem = Path(args.probs).stem
        thr = "" if args.threshold == 0.5 else f"_thr{args.threshold:g}"
        report(f"MODEL/{stem}{thr}/daniel1689", score_all(preds, recs),
               EVAL / f"daniel_rougeL_{stem}{thr}.jsonl")
        return

    if args.preds:
        preds = json.loads(Path(args.preds).read_text(encoding="utf-8"))
        missing = set(recs) - set(preds)
        assert not missing, f"{len(missing)} records missing from preds"
        tag = Path(args.preds).stem
        report(f"MODEL/{tag}/daniel1689", score_all(preds, recs),
               EVAL / f"daniel_rougeL_{tag}.jsonl")
        return

    preds = oracle_preds()
    assert set(preds) == set(recs), "labels/blocks track_ids != records"
    macro = report("ORACLE/daniel1689", score_all(preds, recs),
                   EVAL / "daniel_rougeL_oracle.jsonl")
    if ORACLE_CEILING is None:
        print(f"record  ORACLE_CEILING = {macro['f1']:.4f}")
    else:
        assert round(macro["f1"], 4) == ORACLE_CEILING, \
            f"oracle ceiling {macro['f1']:.4f} != {ORACLE_CEILING}"
        print(f"ceiling gate OK: {macro['f1']:.4f} == {ORACLE_CEILING}")


if __name__ == "__main__":
    main()
