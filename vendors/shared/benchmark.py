"""The Benchmark seam: everything a trainer needs from one board, behind one object.
A trainer holds a Benchmark and never touches ROUGE, serialize, or a board's
feats.jsonl, so fit and the mode trainers stay board-agnostic. Each board implements
this in its adapter/benchmark.py. fit itself takes only prepared pages and a score_val
callable -- never the Benchmark -- so the training loop knows no board at all.
"""
from typing import Protocol


class Benchmark(Protocol):
    name: str                                    # "wmb" / "wcxb" -- for result paths + messages
    holdout_groups: tuple                        # leave-one-out training groups (DAnIEL langs); () = unsupported
    supports_train_limit: bool                   # the data-efficiency ladder (WMB pool prefixes)

    def build_pages(self, tok, fold, window, cap=0, feats=None, limit=None, ids=None) -> list[dict]:
        """This board's fold -> prepared pages: empty renders dropped, z-scored
        structural features attached per page when `feats` is a loaded feature dict."""

    def build_val_scorer(self, val_pages, workers=1):
        """Return score_val(pages, probs) -> float, this board's per-epoch val metric
        (e.g. WMB ROUGE-5 over serialize'd predictions). Caches its references once."""

    def score_fold(self, probs, fold, thr) -> dict:
        """{n, prec, rec, f1} over the fold's referenced records at threshold thr --
        the board's F1 for a probs set. Backs finetune's --export-test."""

    def score_text(self, outputs, fold) -> dict:
        """{n, prec, rec, f1} for {tid: (text, kind)} -- an extractor's STANDARD output
        judged by the board's metric (bench/speed/cpu_compare.py). The board converts as its metric
        needs (WMB html2texts html-kind); ours and the baselines score the same way."""

    def export_probs(self, fold, tok, infer_fn, out_path, window, cap=0, feats=None,
                     limit=None) -> dict:
        """{track_id: probs} for every fold id (empties -> []), also written to out_path
        as {track_id, probs} jsonl (what eval.py --probs scores). infer_fn(page) -> per-
        block probs wraps the trainer's model, so the benchmark stays model-free."""

    def raw_features(self) -> dict:
        """{track_id: [n_blocks, K]} raw features over the FULL A/B/C layout, as written
        to feats.jsonl (empties dropped). The un-sliced, un-z-scored base: load_features
        slices + z-scores it per group, and cross-board carried-z (bench/accuracy/predict.py)
        applies a foreign ckpt's stats to it."""

    def load_features(self, group="ABC") -> tuple:
        """(raw, z-scored) features for a group, its columns sliced from the full set --
        raw feeds XGBoost, z-scored (train-fold stats) feeds torch heads. group defaults
        to the full A/B/C set; the frozen sweep passes A/AB/ABC/BC/C."""

    def feature_stats(self) -> tuple:
        """Train-fold z-score stats (idx, mean, std) over the full feature layout -- what
        a saved table ckpt carries so inference normalizes any board's features by this
        board's training yardstick. Board-specific (its own train fold), so it lives here."""


# Board registry: name -> Benchmark. Lazy per-board imports keep this a load-time leaf,
# so importing the Protocol never drags in a board.
def load_benchmark(name, holdout=None):
    if name == "wmb":
        from vendors.wmb.adapter.benchmark import WMBBenchmark
        return WMBBenchmark()
    if name == "wcxb":
        from vendors.wcxb.adapter.benchmark import WCXBBenchmark
        return WCXBBenchmark()
    if name == "daniel":
        from vendors.daniel.adapter.benchmark import DanielBenchmark
        return DanielBenchmark(holdout=holdout)
    raise SystemExit(f"unknown benchmark: {name}")
