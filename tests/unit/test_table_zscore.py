"""Unit: a table run on a train_limit subset carries the z-score it trained with. The
ckpt's stats, applied the way the keeper path scores (bench_ops.carried_features), give
back the exact features the head trained on, not the full fold's scaling.
"""
import numpy as np

from core.training import _ORDER, GROUPS
from trainers.recipes import Ctx
from trainers.recipes.table import _feats
from vendors.shared import bench_ops

K = sum(len(GROUPS[g]) for g in _ORDER)       # the full feats.jsonl layout


class _Bench:
    def __init__(self, raw, train_ids):
        self.raw, self.train_ids = raw, train_ids

    def raw_features(self):
        return self.raw

    def load_features(self, g):
        return bench_ops.load_features(self.raw, self.train_ids, g)

    def feature_stats(self, g):
        return bench_ops.feature_stats(self.raw, self.train_ids, g)


def test_subset_run_carries_the_subset_zscore():
    rng = np.random.default_rng(0)
    raw = {f"t{i}": (rng.standard_normal((5, K)) * (1 + i)).astype(np.float32)
           for i in range(20)}
    b = _Bench(raw, sorted(raw))
    sub = ["t0", "t1", "t2"]
    cfg = {"feats": "ABC", "board": "x", "train_limit": 3, "extra": None}
    z, carried = _feats(Ctx(dev=None, folder=None, probs=None), cfg, b,
                        [{"tid": t} for t in sub])
    scored = bench_ops.carried_features(b, {"feats": "ABC", "zscore": carried})
    for t in raw:
        assert np.allclose(scored[t], z[t], atol=1e-5)
    full = bench_ops.carried_features(b, {"feats": "ABC", "zscore": b.feature_stats("ABC")})
    assert not np.allclose(full["t0"], z["t0"])


def test_init_run_uses_the_keepers_zscore_not_a_refit(monkeypatch):
    from trainers.recipes import table
    rng = np.random.default_rng(1)
    raw = {f"t{i}": (rng.standard_normal((5, K)) * (1 + i)).astype(np.float32)
           for i in range(20)}
    keeper_stats = bench_ops.feature_stats(raw, ["t7", "t8", "t9"], "ABC")
    monkeypatch.setattr(table, "_init", lambda cfg: {"feats": "ABC", "zscore": keeper_stats})
    b = _Bench(raw, sorted(raw))
    cfg = {"feats": "ABC", "board": "x", "train_limit": 3, "extra": None, "init": "k.pt"}
    z, carried = _feats(Ctx(dev=None, folder=None, probs=None), cfg, b,
                        [{"tid": t} for t in ("t0", "t1", "t2")])
    assert carried is keeper_stats
    want = bench_ops.carried_features(b, {"feats": "ABC", "zscore": keeper_stats})
    assert all(np.array_equal(z[t], want[t]) for t in raw)
