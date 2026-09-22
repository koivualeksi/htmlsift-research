"""Unit: the XGBoost head builds a per-block design matrix carrying neighbour,
page-mean and position context, then fits and predicts per-block keep
probabilities in the {track_id: probs} shape the board --probs eval reads.
Features go in raw (trees are scale-invariant); a trivially separable label is
recovered on held-out pages, which also proves fit_predict wires the
train -> eval page spans correctly.
"""
import numpy as np

from trainers.xgb import fit_predict, page_matrix


def test_page_matrix_shape_and_edges():
    feats = np.arange(12, dtype=np.float32).reshape(4, 3)   # 4 blocks, K=3
    M = page_matrix(feats)
    assert M.shape == (4, 4 * 3 + 3)                        # own+prev+next+mean + 3 pos
    assert np.array_equal(M[0, 3:6], feats[0])              # prev of block 0 clamps to itself
    assert np.array_equal(M[3, 6:9], feats[3])              # next of last block clamps to itself
    assert np.allclose(M[:, -3], [0, 1 / 3, 2 / 3, 1])      # normalised index
    assert np.all(M[:, -2] == 4)                            # page length
    assert np.array_equal(M[:, -1], [0, 1, 2, 3])           # raw index


def test_page_matrix_appends_embedding():
    feats = np.zeros((5, 3), np.float32)
    emb = np.ones((5, 8), np.float32)
    assert page_matrix(feats, emb).shape == (5, 4 * 3 + 3 + 8)
    assert page_matrix(feats, None).shape == (5, 4 * 3 + 3)


def _separable_page(tid, n, rng):
    """A page whose label is (first feature > 0) -- a tree should recover it."""
    feats = rng.standard_normal((n, 3)).astype(np.float32)
    labels = (feats[:, 0] > 0).astype(np.float32)
    return (tid, feats, None, labels)


def test_fit_predict_recovers_separable_label():
    rng = np.random.default_rng(0)
    train = [_separable_page(f"tr{i}", 40, rng) for i in range(6)]
    ev = [_separable_page(f"va{i}", 30, rng) for i in range(2)]
    probs = fit_predict(train, ev, use_emb=False, seed=0)
    assert set(probs) == {"va0", "va1"}
    for tid, feats, _, labels in ev:
        p = probs[tid]
        assert p.shape == (len(labels),)
        assert p.min() >= 0 and p.max() <= 1
        assert ((p > 0.5).astype(np.float32) == labels).mean() > 0.8
