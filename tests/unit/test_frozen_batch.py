"""The batched frozen-head forward must be per-block identical to the unbatched
per-page forward -- packed GRU and masked attention make padding a no-op. That
identity is the correctness guarantee behind trainers.frozen_screen's batching: only the
TRAINING numbers move (mini-batch gradient vs per-page SGD); the forward must not.
"""
import numpy as np
import torch

from trainers.frozen_screen import _batches, _forward
from core.model import build_head

DEV = torch.device("cpu")


def _heads(d):
    return {
        "linear": build_head("linear", d_in=d),
        "bigru": build_head("bigru", d_in=d, hidden=8),
        "transformer": build_head("transformer", d_in=d, d_model=32, nhead=4,
                                  layers=1, window=64),
    }


def _per_page(head, xs):
    with torch.no_grad():
        return [head(torch.from_numpy(x)[None])[0].numpy() for x in xs]


def _batched(head, xs, batch_size, max_len):
    out = [None] * len(xs)
    order = sorted(range(len(xs)), key=lambda i: len(xs[i]))
    with torch.no_grad():
        for idx, packed in _batches(xs, order, batch_size, max_len):
            p = _forward(head, xs, idx, packed, DEV)[0].numpy()
            for r, k in enumerate(idx):
                out[k] = p[r][:len(xs[k])]
    return out


def test_batched_forward_matches_per_page():
    torch.manual_seed(0)
    d = 16
    rng = np.random.default_rng(0)
    xs = [rng.standard_normal((n, d)).astype(np.float32) for n in [5, 17, 1, 40, 8, 8, 23]]
    for kind, head in _heads(d).items():
        head.eval()
        a, b = _per_page(head, xs), _batched(head, xs, batch_size=3, max_len=2048)
        for i, (x, y) in enumerate(zip(a, b)):
            assert x.shape == y.shape, (kind, i, x.shape, y.shape)
            assert np.allclose(x, y, atol=1e-4), (kind, i, float(np.abs(x - y).max()))


def test_train_head_frozen_runs():
    # the full batched training path (warmup + grad-clip + masked BCE + val) runs
    # and returns a best {val, epoch} for every head kind.
    from trainers.frozen_screen import train_head_frozen
    torch.manual_seed(0)
    d = 16
    rng = np.random.default_rng(2)
    train_x = [rng.standard_normal((n, d)).astype(np.float32) for n in [5, 12, 3, 20]]
    train_y = [(rng.random(len(x)) > 0.5).astype(np.float32) for x in train_x]
    val_x = [rng.standard_normal((n, d)).astype(np.float32) for n in [4, 9]]
    val_pages = list(range(len(val_x)))

    def score_val(pages, probs):
        return float(np.mean([p.mean() for p in probs]))

    for kind in ("linear", "bigru", "transformer"):
        best = train_head_frozen(_heads(d)[kind], train_x, train_y, val_x, val_pages,
                                 score_val, DEV, epochs=2, seed=0, batch_size=2)
        assert set(best) == {"val", "epoch"} and 1 <= best["epoch"] <= 2


def test_frozen_head_persists_nothing(tmp_path, monkeypatch):
    # §4 cache invariant: the frozen screen holds pooled embeddings in RAM and writes
    # no embedding cache to disk. Run the head-training path (where the embeddings flow)
    # in an isolated cwd and assert it creates no file -- a guard so a future "cache the
    # embeddings" optimization can't silently reintroduce persisted, provenance-crossing
    # state (root CLAUDE.md §4 cache row).
    from trainers.frozen_screen import train_head_frozen
    monkeypatch.chdir(tmp_path)
    torch.manual_seed(0)
    d = 16
    rng = np.random.default_rng(3)
    train_x = [rng.standard_normal((n, d)).astype(np.float32) for n in [5, 12, 3]]
    train_y = [(rng.random(len(x)) > 0.5).astype(np.float32) for x in train_x]
    val_x = [rng.standard_normal((n, d)).astype(np.float32) for n in [4, 9]]
    val_pages = list(range(len(val_x)))

    def score_val(pages, probs):
        return float(np.mean([p.mean() for p in probs]))

    train_head_frozen(_heads(d)["bigru"], train_x, train_y, val_x, val_pages,
                      score_val, DEV, epochs=2, seed=0, batch_size=2)
    assert not list(tmp_path.iterdir()), [p.name for p in tmp_path.iterdir()]


def test_singleton_long_page_matches():
    # a page longer than max_len is singleton-batched (mask None); the short pages
    # batch together -- both routes must still match per-page.
    torch.manual_seed(0)
    d = 16
    rng = np.random.default_rng(1)
    xs = [rng.standard_normal((n, d)).astype(np.float32) for n in [3, 50, 4]]
    for kind, head in _heads(d).items():
        head.eval()
        a = _per_page(head, xs)
        b = _batched(head, xs, batch_size=8, max_len=10)   # 50-block page singletons
        for x, y in zip(a, b):
            assert np.allclose(x, y, atol=1e-4)
