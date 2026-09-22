"""bench.speed.gpu_throughput.encode_batch must be per-page identical to core.model.encode_window:
padding a short page up to a batch, with the attention mask zeroed on the pad, must not
change a real token's states. That identity is the whole "batching can't move F1" claim --
pooling and the head are deterministic on the hidden states (the head batching is gated
separately in test_frozen_batch), so matching hidden states => matching probs => matching
labels.

A tiny random-init BertModel, not the Linear stub of test_model: the hazard here is real
bidirectional attention leaking across the pad, so the test needs real attention. No
backbone download -- the model is built from a small config in process.
"""
import numpy as np
import torch
from transformers import BertConfig, BertModel

from bench.speed.gpu_throughput import encode_batch
from core.model import encode_window

DEV = torch.device("cpu")


def _enc():
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=100, hidden_size=16, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=32,
                     max_position_embeddings=128)
    return BertModel(cfg).eval()


def test_encode_batch_matches_per_page():
    enc = _enc()
    rng = np.random.default_rng(0)
    # mixed lengths in one batch, incl. repeats and a length-1 page -- the pad rows
    # differ per page, so a mask bug would show as a per-page mismatch.
    ids_list = [rng.integers(1, 100, size=n).astype(np.int64) for n in [5, 17, 1, 9, 9, 23]]
    with torch.no_grad():
        ref = [encode_window(enc, torch.from_numpy(x).to(DEV), autocast=False) for x in ids_list]
        bat = encode_batch(enc, ids_list, DEV, autocast=False)
    assert len(ref) == len(bat)
    for i, (r, b) in enumerate(zip(ref, bat)):
        assert r.shape == b.shape, (i, r.shape, b.shape)
        assert torch.allclose(r, b, atol=1e-4), (i, float((r - b).abs().max()))


def test_singleton_batch_is_a_noop():
    # a batch of one has no padding: it must reproduce the per-page forward exactly.
    enc = _enc()
    rng = np.random.default_rng(1)
    x = rng.integers(1, 100, size=12).astype(np.int64)
    with torch.no_grad():
        ref = encode_window(enc, torch.from_numpy(x).to(DEV), autocast=False)
        bat = encode_batch(enc, [x], DEV, autocast=False)[0]
    assert torch.allclose(ref, bat, atol=1e-6), float((ref - bat).abs().max())
