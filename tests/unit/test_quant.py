"""
Unit: the int8 primitives (core/quant.py). Pure-op checks need no model; the two
structural checks build a tiny random-weight ModernBERT offline. Accuracy of the
schemes is a keeper-screen question -- here we gate the invariants that make them
correct and deployable: STE lets gradient flow through round(), the QAT wrapper
leaves state_dict keys identical (so the checkpoint stays fp32-loadable), and
emb-int8 reproduces int8 storage exactly and is idempotent.
"""
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertModel

from core import quant
from core.quant import QATW8A8Linear, _quant_ste_act, _quant_ste_weight


def _tiny_encoder():
    return ModernBertModel(ModernBertConfig(
        vocab_size=100, hidden_size=64, num_hidden_layers=4,
        num_attention_heads=4, intermediate_size=128,
        global_attn_every_n_layers=2, local_attention=16, pad_token_id=0))


def test_qat_linear_reuses_params_and_matches_arithmetic():
    lin = nn.Linear(8, 4)
    q = QATW8A8Linear.wrap(lin)
    assert q.weight is lin.weight and q.bias is lin.bias      # state_dict keys identical
    x = torch.randn(2, 8)
    ref = F.linear(_quant_ste_act(x), _quant_ste_weight(q.weight), q.bias)
    assert torch.equal(q(x), ref)


def test_ste_passes_gradient_through_round():
    lin = nn.Linear(8, 4)
    q = QATW8A8Linear.wrap(lin)
    q(torch.randn(2, 8)).sum().backward()
    g = q.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_patch_qat_preserves_state_dict_keys():
    enc = _tiny_encoder()
    before = set(enc.state_dict().keys())
    expected = sum(1 for m in enc.layers.modules() if isinstance(m, nn.Linear))
    quant.patch_qat_w8a8(enc)
    after = enc.state_dict()
    assert set(after.keys()) == before                        # fp32-loadable checkpoint
    assert sum(1 for m in enc.layers.modules()
               if isinstance(m, QATW8A8Linear)) == expected


def test_ptq_dynamic_converts_body_linears():
    q = quant.ptq_dynamic(_tiny_encoder())
    assert sum(1 for m in q.modules()
               if type(m).__name__ == "LinearPackedParams") > 0


def test_emb_int8_reproduces_storage_and_is_idempotent():
    emb = nn.Embedding(50, 8)
    enc = types.SimpleNamespace(embeddings=types.SimpleNamespace(tok_embeddings=emb))
    w0 = emb.weight.data.clone()
    quant.emb_int8_fakequant(enc)
    s = (w0.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    stored = torch.round(w0 / s).clamp(-127, 127).to(torch.int8).float() * s
    assert torch.equal(emb.weight.data, stored)               # == int8 store then dequant
    assert quant.emb_int8_fakequant(enc) == 0.0               # snapping to grid is idempotent


# ---- low-rank table: factor, derived keeper, int8 rows ----

def _spd(H, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * H, H, generator=g, dtype=torch.float64)
    return (X.T @ X).numpy()


def test_factor_full_rank_reproduces_W():
    H, G = 16, 24
    W = torch.randn(G, H, dtype=torch.float64).numpy()
    A, B = quant.factor(W, _spd(H), r=H)
    assert A.shape == (G, H) and B.shape == (H, H)
    assert abs(A @ B - W).max() < 1e-8


def test_factor_keeper_replaces_only_embedding_columns():
    from core.model import build_head
    H, K, hidden, r = 16, 3, 8, 4
    head = build_head("bigru", d_in=H + K, hidden=hidden)
    ck = {"kind": "table", "hidden": hidden, "feats": "ABC",
          "head_state": {k: v.clone() for k, v in head.state_dict().items()}}
    orig = {k: v.clone() for k, v in ck["head_state"].items()}
    lk = quant.factor_keeper(ck, _spd(H), r)
    assert lk["rank"] == r and tuple(lk["proj"].shape) == (r, H)
    assert lk["feats"] == "ABC" and "proj" not in ck                     # source untouched
    assert all(torch.equal(ck["head_state"][k], orig[k]) for k in orig)
    for k, v in orig.items():
        if k in quant.GRU_IH:
            assert lk["head_state"][k].shape == (v.shape[0], r + K)
            assert torch.equal(lk["head_state"][k][:, r:], v[:, H:])     # feature columns
        else:
            assert torch.equal(lk["head_state"][k], v)                   # hh, biases, out
    build_head("bigru", d_in=r + K, hidden=hidden).load_state_dict(lk["head_state"])


def test_lowrank_table_is_int8_within_half_a_step():
    V, H, r = 50, 16, 4
    w = torch.randn(V, H)
    norm = nn.LayerNorm(H, bias=False, eps=1e-12)
    nn.init.uniform_(norm.weight, 0.5, 1.5)
    B = torch.randn(r, H)
    q, s = quant.lowrank_table(w, norm, B, chunk=7)                     # chunk edge covered
    assert q.dtype == torch.int8 and q.shape == (V, r) and s.shape == (V, 1)
    assert int(q.abs().max()) <= 127
    with torch.no_grad():
        ref = norm(w) @ B.T
    assert ((q.float() * s - ref).abs() <= s / 2 + 1e-6).all()
