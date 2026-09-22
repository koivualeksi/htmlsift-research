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
