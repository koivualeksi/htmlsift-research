"""
Gate: band attention is the SAME arithmetic as the dense sliding-window path,
reordered into O(L*w) chunks -- gated on numeric identity, not a score
(core/band.py). Three offline checks on a tiny random-weight ModernBERT, no
backbone download (the harness stubs AutoModel; here the config builds a real
but tiny ModernBertModel):
  (0) our band mask IS transformers' bidirectional sliding-window mask,
  (1) the chunked kernel reproduces dense-masked sdpa to float64 exactness,
  (2) enable() leaves a full encoder forward numerically identical over a page
      past MIN_CHUNKED (the chunked branch).
The real-stack fp32 reordering and the end-to-end decision-flip check run in the
keeper screen, where the granite weights are already loaded.
"""
import copy

import pytest
import torch
import torch.nn.functional as F
from transformers import ModernBertConfig, ModernBertModel
from transformers.masking_utils import create_bidirectional_sliding_window_mask

from core import band


def _tiny_config():
    return ModernBertConfig(
        vocab_size=100, hidden_size=64, num_hidden_layers=4,
        num_attention_heads=4, intermediate_size=128,
        global_attn_every_n_layers=2, local_attention=16, pad_token_id=0)


@pytest.mark.parametrize("L", [200, 777, 4096])
def test_band_mask_is_their_sliding_window(L):
    cfg = _tiny_config()
    cfg._attn_implementation = "sdpa"
    h = torch.zeros(1, L, cfg.hidden_size)
    ref = create_bidirectional_sliding_window_mask(
        config=cfg, inputs_embeds=h,
        attention_mask=torch.ones(1, L, dtype=torch.long))
    mine = band._dense_band_mask(L, cfg.sliding_window, h.device)
    assert bool((ref == mine).all())


class _Mod:
    class config:
        sliding_window = 64


@pytest.mark.parametrize("L", [1024, 2048, 3000, 8192])
def test_band_kernel_exact_at_float64(L):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, L, 32, dtype=torch.float64) for _ in range(3))
    ref = F.scaled_dot_product_attention(
        q, k, v, attn_mask=band._dense_band_mask(L, 64, q.device),
        scale=32 ** -0.5, is_causal=False).transpose(1, 2).contiguous()
    got, _ = band.band_sdpa_forward(_Mod, q, k, v, None, scaling=32 ** -0.5,
                                    sliding_window=65)
    assert float((ref - got).abs().max()) < 1e-12


def test_enable_is_numerically_identical():
    torch.manual_seed(0)
    enc = ModernBertModel(_tiny_config()).eval()
    ids = torch.randint(0, 100, (1, 1000))          # > MIN_CHUNKED -> chunked path
    mask = torch.ones_like(ids)
    with torch.no_grad():
        ref = enc(input_ids=ids, attention_mask=mask).last_hidden_state
        b = band.enable(copy.deepcopy(enc))
        got = b(input_ids=ids, attention_mask=mask).last_hidden_state
    assert b._band_enabled
    assert float((ref - got).abs().max() / ref.abs().max()) < 1e-4
