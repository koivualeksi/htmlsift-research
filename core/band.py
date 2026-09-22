"""
Chunked band attention for the encoder's sliding-attention layers -- a speed
optimization that changes nothing about the output.

transformers builds the sliding window as a dense (B, 1, L, L) boolean mask and
hands it to scaled_dot_product_attention, whose CPU kernel scores the whole
L x L matrix and then discards the out-of-band entries -- so a sliding layer
costs as much as a full one and local_attention buys nothing. This computes the
same numbers in O(L*w) by cutting the sequence into chunks and letting each
query chunk attend only to the key slice it can reach.

The arithmetic is unchanged: the per-chunk predicate is exactly
abs(q_idx - kv_idx) <= config.sliding_window, the same one transformers' sliding
mask applies. So this is gated on numeric identity, not on a val score
(tests/gate/test_band.py checks |delta| < 1e-12 on the kernel, < 1e-4 on the fp32
end-to-end) -- the speed-only lever of the optimization set (root CLAUDE.md
section 4). The int8 levers that DO change the
numbers live in core/quant.py and compose on top: quantize FIRST, then enable()
here (reverse order silently runs fp32 -- see _no_mask_hook).

    import core.band
    core.band.enable(encoder)      # in place, idempotent
"""
import torch
import torch.nn.functional as F
from transformers.integrations.sdpa_attention import sdpa_attention_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

IMPL = "sdpa_band"
CHUNK = 128
# Below this the reshaping costs more than the scores it saves, and the dense
# band mask is only a few hundred KB -- exactly what transformers would build.
MIN_CHUNKED = 4 * CHUNK
_NO_MASK = {"full_attention": None, "sliding_attention": None}


def _dense_band_mask(n_tok, half, device):
    """(1, 1, L, L) bool -- the sliding mask transformers would have built."""
    i = torch.arange(n_tok, device=device)
    return ((i[None, :] - i[:, None]).abs() <= half)[None, None]


def _chunk_mask(chunk, half, n_chunks, n_tok, device):
    """(n_chunks, 1, chunk, chunk + 2*half) bool: inside the band AND a real
    token. A query row past the sequence end has no valid key and softmax over an
    all-False row is NaN, so each dead row gets its own zero slot; those rows are
    sliced off before the result returns."""
    span = chunk + 2 * half
    j = torch.arange(chunk, device=device)
    t = torch.arange(span, device=device)
    band = (t[None, :] - half - j[:, None]).abs() <= half
    c = torch.arange(n_chunks, device=device)
    kpos = c[:, None] * chunk - half + t[None, :]
    m = band[None] & ((kpos >= 0) & (kpos < n_tok))[:, None, :]
    dead = (c[:, None] * chunk + j[None, :]) >= n_tok
    if bool(dead.any()):
        ci, ji = torch.nonzero(dead, as_tuple=True)
        m[ci, ji, half + ji] = True
    return m[:, None]


def band_sdpa_forward(module, query, key, value, attention_mask, dropout=0.0,
                      scaling=None, sliding_window=None, **kwargs):
    """Drop-in for sdpa_attention_forward. The band path runs only for a sliding
    layer with nothing to pad; a full layer (sliding_window is None) or any real
    padding mask is delegated to stock sdpa, unchanged."""
    if sliding_window is None or attention_mask is not None:
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling,
                                      sliding_window=sliding_window, **kwargs)

    half = int(module.config.sliding_window)
    B, H, L, D = query.shape

    if L <= MIN_CHUNKED:
        out = F.scaled_dot_product_attention(
            query, key, value, attn_mask=_dense_band_mask(L, half, query.device),
            dropout_p=dropout, scale=scaling, is_causal=False)
        return out.transpose(1, 2).contiguous(), None

    chunk = max(CHUNK, half)
    span = chunk + 2 * half
    pad = (-L) % chunk
    n = (L + pad) // chunk

    q = F.pad(query, (0, 0, 0, pad)).view(B, H, n, chunk, D)
    q = q.permute(0, 2, 1, 3, 4).reshape(B * n, H, chunk, D)
    kv = []
    for x in (key, value):
        x = F.pad(x, (0, 0, half, pad + half)).unfold(2, span, chunk)
        kv.append(x.permute(0, 2, 1, 4, 3).reshape(B * n, H, span, D))

    m = _chunk_mask(chunk, half, n, L, query.device)
    if B > 1:
        m = m.repeat(B, 1, 1, 1)

    out = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=m,
                                         dropout_p=dropout, scale=scaling,
                                         is_causal=False)
    out = out.view(B, n, H, chunk, D).permute(0, 2, 1, 3, 4)
    out = out.reshape(B, H, n * chunk, D)[:, :, :L]
    return out.transpose(1, 2).contiguous(), None


def register():
    ALL_ATTENTION_FUNCTIONS.register(IMPL, band_sdpa_forward)
    # Only reached on the fallback path (a real padding mask), but a missing key
    # here would be a KeyError deep inside mask construction.
    ALL_MASK_ATTENTION_FUNCTIONS.register(IMPL, sdpa_mask)


def _no_mask_hook(module, args, kwargs):
    """Replace an all-ones 2D mask with the "no mask" mapping so the model never
    materialises the dense (1, 1, L, L) sliding mask -- the very thing the band
    path exists to avoid. ModernBertModel.forward uses a mask dict verbatim when
    given one, so {'full_attention': None, 'sliding_attention': None} makes every
    layer receive None and band_sdpa_forward take the fast path. Anything that is
    NOT an all-ones 2D mask (real padding) is left alone: the dense mask is built
    and band_sdpa_forward falls back to stock sdpa. The fast path can therefore
    only run when there is nothing to pad.

    A forward_pre_hook, NOT a wrapped encoder.forward, and that is load-bearing.
    torch.ao.quantization.quantize_dynamic returns a deep copy; a closure stored
    as encoder.forward keeps pointing at the original fp32 module, so enable()
    then quantize would silently run fp32 while reporting every layer quantized.
    Hooks live in _forward_pre_hooks, are copied with the module, and take module
    as an argument instead of capturing it.
    """
    am = kwargs.get("attention_mask")
    if am is None or (torch.is_tensor(am) and am.dim() == 2 and bool(am.all())):
        kwargs["attention_mask"] = _NO_MASK
        return args, kwargs
    return None


def enable(encoder):
    """Switch a ModernBERT encoder to band attention, in place. Idempotent."""
    register()
    encoder.config._attn_implementation = IMPL
    for m in encoder.modules():          # layers may hold their own config copy
        cfg = getattr(m, "config", None)
        if cfg is not None:
            cfg._attn_implementation = IMPL
    if not getattr(encoder, "_band_enabled", False):
        encoder.register_forward_pre_hook(_no_mask_hook, with_kwargs=True)
        encoder._band_enabled = True
    return encoder
