"""Unit: the model builders resolve dtype, truncate to N layers, and size heads
from d_in -- the four infer_wcxb hazards designed out (no fp32->bf16 downcast, no
11-layer / BiGRUHead / model_name hardcode). AutoModel is stubbed, so no backbone
download; stitch_bounds (the window-ownership rule) is pure and checked directly.
"""
import pytest
import torch
import torch.nn as nn

from core import model
from core.model import build_head, stitch_bounds


class _Cfg:
    def __init__(self, h):
        self.hidden_size = h


class _StubBackbone(nn.Module):
    def __init__(self, hidden=8, n_layers=6):
        super().__init__()
        self.config = _Cfg(hidden)
        self.layers = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(hidden)


@pytest.fixture
def stub_automodel(monkeypatch):
    def from_pretrained(name, attn_implementation="sdpa", dtype=torch.float32):
        return _StubBackbone().to(dtype)
    monkeypatch.setattr(model, "AutoModel",
                        type("A", (), {"from_pretrained": staticmethod(from_pretrained)}))


def test_encoder_fp32_and_truncation(stub_automodel, capsys):
    enc = model.build_encoder("stub", n_layers=3, dtype=torch.float32)
    assert len(enc.layers) == 3                                    # truncated, not hardcoded 11
    assert all(p.dtype == torch.float32 for p in enc.parameters())  # no fp32->bf16 downcast
    assert isinstance(enc.final_norm, nn.LayerNorm)                # fresh final norm
    assert enc.final_norm.normalized_shape == (enc.config.hidden_size,)
    assert "param dtype float32" in capsys.readouterr().out        # §4: prints resolved dtype


def test_encoder_layer_range_asserts(stub_automodel):
    with pytest.raises(AssertionError):
        model.build_encoder("stub", n_layers=0)
    with pytest.raises(AssertionError):
        model.build_encoder("stub", n_layers=99)


@pytest.mark.parametrize("d_in", [8, 33 + 8])
def test_head_dims_come_from_d_in(d_in):
    assert build_head("linear", d_in).out.in_features == d_in
    assert build_head("bigru", d_in).gru.input_size == d_in
    assert build_head("transformer", d_in, d_model=16, nhead=2, layers=1).proj.in_features == d_in


@pytest.mark.parametrize("kind, kw", [
    ("linear", {}),
    ("bigru", {}),
    ("transformer", {"d_model": 16, "nhead": 2, "layers": 1}),
])
def test_head_forward_shape(kind, kw):
    head = build_head(kind, 8, **kw)
    assert head(torch.randn(1, 5, 8)).shape == (1, 5)


def test_unknown_head_raises():
    with pytest.raises(ValueError):
        build_head("bogus", 8)


def test_stitch_bounds_tiles_the_page():
    starts, window, n = [0, 4], 6, 10
    assert stitch_bounds(starts, window, 0) == (0, 5)
    assert stitch_bounds(starts, window, 1) == (1, 6)
    covered = []
    for k, s in enumerate(starts):
        lo, hi = stitch_bounds(starts, window, k)
        covered.extend(range(s + lo, s + hi))
    assert covered == list(range(n))            # no gap, no overlap


# ---- low-rank table embedder (core.quant.factor_keeper keepers) ----

def _tiny_table(V=60, H=16, seed=0):
    torch.manual_seed(seed)
    norm = nn.LayerNorm(H, bias=False, eps=1e-12)
    nn.init.uniform_(norm.weight, 0.5, 1.5)
    return model.TableEmbedder(nn.Embedding(V, H), norm, H).eval()


def test_lowrank_embedder_rows_are_the_shipped_int8_rows():
    from core.quant import lowrank_table
    emb, B = _tiny_table(), torch.randn(4, 16)
    lr = model.lowrank_embedder(emb, B)
    assert lr.hidden_size == 4 and isinstance(lr.norm, nn.Identity)
    q, s = lowrank_table(emb.tok.weight, emb.norm, B)
    ids = torch.arange(60)
    assert torch.equal(lr(ids), q.float() * s)                    # loader == build rows


def test_lowrank_keeper_full_rank_matches_source_keeper():
    import numpy as np
    from core.quant import factor_keeper, second_moment
    V, H, K, hidden = 60, 16, 3, 8
    emb = _tiny_table(V, H)
    head = build_head("bigru", d_in=H + K, hidden=hidden).eval()
    ck = {"hidden": hidden, "head_state": head.state_dict()}
    rng = np.random.default_rng(0)

    def page():
        T, nb = int(rng.integers(30, 80)), int(rng.integers(3, 12))
        members = [[] for _ in range(nb)]
        for t in range(T):
            members[int(rng.integers(nb))].append(t)
        return {"ids": rng.integers(0, V, T).astype(np.int64), "members": members,
                "feats": rng.standard_normal((nb, K)).astype(np.float32)}

    C, n = second_moment(emb, [page() for _ in range(40)])
    assert n > 0
    lk = factor_keeper(ck, C, r=H)
    lr = model.lowrank_embedder(emb, lk["proj"])
    lr_head = build_head("bigru", d_in=H + K, hidden=hidden).eval()
    lr_head.load_state_dict(lk["head_state"])
    for _ in range(10):                                           # B moved before the mean
        p = page()
        ref = model.infer_table_page(emb, head, p, torch.device("cpu"))
        got = model.infer_table_page(lr, lr_head, p, torch.device("cpu"))
        assert abs(ref - got).max() < 1e-2                        # int8 rounding only


def test_table_embedder_refuses_int8_with_proj():
    with pytest.raises(ValueError, match="quantize twice"):
        model.build_table_embedder("stub", int8=True, proj=torch.zeros(2, 4))
