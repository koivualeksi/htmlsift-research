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
