"""Gate: the built mini ONNX graphs reproduce the torch mini bit-for-bit (0 flips at the
0.5 threshold). Same value-independence as tests/gate/test_pool -- random ids / membership /
features are a complete check -- so a tiny offline table embedder + head stand in for granite;
export/gate.parity runs the shipped load_mini forward against the torch reference. §4.
"""
import torch.nn as nn

from core.model import TableEmbedder, build_head
from export.build import build_head_onnx, build_table_int8
from export.gate import parity


def test_mini_onnx_parity_zero_flips(tmp_path):
    H, V, K, hidden = 32, 200, 3, 16
    emb = TableEmbedder(nn.Embedding(V, H), nn.LayerNorm(H, bias=False, eps=1e-12), H)
    d_in = H + K
    head = build_head("bigru", d_in=d_in, hidden=hidden)
    ck = {"head_state": head.state_dict(), "hidden": hidden}

    table_p, head_p = tmp_path / "table.onnx", tmp_path / "head.onnx"
    build_table_int8(emb, table_p)                 # dequantizes emb in place -> int8 reference
    ref_head = build_head_onnx(ck, d_in, head_p)

    flips, n = parity(table_p, head_p, emb, ref_head, d_in)
    assert n > 0 and flips == 0
