"""Build the mini artifact's two ONNX graphs from a trained keeper (torch-side, one-time).

build_table_int8 hand-builds the int8-stored token-embedding graph (ORT's quantizers skip
the Gather an embedding is); build_table_lowrank the r-wide one of a low-rank keeper (proj);
build_head_onnx exports the trained BiGRU head. Both return the
torch reference object (int8-dequantized embedder / the head) so export/gate.py can check the
built graph reproduces torch bit-for-bit before anything ships. Not a runtime import -- the
production repo runs export/infer.py, never this.
"""
import copy
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from htmlsift.core.features import feature_dim
from core.model import build_head, build_table_embedder, lowrank_embedder
from core.quant import lowrank_table
from export.paths import mini_paths

IR_VERSION = 10                                     # onnxruntime CUDA-12 build (1.22) supports max IR 10


def _table_graph(q, scale, out, ln_w=None, eps=None):
    """Write the int8 table graph: int8 rows q [V, W] + per-row fp32 scale [V, 1] as
    initializers, gather -> cast fp32 -> *scale, then LayerNorm in-graph only when ln_w
    is given (a low-rank table has it folded into its rows)."""
    inits = [numpy_helper.from_array(q, "table_int8"),
             numpy_helper.from_array(scale.astype(np.float32), "scale")]
    nodes = [helper.make_node("Gather", ["table_int8", "ids"], ["g_i8"], axis=0),
             helper.make_node("Cast", ["g_i8"], ["g_f"], to=TensorProto.FLOAT),
             helper.make_node("Gather", ["scale", "ids"], ["sc"], axis=0),
             helper.make_node("Mul", ["g_f", "sc"], ["deq" if ln_w is not None else "out"])]
    if ln_w is not None:
        inits.append(numpy_helper.from_array(ln_w.astype(np.float32), "ln_w"))
        nodes.append(helper.make_node("LayerNormalization", ["deq", "ln_w"], ["out"],
                                      axis=-1, epsilon=float(eps)))
    graph = helper.make_graph(
        nodes, "mini_table_int8",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, ["t"])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["t", q.shape[1]])], inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = IR_VERSION
    onnx.checker.check_model(m)
    onnx.save(m, str(out))


def build_table_int8(emb, out):
    """The int8-stored token-embedding graph from a fp32 table embedder
    (build_table_embedder): per-row symmetric int8 weights + per-row fp32 scale as
    initializers, then gather -> cast fp32 -> *scale -> LayerNorm in-graph. Arm-independent
    (the table is frozen granite). Hand-built because ORT's quantizers skip the Gather an
    embedding is. Scale/round match core.quant.emb_int8_fakequant. Dequantizes `emb` IN PLACE
    so the caller holds the exact torch int8 reference the graph encodes (the gate compares
    against it)."""
    w = emb.tok.weight.data
    scale = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    table = torch.round(w / scale).clamp(-127, 127).to(torch.int8).numpy()
    assert emb.norm.bias is None                    # granite embeddings.norm: no bias, eps 1e-12
    _table_graph(table, scale.numpy(), out, emb.norm.weight.detach().numpy(), emb.norm.eps)
    emb.tok.weight.data.copy_(torch.from_numpy(table.astype(np.float32)) * scale)   # int8 ref
    return emb


def build_table_lowrank(emb, proj, out):
    """The r-wide int8 table graph of a low-rank keeper from the fp32 table embedder: the
    core.quant.lowrank_table rows, no LayerNorm node (folded in). Returns
    core.model.lowrank_embedder(emb, proj) -- the very reference core.loader runs -- so
    the gate checks the graph against what was scored, not a build-side copy."""
    q, scale = lowrank_table(emb.tok.weight, emb.norm, proj)
    _table_graph(q.numpy(), scale.numpy(), out)
    return lowrank_embedder(emb, proj)


def build_head_onnx(ck, d_in, out):
    """The trained BiGRU head as ONNX (pooled+features [1, n, d_in] -> logits [1, n]).
    eval() so dropout is identity; the head's mask arg defaults None (per-page inference).
    Returns the torch head (the self-gate's reference)."""
    head = build_head("bigru", d_in=d_in, hidden=ck.get("hidden", 256))
    head.load_state_dict(ck["head_state"])
    head.eval()

    class Wrap(torch.nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h

        def forward(self, x):
            return self.h(x)

    # export a copy: torch.onnx.export mutates an nn.GRU in place (flatten_parameters
    # during tracing), so exporting head itself would corrupt the reference the gate reuses.
    torch.onnx.export(Wrap(copy.deepcopy(head)), (torch.zeros(1, 64, d_in),), str(out),
                      input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {1: "n"}, "logits": {1: "n"}},
                      opset_version=17, dynamo=False)
    m = onnx.load(str(out))
    m.ir_version = IR_VERSION
    onnx.save(m, str(out))
    return head


def build_mini(ck, out_dir):
    """Build both mini graphs under out_dir from keeper ck; return (table_p, head_p, emb,
    head, d_in). d_in = table width + feature width -- the width the head was trained with
    (r for a low-rank keeper), derived here so release.py and the parity test never re-derive it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_p, head_p = mini_paths(out_dir)
    emb = build_table_embedder(ck["model"], int8=False)
    if ck.get("proj") is not None:
        emb = build_table_lowrank(emb, ck["proj"], table_p)
    else:
        build_table_int8(emb, table_p)
    d_in = emb.hidden_size + feature_dim(ck.get("feats"))
    head = build_head_onnx(ck, d_in, head_p)
    return table_p, head_p, emb, head, d_in
