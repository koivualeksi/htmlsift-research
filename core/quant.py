"""
INT8 for the lite (97m-6) CPU arm: the compute path (QAT / PTQ, numerics.md) and
the storage path (embedding table, packaging.md). Two objectives, never one
number -- QAT/PTQ are a speed lever, emb-int8 a size lever.

QAT is the keeper int8 path: quantization-aware training makes the fast per-tensor
activation scheme (the one PTQ lost -1.92 fold on) accuracy-neutral. QATW8A8Linear
subclasses nn.Linear and REUSES its Parameters, so state_dict keys are unchanged
and the checkpoint still loads into a plain fp32 encoder for inference, storing
quantization-robust fp32 masters. PTQ (torch quantize_dynamic) is kept only as the
foil the QAT win is measured against.

emb-int8 is a size lever with ~0 speed (the table is a gather, not a matmul): it
shrinks the ~80%-of-params token embedding table ~4x by storing it per-row int8,
and scored fold-neutral. All int8 here is fake-quant -- the numerics of int8
storage/compute, faithful because a real int8 op is exact given its quantised
inputs -- so these produce admissible F1/size fields, never a speedup on their
own. Speed is measured separately and is host-scoped (root CLAUDE.md section 7).

Compose with band: quantize FIRST, then core.band.enable() (core/band.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- QAT: W8A8 fake-quant for the encoder body, straight-through estimator ----

def _quant_ste_weight(w):
    """Per-output-channel symmetric int8, straight-through. Deployable: the scales
    are static and onednn/fbgemm consume per-channel int8 weights directly."""
    s = (w.detach().abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    wq = torch.clamp(torch.round(w / s), -127, 127) * s
    return w + (wq - w).detach()


def _quant_ste_act(x):
    """Per-tensor affine uint8, scale recomputed each forward (dynamic), straight-
    through. This is the granularity torch.quantize_dynamic runs on CPU -- the fast
    scheme with a real onednn kernel, and the one PTQ lost -1.92 fold on. QAT's
    whole job is to make exactly this tolerable."""
    lo, hi = x.detach().amin(), x.detach().amax()
    s = ((hi - lo).clamp_min(1e-12)) / 255.0
    zp = torch.round(-lo / s)
    xq = (torch.clamp(torch.round(x / s) + zp, 0, 255) - zp) * s
    return x + (xq - x).detach()


class QATW8A8Linear(nn.Linear):
    """nn.Linear that fake-quantizes weight (per-channel int8) and input (per-tensor
    dynamic uint8) in the forward, with straight-through gradients. Subclasses
    nn.Linear and reuses its Parameters, so state_dict keys are IDENTICAL: the
    checkpoint loads into a plain fp32 encoder and stores quantization-robust fp32
    masters. Active in train AND eval, so a run's exported val_probs already
    reflect the quantized forward."""

    @classmethod
    def wrap(cls, lin):
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None)
        m.weight, m.bias = lin.weight, lin.bias          # reuse the fp32 masters
        return m

    def forward(self, x):
        return F.linear(_quant_ste_act(x), _quant_ste_weight(self.weight), self.bias)


def patch_qat_w8a8(encoder):
    """Replace every nn.Linear inside the encoder body (attn Wqkv/Wo, MLP Wi/Wo)
    with a QAT wrapper. Embedding table, attention matmuls, the fresh final
    LayerNorm and the head stay fp32 -- the exact layer set quantize_dynamic hits."""
    n = 0
    for mod in list(encoder.layers.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and not isinstance(child, QATW8A8Linear):
                setattr(mod, name, QATW8A8Linear.wrap(child))
                n += 1
    assert n > 0, "QAT patched no Linear layers"
    print(f"QAT W8A8: {n} body Linear layers fake-quantized (per-channel int8 "
          f"weight + per-tensor dynamic uint8 act, STE)", flush=True)
    return encoder


# ---- PTQ: torch dynamic quantization, the -1.92 foil (numerics.md 4.2) ----

def ptq_dynamic(encoder):
    """torch dynamic int8 on the body Linears -- per-tensor activations, the dead
    -1.92 arm kept as the foil QAT is measured against. A QAT-trained model deploys
    through this with a PER-CHANNEL weight qconfig to match what QAT trained."""
    import torch.ao.quantization as tq
    n_lin = sum(1 for m in encoder.modules() if isinstance(m, nn.Linear))
    q = tq.quantize_dynamic(encoder, {nn.Linear}, dtype=torch.qint8)
    n_q = sum(1 for m in q.modules() if type(m).__name__ == "LinearPackedParams")
    assert n_q > 0, "quantize_dynamic converted nothing"
    print(f"PTQ int8: {n_q}/{n_lin} Linear layers -> qint8 "
          f"(attention matmuls stay fp32)", flush=True)
    return q


# ---- Storage: per-row int8 of the token embedding table (packaging.md 4.1) ----

def emb_int8_fakequant(encoder):
    """Per-row symmetric int8 fake-quant of the token embedding table, in place --
    one scale per vocabulary row. Reproduces int8 storage numerics exactly (an int8
    embedding lookup returns row_int8 * scale) and shrinks the ~80%-of-params table
    ~4x. Returns the worst per-row weight perturbation so the run records it."""
    w = encoder.embeddings.tok_embeddings.weight.data
    s = (w.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    wq = torch.round(w / s).clamp(-127, 127) * s
    dw = float((wq - w).abs().max())
    w.copy_(wq)
    print(f"EMB-INT8: {w.shape[0]:,} x {w.shape[1]} table fake-quantized per-row "
          f"(max |dw| {dw:.2e})", flush=True)
    return dw
