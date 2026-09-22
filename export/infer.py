"""Torch-free onnxruntime forward for the mini artifact (int8 table + BiGRU head).

Mirrors core.model.infer_table_page -- embed, mean-pool by block membership, concat
features, head, sigmoid -- through the two graphs export/build.py builds. This is the
SHIPPED forward: the bench extractor, the parity gate, and the production runtime all
call load_mini, and `pool` is the single numpy mirror of core.model.pool they share.

Torch-free by contract: this module imports numpy + onnxruntime only, never torch, so
the production repo can lift it as the runtime unchanged.
"""
import numpy as np
import onnxruntime as ort


def pool(emb, members):
    """Token embeddings emb [T, H] -> per-block mean vectors [n, H]. A token-less block
    pools to zeros -- the numpy echo of core.model.pool (sum of nothing / 1)."""
    pooled = np.zeros((len(members), emb.shape[1]), dtype=np.float32)
    for i, mem in enumerate(members):
        if mem:
            pooled[i] = emb[mem].mean(0)
    return pooled


def load_mini(table_path, head_path, threads=None):
    """The two mini graphs -> infer_fn(page) -> per-block probs (np.float32): the same
    (page -> probs) contract as build_infer's table branch, so it drops into the extractor."""
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads, opts.inter_op_num_threads = threads, 1
    table = ort.InferenceSession(str(table_path), opts, providers=["CPUExecutionProvider"])
    head = ort.InferenceSession(str(head_path), opts, providers=["CPUExecutionProvider"])

    def infer_fn(page):
        emb = table.run(["out"], {"ids": page["ids"].astype(np.int64)})[0]     # [T, H]
        x = pool(emb, page["members"])                                         # [n, H]
        if page.get("feats") is not None:
            x = np.concatenate([x, page["feats"].astype(np.float32)], 1)       # [n, H+K]
        logits = head.run(["logits"], {"x": x[None]})[0][0]                    # [n]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
    return infer_fn
