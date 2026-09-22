"""Load a saved keeper into a runnable block scorer -- the one definition of "what this
model is", shared by every consumer that runs a checkpoint (the bench harnesses,
tools/release, and the contract export/infer.py mirrors torch-free).

resolve_ckpt is the only IO (a local path, or a pull from $HF_MODELS_REPO); build_infer is
pure -- it takes the already-loaded dict and rebuilds the modules -- so the encoder/table
dispatch and the QAT int8 rebuild live here once, not once per caller.
"""
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from core.features import feature_dim
from core.model import (build_encoder, build_head, build_table_embedder,
                        infer_page, infer_table_page)
from core.quant import patch_qat_w8a8


def resolve_ckpt(ref, device):
    """Load a ckpt local-or-hub: an existing local path is used as-is; otherwise ref is a
    repo-relative path pulled from $HF_MODELS_REPO (the keeper model repo)."""
    path = Path(ref)
    if not path.exists():
        load_dotenv()
        repo = os.environ.get("HF_MODELS_REPO")
        if not repo:
            raise SystemExit(f"{ref} not found locally; set HF_MODELS_REPO to pull it")
        path = hf_hub_download(repo, ref, repo_type="model")
    return torch.load(path, map_location=device, weights_only=False)   # our own ckpt


def build_infer(ck, device, band=True):
    """Rebuild a keeper into (tok, infer_fn, kind, cap). infer_fn maps a prepped page to
    per-block probs; the caller preps pages with tok. Dispatch is on the ckpt's kind: a
    table ckpt carries kind=="table", an encoder ckpt carries no kind key.

    band switches the encoder to chunked band attention (core.band, bit-identical,
    speed-only) and is on by default -- it is what ships. Encoder-only, ignored on the
    table path (no attention). The A/B harness passes band= explicitly to turn it off."""
    tok = AutoTokenizer.from_pretrained(ck["model"])
    if ck.get("kind") == "table":
        emb = build_table_embedder(ck["model"], int8=ck["int8"], device=device)
        head = build_head(ck.get("head", "bigru"),
                          d_in=emb.hidden_size + feature_dim(ck.get("feats")),
                          hidden=ck.get("hidden", 256)).to(device)
        head.load_state_dict(ck["head_state"]); head.eval()
        return tok, lambda page: infer_table_page(emb, head, page, device), "table", 0
    enc = build_encoder(ck["model"], ck["layers"], device=device)
    if ck.get("qat"):
        # QAT keeper state was captured after patch_qat_w8a8; patch before load so the
        # fake-quant modules exist and it runs W8A8, not fp32.
        patch_qat_w8a8(enc)
    head = build_head(ck.get("head", "bigru"),
                      d_in=enc.config.hidden_size + feature_dim(ck.get("feats")),
                      hidden=ck.get("hidden", 256)).to(device)
    enc.load_state_dict(ck["encoder"]); enc.eval()
    head.load_state_dict(ck["head_state"]); head.eval()
    if band:
        import core.band
        core.band.enable(enc)                    # after any qat patch: compose, don't precede
    autocast = device.type == "cuda"
    return tok, lambda page: infer_page(enc, head, page, device, autocast), "encoder", ck.get("cap", 0)
