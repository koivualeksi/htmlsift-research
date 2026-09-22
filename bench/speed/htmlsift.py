"""Our model (htmlsift) as an extractor: html -> main-content HTML, the same shape as the
third-party baselines (bench/speed/competitors.Output). Torch + WMB serialize live here, not
in the torch-free competitors registry. Output is HTML only -- no html2text: a board's
score_text() converts and scores, so htmlsift is judged exactly like trafilatura. Speed is
render -> model -> serialize(output="html"), one render (frontend.render_blocks) feeding both
the model and serialize.
"""
import time

import torch

from bench.speed.competitors import Output
from bench.speed.frontend import build_feats_fn, render_blocks
from core.constants import THRESHOLD
from core.loader import build_infer, resolve_ckpt
from core.prep_page import prep_page
from vendors.shared.arms import WINDOW
from vendors.wmb.adapter.serialize import Prepared, apply


class HtmlsiftExtractor:
    """A keeper ckpt as an extractor. `run` returns {tid: Output} (kind always "html") plus
    {tid: n_tokens} for the size buckets -- the token count is a by-product of prep."""

    name = "htmlsift"

    def __init__(self, ref, device="cpu", threads=None):
        dev = torch.device(device)
        if threads and dev.type == "cpu":
            torch.set_num_threads(threads)
        ck = resolve_ckpt(ref, dev)
        self.tok, self.infer_fn, self.kind, self.cap = build_infer(ck, dev)
        self.feats_fn = build_feats_fn(ck)
        print(f"htmlsift: {ref} kind={self.kind} feats={ck.get('feats') or 'none'} cap={self.cap} "
              f"threads={torch.get_num_threads()}", flush=True)

    def _one(self, tid, html):
        t0 = time.perf_counter()
        root, blocks, block_src, feats = render_blocks(html, self.feats_fn)
        if not blocks:
            return Output((time.perf_counter() - t0) * 1e3, "", "html"), 0   # empty render: scorable empty prediction
        page = prep_page(self.tok, tid, blocks, None, WINDOW, cap=self.cap, feats=feats)
        labels = [int(v > THRESHOLD) for v in self.infer_fn(page)]
        html_out = apply(Prepared(root, block_src, ""), labels, output="html")   # "" when no block clears 0.5 -- kept, scored
        ms = (time.perf_counter() - t0) * 1e3
        return Output(ms, html_out, "html"), len(page["ids"])

    def run(self, pages, warmup=5):
        items = [(tid, p["html"]) for tid, p in pages.items() if p.get("html")]
        for tid, html in items[:warmup]:
            self._one(tid, html)
        out, tokens, empty = {}, {}, 0
        for tid, html in items:
            o, n = self._one(tid, html)
            out[tid], tokens[tid] = o, n
            empty += not o.text
        print(f"{self.name}: {len(out)}/{len(items)} pages ({empty} empty)", flush=True)
        return out, tokens, {"empty": empty, "raised": 0}


class HtmlsiftOnnxExtractor(HtmlsiftExtractor):
    """The mini ONNX graphs as an extractor: same render/prep/serialize as HtmlsiftExtractor,
    but the forward is onnxruntime (export.infer) with no torch on the timed path. The
    tokenizer and feature z-score stats still come from the keeper .pt, off the timed path."""

    name = "htmlsift"

    def __init__(self, ref, table_path, head_path, threads=None):
        from transformers import AutoTokenizer

        from export.infer import load_mini
        ck = resolve_ckpt(ref, torch.device("cpu"))
        assert ck.get("kind") == "table", "mini ONNX wraps a table keeper (pass --model 311m-table-...)"
        self.tok = AutoTokenizer.from_pretrained(ck["model"])
        self.infer_fn = load_mini(table_path, head_path, threads)
        self.feats_fn = build_feats_fn(ck)
        self.kind, self.cap = "table", 0
        print(f"htmlsift-onnx: {table_path.name} + {head_path.name} "
              f"feats={ck.get('feats') or 'none'} threads={threads}", flush=True)
