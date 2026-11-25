"""A torch keeper as an extractor: html -> main-content HTML, the same shape as the
third-party baselines (bench/speed/competitors.Output). Torch + WMB serialize live here, not
in the torch-free competitors registry. Output is HTML only -- no html2text: a board's
score_text() converts and scores, so the keeper is judged exactly like trafilatura. Speed is
render -> model -> serialize(output="html"), one render (frontend.render_blocks) feeding both
the model and serialize. The shipped mini is timed through the package instead (package.py).
"""
import time

import torch

from bench.speed.competitors import Output
from bench.speed.frontend import build_feats_fn, render_blocks
from htmlsift.extractor import THRESHOLD
from core.loader import build_infer, resolve_ckpt
from core.training import train_page
from vendors.shared.arms import WINDOW
from vendors.wmb.adapter.serialize import Prepared, apply


class HtmlsiftExtractor:
    """A keeper ckpt as an extractor."""

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

    def _page(self, blocks, feats):
        return train_page(self.tok, "page", blocks, None, WINDOW, cap=self.cap, feats=feats)

    def extract(self, html):
        t0 = time.perf_counter()
        root, blocks, block_pieces, feats = render_blocks(html, self.feats_fn)
        if not blocks:
            return Output((time.perf_counter() - t0) * 1e3, "", "html")   # empty render: scorable empty prediction
        labels = [int(v > THRESHOLD) for v in self.infer_fn(self._page(blocks, feats))]
        html_out = apply(Prepared(root, block_pieces, ""), labels, output="html")   # "" when no block clears 0.5 -- kept, scored
        return Output((time.perf_counter() - t0) * 1e3, html_out, "html")

    def tokens(self, html):
        """The page's model tokens, for the size buckets; off the clock."""
        _root, blocks, _src, feats = render_blocks(html, self.feats_fn)
        return len(self._page(blocks, feats)["ids"]) if blocks else 0
