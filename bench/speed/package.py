"""The installed htmlsift package as an extractor: Extractor(bundle_dir).extract(output="html"),
the code a user runs, timed and scored like the third-party baselines. Torch-free, so the
timing process and the RAM probe (memory.py) carry no torch.
"""
import time

from htmlsift import Extractor
from htmlsift.core.prep_page import prep_page

from bench.speed.competitors import Output


class PackageExtractor:
    """A mini bundle directory behind the package's own pipeline. The model loads on the
    first extract(), so a warmup page absorbs it."""

    name = "htmlsift"

    def __init__(self, bundle, threads=None):
        self.ex = Extractor(bundle_dir=bundle, threads=threads)

    def extract(self, html):
        t0 = time.perf_counter()
        out = self.ex.extract(html, output="html")
        return Output((time.perf_counter() - t0) * 1e3, out, "html")

    def tokens(self, html):
        """The page's model tokens, for the size buckets; off the clock."""
        blocks, _ = self.ex.extract(html, with_blocks=True)
        m = self.ex.model
        return len(prep_page(m.tok, blocks, m.window, cap=m.cap)["ids"]) if blocks else 0
