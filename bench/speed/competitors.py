"""Third-party main-content extractors for the speed+F1 comparison (bench/speed/cpu_compare.py).

Torch-free, like vendors/wmb/adapter/eval.py: the baselines our model is timed and scored
against, and they never touch the model. Each wrapper turns a page's HTML into main-content
HTML (trafilatura, readability) or plain text (resiliparse) -- the extractor's STANDARD
output, nothing more. No html2text and no scoring here: a board's score_text() converts and
judges (WMB html2texts html-kind output), so every extractor -- ours included -- is scored
the same way. `Output` is the shape our model extractor emits too.

The libraries are the [bench] extra, imported lazily inside each wrapper so this module
imports without them. An empty return is kept -- a completed run, timed, scored as an empty
prediction (F1 0 vs non-empty gold); only a raised exception is dropped (no valid output or
time). The report scores every method over one population and asterisks the incomplete ones.
"""
import time
from dataclasses import dataclass

EXTRACTORS: dict[str, "Spec"] = {}


@dataclass
class Output:
    extract_ms: float     # extraction time -- speed is this alone (html2text is off the clock)
    text: str             # the extractor's raw main-content output
    kind: str             # "html" | "text" -- how a board's score_text() reads it


@dataclass
class Spec:
    name: str
    fn: callable          # html -> main-content str, or None/"" for a miss
    kind: str             # "html" or "text"


def register(name, kind):
    def deco(fn):
        EXTRACTORS[name] = Spec(name, fn, kind)
        return fn
    return deco


@register("trafilatura", "html")
def _trafilatura(html):
    import trafilatura
    return trafilatura.extract(html, output_format="html")


@register("readability", "html")
def _readability(html):
    from readability import Document
    return Document(html).summary()


@register("resiliparse", "text")
def _resiliparse(html):
    from resiliparse.extract.html2text import extract_plain_text
    return extract_plain_text(html, main_content=True)


def _one(spec, html):
    t0 = time.perf_counter()
    raw = spec.fn(html)
    ms = (time.perf_counter() - t0) * 1e3
    return Output(ms, raw or "", spec.kind)   # an empty return is a completed run: timed, and scored as an empty prediction


def run(name, pages, warmup=5):
    """Time `name` over `pages` (tid -> {html, ...}); return (tid -> Output, notes). Warmup pages
    run untimed first to absorb the library's first-call cost, then every page is timed. An empty
    return is KEPT -- it ran to completion, so its time is valid and it scores as an empty
    prediction (F1 0 vs non-empty gold). Only a raised exception is dropped: no valid output, no
    valid time. notes carries the empty/crash counts for the report's asterisk."""
    spec = EXTRACTORS[name]
    items = [(tid, p["html"]) for tid, p in pages.items() if p.get("html")]
    for _, html in items[:warmup]:
        try:
            _one(spec, html)
        except Exception:
            pass
    out, empty, raised = {}, 0, 0
    for tid, html in items:
        try:
            o = _one(spec, html)
        except Exception:
            raised += 1
            continue
        out[tid] = o
        empty += not o.text
    print(f"{name}: {len(out)}/{len(items)} timed ({empty} empty, {raised} raised)", flush=True)
    return out, {"empty": empty, "raised": raised}
