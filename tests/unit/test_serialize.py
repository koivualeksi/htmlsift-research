"""Unit: serialize's scored html is the htmlsift package's html output. For the same block
labels, apply(output="html") equals Extractor.extract(output="html"), and a second apply on
the same Prepared is unaffected by the first (prune() mutates a copy).
"""
from types import SimpleNamespace

import numpy as np
import pytest
from htmlsift import Extractor

from vendors.wmb.adapter import serialize

PAGES = [
    "<html><body><h1>Title</h1><p>First <b>bold</b> line.<br>Second line.</p>"
    "<ul><li>one</li><li>two <a href='/x'>link</a></li></ul></body></html>",
    "<html><body><p>Intro.</p><table><tr><th>a</th><th>b</th></tr>"
    "<tr><td>1</td><td>2</td></tr></table><p>Outro <i>here</i>.</p></body></html>",
    "<html><body><p>Code:</p><pre><code>x = 1\ny = 2\nz = 3</code></pre>"
    "<pre>plain one\nplain two</pre><div>tail <span>span</span> text</div></body></html>",
]


class CharTok:
    """One token per character: enough for prep_page, which the stub model ignores."""

    def __call__(self, text, **_):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
                "special_tokens_mask": [0] * len(text)}


def package_html(html, labels):
    ex = Extractor()
    ex._model = SimpleNamespace(tok=CharTok(), window=8192, cap=0,
                                features=lambda lines, src: None,
                                infer_fn=lambda page: np.asarray(labels, dtype=np.float32))
    return ex.extract(html, output="html")


def label_sets(n):
    yield [1] * n
    yield [i % 2 for i in range(n)]
    yield [(i + 1) % 2 for i in range(n)]
    yield [int(i == n - 1) for i in range(n)]


@pytest.mark.parametrize("html", PAGES)
def test_apply_matches_package_html(html):
    prep = serialize.prepare(html)
    n = len(prep.block_pieces)
    for labels in label_sets(n):
        assert serialize.apply(prep, labels, output="html") == \
            package_html(html, labels), labels


def test_apply_leaves_the_prepared_tree_alone():
    prep = serialize.prepare(PAGES[0])
    n = len(prep.block_pieces)
    first = serialize.apply(prep, [1] * n, output="html")
    serialize.apply(prep, [int(i == 0) for i in range(n)], output="html")
    assert serialize.apply(prep, [1] * n, output="html") == first


def test_empty_selection_is_empty():
    prep = serialize.prepare(PAGES[1])
    assert serialize.apply(prep, [0] * len(prep.block_pieces)) == ""
