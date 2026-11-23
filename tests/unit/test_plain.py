"""Unit: the table recipe's plain arm. Plain rows must stay 1:1 with the markdown blocks,
labels and features, or the head trains on shifted rows; and only the model input
changes, so the default path must still tokenize the markdown.
"""
import json

import pytest

from vendors.wmb.adapter import benchmark as wmb_bench
from vendors.wmb.adapter import blocks as wmb_blocks
from vendors.wmb.adapter import eval as wmb_eval

PAGE = ('<html><body><h1 cc-select="true">Header</h1>'
        '<p cc-select="true">Some <b>bold</b> text.</p>'
        "<table><tr><th>a</th><th>b</th></tr><tr><td>1</td><td>2</td></tr></table>"
        "</body></html>")


class CharTok:
    """One token per character, so the page text the tokenizer saw is recoverable."""

    def __call__(self, text, **_):
        return {"input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
                "special_tokens_mask": [0] * len(text)}


def test_plain_rows_align_with_blocks():
    blocks, labels, _box, plain = wmb_blocks.label_page(PAGE)
    assert blocks == ["# Header", "Some **bold** text.", "| a | b |", "| --- | --- |",
                      "| 1 | 2 |"]
    assert plain == ["Header", "Some bold text.", "a | b", "", "1 | 2"]
    assert labels == [1, 1, 0, 0, 0]


@pytest.mark.parametrize("plain, first", [(False, "# Header"), (True, "Header")])
def test_plain_reaches_the_tokenizer(tmp_path, monkeypatch, plain, first):
    blocks, labels, box, rows = wmb_blocks.label_page(PAGE)
    path = tmp_path / "blocks.jsonl"
    path.write_text(json.dumps({"track_id": "t", "blocks": blocks, "labels": labels,
                                "box": box, "plain": rows}) + "\n", encoding="utf-8")
    monkeypatch.setattr(wmb_eval, "BLOCKS", path)
    monkeypatch.setattr(wmb_eval, "fold_ids", lambda fold: {"t"})
    b = wmb_bench.WMBBenchmark(plain)
    page, = b.build_pages(CharTok(), "train", 512)
    text = ["".join(chr(page["ids"][t]) for t in m) for m in page["members"]]
    assert text[0] == first
    assert text == (rows if plain else blocks)
    assert list(page["y"]) == labels
    assert b._fold_blocks("val")["t"] == (rows if plain else blocks)
