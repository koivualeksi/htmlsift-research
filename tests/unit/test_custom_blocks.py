"""Unit: vendors/custom render-match check -- a record is kept only when the re-render
equals its stored blocks; any drift drops it (None), never realigns. Malformed records
(hash, libxml2, label shape) raise. Authored HTML only, no private data."""
import hashlib

import pytest

from vendors.custom.adapter.blocks import LIBXML2, check_record, render_page

HTML = ("<html><body><nav><a href='/'>Home</a> <a href='/a'>About</a></nav>"
        "<article><h1>Title</h1><p>First paragraph.</p><p>Second paragraph.</p></article>"
        "<footer>Copyright</footer></body></html>")


def record(html=HTML, **over):
    blocks, _ = render_page(html)
    rec = {"id": "t", "html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
           "render": {"libxml2": LIBXML2}, "blocks": blocks, "labels": [0] * len(blocks)}
    return {**rec, **over}


def test_match_kept():
    rec = record()
    blocks, labels, fa = check_record(rec, HTML)
    assert blocks == rec["blocks"] and labels == rec["labels"]
    assert fa.shape[0] == len(blocks) > 2


@pytest.mark.parametrize("edit", [
    lambda b: [b[0] + " x"] + b[1:],          # one line changed
    lambda b: b[:-1],                          # one line missing
    lambda b: b + ["extra"],                   # one line added
    lambda b: [b[0].rstrip() + " "] + b[1:],   # trailing whitespace differs
])
def test_drift_dropped(edit):
    rec = record()
    blocks = edit(rec["blocks"])
    assert check_record({**rec, "blocks": blocks, "labels": [0] * len(blocks)}, HTML) is None


def test_hash_mismatch_raises():
    with pytest.raises(ValueError, match="html_sha256"):
        check_record(record(), HTML.replace("Title", "Other"))


def test_libxml2_mismatch_raises():
    with pytest.raises(ValueError, match="libxml2"):
        check_record(record(render={"libxml2": "0.0.0"}), HTML)


def test_label_shape_raises():
    rec = record()
    with pytest.raises(ValueError, match="parallel"):
        check_record({**rec, "labels": rec["labels"][:-1]}, HTML)
    with pytest.raises(ValueError, match="parallel"):
        check_record({**rec, "labels": [2] + rec["labels"][1:]}, HTML)
