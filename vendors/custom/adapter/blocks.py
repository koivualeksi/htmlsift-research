"""
Custom annotated pages -> blocks.jsonl + feats.jsonl in WMB's shapes, training pool only.

Each record carries the blocks htmlsift rendered when it was labelled, and its labels
are aligned to those blocks. The page is re-rendered here on the WMB block path (parse,
sanitize_tree, render_tree, rstripped non-empty lines); a page whose render differs from
the stored blocks is dropped and counted, never realigned, because realigning would
attach labels to text the annotator did not see. Features are gathered on the same
render, full A/B/C layout like WMB's feats.jsonl.

Raises unless libxml2 matches each record's own render version: libxml2 2.14 renders
different lines from the Windows wheels, so run this in WSL/Linux.

Also asserts no custom page shares a normalized URL with any WMB page, and reports how
many custom domains appear in each WMB fold (WMB's own domain_of).

    python vendors/custom/adapter/blocks.py
"""
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from htmlsift.core import render as cr
from htmlsift.core.features import collect_features
from lxml import etree

from vendors.wmb.adapter.sanitize import sanitize_tree
from vendors.wmb.adapter.split import domain_of

CUSTOM = Path(__file__).resolve().parents[1]
DATA = CUSTOM / "data"
RAW = DATA / "raw"
BLOCKS = DATA / "blocks.jsonl"
FEATS = DATA / "feats.jsonl"
WMB_DATA = CUSTOM.parent / "wmb" / "data"
LIBXML2 = ".".join(map(str, etree.LIBXML_VERSION))


def render_page(html):
    """html -> (blocks, feats [n_blocks, K]) on the WMB block path, features over A/B/C."""
    root = cr.parse(html)
    sanitize_tree(root)
    lines, line_src = cr.render_tree(root)
    blocks = [ln.rstrip() for ln in lines if ln.strip()]
    fa = collect_features(lines, line_src)
    assert fa.shape[0] == len(blocks), (fa.shape[0], len(blocks))
    return blocks, fa


def check_record(rec, html):
    """(blocks, labels, feats) when the re-render equals the stored blocks, else None.
    A malformed record raises: those are dataset bugs, not render drift."""
    if hashlib.sha256(html.encode("utf-8")).hexdigest() != rec["html_sha256"]:
        raise ValueError(f"{rec['id']}: html does not match html_sha256")
    if rec["render"]["libxml2"] != LIBXML2:
        raise ValueError(f"{rec['id']}: labelled under libxml2 {rec['render']['libxml2']}, "
                         f"running {LIBXML2}")
    if len(rec["labels"]) != len(rec["blocks"]) or set(rec["labels"]) - {0, 1}:
        raise ValueError(f"{rec['id']}: labels are not a 0/1 list parallel to blocks")
    blocks, fa = render_page(html)
    if blocks != rec["blocks"]:
        return None
    return blocks, rec["labels"], fa


def _norm_url(url):
    u = (url or "").strip().lower().split("#")[0].rstrip("/")
    for p in ("https://", "http://"):
        u = u.removeprefix(p)
    return u.removeprefix("www.")


def _wmb_overlap(records):
    """Raise on a shared URL with any WMB page; return {fold: n custom domains in it}."""
    with open(WMB_DATA / "splits.json", encoding="utf-8") as f:
        fold = {tid: v["fold"] for tid, v in json.load(f).items()}
    urls, domains = set(), {}
    with open(WMB_DATA / "wmb.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            urls.add(_norm_url(r.get("url", "")))
            domains.setdefault(domain_of(r.get("url", "")), set()).add(fold[r["track_id"]])
    shared = [r["id"] for r in records if _norm_url(r["url"]) in urls]
    if shared:
        raise SystemExit(f"{len(shared)} custom pages share a URL with WMB: {shared}")
    by_fold = Counter()
    for d in {domain_of(r["url"]) for r in records}:
        for fo in domains.get(d, ()):
            by_fold[fo] += 1
    return by_fold


def main():
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((RAW / "pages").glob("*.json"))]
    by_fold = _wmb_overlap(recs)
    kept, dropped = 0, []
    with open(BLOCKS, "w", encoding="utf-8") as fb, open(FEATS, "w", encoding="utf-8") as ff:
        for rec in recs:
            html = gzip.decompress((RAW / "html" / f"{rec['id']}.html.gz").read_bytes()).decode("utf-8")
            out = check_record(rec, html)
            if out is None:
                dropped.append(rec["id"])
                continue
            blocks, labels, fa = out
            fb.write(json.dumps({"track_id": rec["id"], "blocks": blocks, "labels": labels},
                                ensure_ascii=False) + "\n")
            ff.write(json.dumps({"track_id": rec["id"], "feats": fa.round(4).tolist()}) + "\n")
            kept += 1

    print(f"{BLOCKS.name}: {kept}/{len(recs)} pages kept, {len(dropped)} dropped on render "
          f"mismatch (libxml2 {LIBXML2})")
    if dropped:
        print(f"  dropped: {dropped}")
    print(f"  no URL shared with WMB; custom domains also in WMB folds: {dict(by_fold)}")


if __name__ == "__main__":
    main()
