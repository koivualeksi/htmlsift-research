"""
Collapse the acquired DAnIEL tree into one eval corpus.

DAnIEL ships as html/ (raw inputs) and reference/ (<p>-wrapped main-content gold)
trees plus doc_lg.json (id -> language). This folds the 1,689 documents that have
both an html input and a reference gold into a single data/daniel.jsonl, one
record per page, dropping the 431 reference-only orphans. Eval-only: no split;
the leave-one-language-out folds, if ever built, cut on the language field.

The reference gold is <p>-segmented; we parse it, drop script/style, and join
paragraph text with newlines. 13 files carry the corpus's own nonstandard &X;
entities (e.g. &E;, &Partner;), left literal, and 1 file leaks a <script>; both
are noted in docs/LIMITS.md.

2012-era wild html is not guaranteed utf-8: decode strict, fall back to
charset_normalizer, count fallbacks.

    python vendors/daniel/adapter/collapse.py
"""

import collections
import json
import re
from pathlib import Path

from lxml import html as lxml_html

from vendors.shared.acquire import read_html

DATA = Path(__file__).resolve().parents[1] / "data"
DOC_LG = DATA / "doc_lg.json"
COMBINED = DATA / "daniel.jsonl"

LANG_COUNTS = {"Greek": 273, "Polish": 274, "Russian": 266, "Chinese": 401, "English": 475}

_WS = re.compile(r"\s+")


def clean_reference(text: str) -> str:
    if not text.strip():
        return ""
    frag = lxml_html.fragment_fromstring(text, create_parent="div")
    for el in frag.iter("script", "style"):
        el.drop_tree()
    paras = [t for p in frag.iter("p") if (t := _WS.sub(" ", p.text_content()).strip())]
    return "\n".join(paras)


def main():
    doc_lg = json.loads(DOC_LG.read_text(encoding="utf-8"))
    html_dir = DATA / "html"
    ref_dir = DATA / "reference"
    ids = sorted(p.name for p in html_dir.iterdir()
                 if p.is_file() and (ref_dir / p.name).is_file())

    langs = collections.Counter()
    n_fallback = n_empty = 0
    with open(COMBINED, "w", encoding="utf-8") as out:
        for doc_id in ids:
            language = doc_lg[doc_id]
            html, fb = read_html(html_dir / doc_id)
            n_fallback += fb
            reference = clean_reference((ref_dir / doc_id).read_text(encoding="utf-8"))
            n_empty += not reference
            langs[language] += 1
            out.write(json.dumps({
                "track_id": doc_id,
                "language": language,
                "html": html,
                "reference": reference,
            }, ensure_ascii=False) + "\n")

    assert len(ids) == 1689, f"{len(ids)} paired docs != 1689"
    assert dict(langs) == LANG_COUNTS, f"language counts {dict(langs)} != {LANG_COUNTS}"

    print(f"{COMBINED.name}: {len(ids)} records "
          f"({', '.join(f'{k} {v}' for k, v in LANG_COUNTS.items())})")
    print(f"charset fallback on {n_fallback} html files")
    print(f"empty reference after cleaning: {n_empty}")


if __name__ == "__main__":
    main()
