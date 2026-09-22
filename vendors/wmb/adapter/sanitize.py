"""
Sanitizer: strip annotation-tool and Immersive-Translate residue from raw html.

The WMB `html` field is the page as it sat in the annotators' browsers, so it
carries the marks of how it was labeled (cc-select, data-anno-uid, mark-selected)
and of the Immersive-Translate extension they ran (injected <font> wrappers,
immersive-translate-* elements, a cc-extraStyle block). Both must go before the
html reaches the model: the annotation marks ARE the label, and the injected
elements are rendering-visible boilerplate present in no real page.

Applied identically to the model-input side and the label-generation side, and
NEVER to the eval references (convert_main_content stays render(main_html)
verbatim). The block builder harvests cc-select from the raw DOM first, then calls
this; inverting that order strips the labels before they are read.

Two layers:
  1. decompose injected elements (the only rendering-visible residue)
  2. attribute hygiene: remove label leakage and tool fingerprints without
     changing what renders
"""

import re
from pathlib import Path

from vendors.shared.upstream import load

WMB = Path(__file__).resolve().parents[1]

_upstream = load(WMB / "upstream" / "main_html.py", "wmb_main_html")
html_to_element = _upstream.html_to_element
element_to_html = _upstream.element_to_html


# Annotator residue: the cc-select markup tool.
ANNO_ATTRS = ("data-anno-uid", "cc-select")
ANNO_CLASS_TOKENS = ("mark-selected", "cc-unloaded")
ANNO_STYLE_ID = "cc-extraStyle"       # injected <style id=...>
ANNO_STYLE_NAME = "cc"                 # injected <style name=...>
USER_SELECT_RE = re.compile(r"\s*user-select\s*:\s*none\s*;?", re.I)  # injected declaration

# Immersive-Translate residue: the browser extension the annotators ran.
IMT_ATTR_PREFIX = "data-immersive-translate-"
IMT_MARK_ATTR = "data-immersive-translate-translation-element-mark"
IMT_CLASS_PREFIX = "immersive-translate-"
IMT_ID_PREFIX = "immersive-translate"
# 'notranslate' is a legit Google convention on real sites and is kept.


def _is_injected(el) -> bool:
    if el.get(IMT_MARK_ATTR) is not None:
        return True
    # class-token prefix, not a substring 'imt': real classes like 'disclaimtext',
    # 'level1_BMIMT', 'kvIMTB' would otherwise be false positives.
    if any(tok.startswith(IMT_CLASS_PREFIX) for tok in (el.get("class") or "").split()):
        return True
    if (el.get("id") or "").startswith(IMT_ID_PREFIX):
        return True
    if el.tag == "style" and (el.get("id") == ANNO_STYLE_ID or el.get("name") == ANNO_STYLE_NAME):
        return True
    return False


def sanitize_tree(root, decompose_injected: bool = True):
    """Sanitize an lxml tree in place. decompose_injected=False runs only the
    rendering-neutral hygiene layer, so a caller can prove that layer changes no
    render."""
    if decompose_injected:
        for el in [el for el in root.iter() if isinstance(el.tag, str) and _is_injected(el)]:
            if el.getparent() is not None:
                el.drop_tree()

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        for name in list(el.attrib):
            if name in ANNO_ATTRS or name.startswith(IMT_ATTR_PREFIX):
                del el.attrib[name]
        cls = el.get("class")
        if cls is not None:
            tokens = cls.split()
            kept = [t for t in tokens
                    if t not in ANNO_CLASS_TOKENS and not t.startswith(IMT_CLASS_PREFIX)]
            if len(kept) != len(tokens):
                if kept:
                    el.set("class", " ".join(kept))
                else:
                    del el.attrib["class"]
        style = el.get("style")
        if style is not None and "user-select" in style:
            new_style = USER_SELECT_RE.sub("", style).strip()
            if new_style:
                el.set("style", new_style)
            else:
                del el.attrib["style"]
    return root


def sanitize_html(html_str: str, decompose_injected: bool = True) -> str:
    root = html_to_element(html_str)
    sanitize_tree(root, decompose_injected)
    return element_to_html(root)


def residue_report(root) -> list[str]:
    """List residue findings in a sanitized tree (empty = clean). Structural only:
    text content may legitimately mention these strings."""
    findings = []
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        if _is_injected(el):
            findings.append(f"injected element survived: <{el.tag} class={el.get('class')!r}>")
        for name in el.attrib:
            if name in ANNO_ATTRS or name.startswith(IMT_ATTR_PREFIX):
                findings.append(f"attr survived: {name} on <{el.tag}>")
        for tok in (el.get("class") or "").split():
            if tok in ANNO_CLASS_TOKENS or tok.startswith(IMT_CLASS_PREFIX):
                findings.append(f"class token survived: {tok} on <{el.tag}>")
        if USER_SELECT_RE.search(el.get("style") or ""):
            findings.append(f"user-select survived in style attr on <{el.tag}>")
    return findings


if __name__ == "__main__":
    import json
    import time

    corpus = WMB / "data" / "wmb.jsonl"
    n = residue_free = neutral = layer1_changed = 0
    t = time.time()
    with open(corpus, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            html = r.get("html")
            if not html:
                continue
            n += 1

            root = html_to_element(html)
            sanitize_tree(root, decompose_injected=True)
            findings = residue_report(root)
            assert not findings, f"{r['track_id']}: residue survived: {findings[:3]}"
            residue_free += 1

            # A fresh converter per page: html2text.HTML2Text carries state across calls.
            base = _upstream.HTML2TextWrapper()(element_to_html(html_to_element(html)), "")
            hygiene = _upstream.HTML2TextWrapper()(sanitize_html(html, decompose_injected=False), "")
            assert base == hygiene, f"{r['track_id']}: layer-2 hygiene changed the render"
            neutral += 1

            if _upstream.HTML2TextWrapper()(sanitize_html(html, decompose_injected=True), "") != base:
                layer1_changed += 1

    print(f"{n} pages: residue-free {residue_free}/{n}, layer-2 rendering-neutral {neutral}/{n}, "
          f"layer-1 changed the render on {layer1_changed}")
    print(f"gate passed in {time.time() - t:.0f}s")
