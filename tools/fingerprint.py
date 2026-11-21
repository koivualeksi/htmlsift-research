"""Model-input fingerprint of the installed htmlsift package.

Runs every page of WMB, DAnIEL and WCXB through the boards' own block builders on
htmlsift's core, then prep_page, and hashes what training consumes: blocks, labels
(WMB), raw features, token ids and block members. Same hash as the one recorded for
the current bundle: the package change leaves model input alone. Different: it needs
a bundle trained on it, released together.

Install the package under test from GitHub (a branch, tag or commit), then run on
Linux (lxml 6.1.1, libxml2 2.14); another libxml2 renders some pages differently.

    pip install --force-reinstall --no-deps "htmlsift @ git+https://github.com/koivualeksi/htmlsift@main"
    python tools/fingerprint.py            # print, compare with fingerprints.json
    python tools/fingerprint.py --record   # also add this result to fingerprints.json
"""
import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path

import htmlsift
import numpy as np
from htmlsift._artifacts import Manifest, resolve_bundle
from htmlsift.core.prep_page import prep_page
from htmlsift.core.tokenizer import FastTokenizer
from lxml import etree

from vendors.daniel.adapter import blocks as daniel
from vendors.wcxb.adapter import blocks as wcxb
from vendors.wmb.adapter import blocks as wmb

REGISTRY = Path(__file__).with_name("fingerprints.json")
BOARDS = {"wmb": (wmb, wmb.COMBINED, 7825),
          "daniel": (daniel, daniel.COMBINED, 1689),
          "wcxb": (wcxb, wcxb.COMBINED, 2008)}
FEATS = "ABC"


def _sha(b):
    return hashlib.sha256(b).digest()


def page_digest(blocks, labels, feats, ids, members):
    """Digest of one page's model inputs."""
    parts = [
        "\n".join(blocks).encode("utf-8"),
        b"none" if labels is None else np.asarray(labels, dtype="<i8").tobytes(),
        np.asarray(feats, dtype="<f4").tobytes(),
        np.asarray(ids, dtype="<i8").tobytes(),
        np.asarray([len(m) for m in members], dtype="<i8").tobytes(),
        np.asarray([t for m in members for t in m], dtype="<i8").tobytes(),
    ]
    return _sha(b"".join(_sha(p) for p in parts))


def board_inputs(name, html):
    """(blocks, labels, feats) for one page, as the board's block builder makes them."""
    mod = BOARDS[name][0]
    if name == "wmb":
        blocks, labels, _box, feats = mod.label_page(html, FEATS)
        return blocks, labels, feats
    blocks, feats = mod.render_page(html, FEATS)
    return blocks, None, feats


def board_hash(name, tok, window, cap, limit=None):
    """Hex hash of every page of one board, in file order."""
    path, n_expected = BOARDS[name][1:]
    h, n = hashlib.sha256(), 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and n == limit:
                break
            blocks, labels, feats = board_inputs(name, json.loads(line)["html"])
            page = prep_page(tok, blocks, window, cap=cap)
            h.update(page_digest(blocks, labels, feats, page["ids"], page["members"]))
            n += 1
    if limit is None:
        assert n == n_expected, f"{name}: {n} pages, expected {n_expected}"
    return h.hexdigest()[:16]


def package_source():
    """Where pip installed htmlsift from: 'git+<url>@<commit>', 'local:<path>' or 'pypi'."""
    raw = metadata.distribution("htmlsift").read_text("direct_url.json")
    if not raw:
        return "pypi"
    d = json.loads(raw)
    if "vcs_info" in d:
        return f"git+{d['url']}@{d['vcs_info']['commit_id']}"
    return f"local:{d['url']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true", help="add the result to fingerprints.json")
    args = ap.parse_args()

    paths = resolve_bundle("mini")
    manifest = Manifest.from_file(paths["manifest"])
    tok = FastTokenizer.from_file(paths["tokenizer"])
    boards = {}
    for name in BOARDS:
        boards[name] = board_hash(name, tok, manifest.window, manifest.cap)
        print(f"{name:7} {boards[name]}", flush=True)
    total = hashlib.sha256("".join(boards.values()).encode()).hexdigest()[:16]
    print(f"{'total':7} {total}")

    entries = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else []
    same = [e for e in entries if e["total"] == total]
    if same:
        print(f"matches: {', '.join(e['htmlsift'] + ' ' + e['source'] for e in same)}")
    elif entries:
        print("no recorded entry matches: model input changed")
    if args.record:
        entries.append({"htmlsift": htmlsift.__version__, "source": package_source(),
                        "lxml": ".".join(map(str, etree.LXML_VERSION[:3])),
                        "libxml2": ".".join(map(str, etree.LIBXML_VERSION)),
                        "boards": boards, "total": total})
        REGISTRY.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        print(f"recorded in {REGISTRY.name}")


if __name__ == "__main__":
    main()
