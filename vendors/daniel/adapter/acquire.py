"""
Download the frozen DAnIEL corpus - waddle be875ce1, corpora/Corpus_daniel_v2.1.

DAnIEL lives in a GitHub repo, not on HuggingFace, so it is pulled as the repo
tarball at a pinned commit and only the corpus subtree is extracted into data/.
GitHub's generated tarball wrapper is not byte-stable, so integrity is anchored
on file *contents* the way WCXB anchors its tree: the sha256 of the sorted
'sha256  relpath' manifest of every extracted file. Leave MANIFEST_SHA256 = None
to print (record) it on the first run, then paste it in.
"""

import io
import json
import tarfile
import urllib.request
from pathlib import Path

from vendors.shared.acquire import manifest_sha256

REPO = "rundimeco/waddle"
COMMIT = "be875ce12ad9e23ef68ef7f3b0605db6d06bcdb1"
SUBTREE = "corpora/Corpus_daniel_v2.1"
TARBALL = f"https://github.com/{REPO}/archive/{COMMIT}.tar.gz"
DATA = Path(__file__).resolve().parents[1] / "data"

MANIFEST_SHA256 = "7f5723fb20a7d533bf6d2334a09e3917746ac5d0a58bbe50dc7f01a1a9203502"

N_HTML = 1689
N_REFERENCE = 2120
N_DOC_LG = 2089


def _extract() -> list[Path]:
    prefix = f"waddle-{COMMIT}/{SUBTREE}/"
    req = urllib.request.Request(TARBALL, headers={"User-Agent": "htmlsift-research"})
    with urllib.request.urlopen(req) as r:
        blob = r.read()
    written = []
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile() or not m.name.startswith(prefix):
                continue
            dest = DATA / m.name[len(prefix):]
            assert dest.resolve().is_relative_to(DATA.resolve()), f"unsafe path {m.name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as src:
                dest.write_bytes(src.read())
            written.append(dest)
    return written


def _check_structure() -> None:
    html = {p.name for p in (DATA / "html").iterdir() if p.is_file()}
    ref = {p.name for p in (DATA / "reference").iterdir() if p.is_file()}
    doc_lg = json.loads((DATA / "doc_lg.json").read_text(encoding="utf-8"))
    assert len(html) == N_HTML, f"html: {len(html)} != {N_HTML}"
    assert len(ref) == N_REFERENCE, f"reference: {len(ref)} != {N_REFERENCE}"
    assert len(doc_lg) == N_DOC_LG, f"doc_lg: {len(doc_lg)} != {N_DOC_LG}"
    assert html <= ref, f"{len(html - ref)} html files without a reference"


def acquire() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    # Manifest over exactly what _extract wrote, not DATA.rglob: the derived outputs
    # (daniel.jsonl, blocks/labels, eval/) land in the same dir and would otherwise
    # pollute this download-integrity check on a re-run.
    files = _extract()
    _check_structure()
    got = manifest_sha256(files, DATA)
    if MANIFEST_SHA256 is None:
        print(f"record  manifest sha256={got}")
    elif got != MANIFEST_SHA256:
        raise SystemExit(f"manifest sha256 mismatch: {got} != {MANIFEST_SHA256}")
    else:
        print(f"ok      {N_HTML} pages verified in {DATA}")


if __name__ == "__main__":
    acquire()
