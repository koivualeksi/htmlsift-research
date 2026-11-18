"""
Download the pinned snapshot of the private htmlsift annotations dataset.

Needs an HF login with read access to the dataset. The tree lands in data/raw/
and is checked as a whole against MANIFEST_SHA256 (sha256 over the sorted
'sha256  relpath' lines of every pages/ and html/ file), so a moved pin or a
partial download raises.

    python vendors/custom/adapter/acquire.py
"""
from pathlib import Path

from huggingface_hub import snapshot_download

from vendors.shared.acquire import manifest_sha256

REPO_ID = "koivualeksi/htmlsift-annotations"
REPO_TYPE = "dataset"
REVISION = "88ba5c55e402e5bc6c9dde66ac96945e7ea78e94"
DATA = Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "raw"

N_PAGES = 112
MANIFEST_SHA256 = "849833c06e816922ee6f7bb1878927e9b6cb7103500280858d7eb6135ccedf95"


def acquire() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    snapshot_download(REPO_ID, repo_type=REPO_TYPE, revision=REVISION,
                      local_dir=str(RAW), allow_patterns=["pages/*.json", "html/*.html.gz"])
    pages = sorted((RAW / "pages").glob("*.json"))
    html = sorted((RAW / "html").glob("*.html.gz"))
    if len(pages) != N_PAGES or len(html) != N_PAGES:
        raise SystemExit(f"expected {N_PAGES} pages + html, got {len(pages)} + {len(html)}")
    got = manifest_sha256(pages + html, RAW)
    if MANIFEST_SHA256 is None:
        print(f"record  manifest sha256={got}")
    elif got != MANIFEST_SHA256:
        raise SystemExit(f"manifest sha256 mismatch: {got} != {MANIFEST_SHA256}")
    else:
        print(f"ok      {N_PAGES} pages at {REVISION[:12]}")


if __name__ == "__main__":
    acquire()
