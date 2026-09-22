"""
Download frozen WebContentExtractionBenchmark snapshot - be72432c revision.

WCXB ships as a directory tree (dev/ and test/, each with ground-truth/*.json
and html/*.html) rather than WMB's two JSONL files, so it is pulled with one
snapshot_download at a pinned revision. Integrity is anchored the way WMB anchors
its per-file sha256 -- hardcoded -- but at 4,016 files that is a single constant:
the sha256 of the sorted 'sha256  relpath' manifest of every downloaded file.
Leave MANIFEST_SHA256 = None to print (record) it on the first run, then paste
it in.
"""

from pathlib import Path

from huggingface_hub import list_repo_files, snapshot_download

from vendors.shared.acquire import manifest_sha256

REPO_ID = "murrough-foley/web-content-extraction-benchmark"
REPO_TYPE = "dataset"
REVISION = "be72432cfa012ac918af47010bf106a2801afeef"
DATA = Path(__file__).resolve().parents[1] / "data"

MANIFEST_SHA256 = "e803b06b223bf932f7c682e13903f947d18835ad9d84bbd86e13a1061de18ad2"

N_DEV = 1497
N_TEST = 511


def _check_pairs(split: str, expected: int) -> None:
    gt = {p.stem for p in (DATA / split / "ground-truth").glob("*.json")}
    html = {p.stem for p in (DATA / split / "html").glob("*.html")}
    assert gt == html, f"{split}: {len(gt ^ html)} unpaired files"
    assert len(gt) == expected, f"{split}: {len(gt)} pages != {expected}"


def acquire() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO_ID, repo_type=REPO_TYPE, revision=REVISION, local_dir=str(DATA),
    )
    _check_pairs("dev", N_DEV)
    _check_pairs("test", N_TEST)
    # Manifest over the repo's own file list, not DATA.rglob: the derived outputs
    # (wcxb.jsonl, splits.json, blocks/labels, eval/) land in the same dir and would
    # otherwise pollute this download-integrity check on a re-run.
    repo_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE, revision=REVISION)
    got = manifest_sha256([DATA / rel for rel in repo_files], DATA)
    if MANIFEST_SHA256 is None:
        print(f"record  manifest sha256={got}")
    elif got != MANIFEST_SHA256:
        raise SystemExit(f"manifest sha256 mismatch: {got} != {MANIFEST_SHA256}")
    else:
        print(f"ok      {N_DEV + N_TEST} pages verified in {DATA}")


if __name__ == "__main__":
    acquire()
