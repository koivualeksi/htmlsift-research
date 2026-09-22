"""
Download frozen versions of WebMainBench dataset - 4.8.2026 revision
"""

from pathlib import Path

from huggingface_hub import hf_hub_download

from vendors.shared.acquire import sha256

REPO_ID = "opendatalab/WebMainBench"
REPO_TYPE = "dataset"
REVISION = "5da0972e9b58d0c7891ae75053ced97c268f52e3" # 4.8.2026 revision
DATA = Path(__file__).resolve().parents[1] / "data"

SHA256 = {
    "webmainbench.jsonl": "85765fe798f07c14eb1c92945046eaa56e0da59663f70b9c498647d7dfd78884",
    "WebMainBench_545.jsonl": "0efaa4b49a45e320a27fe6e5a0b6aad5b57259fc3321ac3448519cacc74c537e",
}

def acquire() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for filename, expected in SHA256.items():
        path = hf_hub_download(
            REPO_ID, filename, repo_type=REPO_TYPE,
            revision=REVISION, local_dir=str(DATA),
        )
        got = sha256(path)
        if expected is None:
            print(f"record  {filename}  sha256={got}")
        elif got != expected:
            raise SystemExit(f"sha256 mismatch for {filename}: {got} != {expected}")
        else:
            print(f"ok      {filename}  {path}")

if __name__ == "__main__":
    acquire()