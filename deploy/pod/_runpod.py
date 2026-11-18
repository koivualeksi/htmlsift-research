"""Shared RunPod plumbing for the operator tools (launch.py, tools/pod_logs.py): .env
loading, the RunPod REST call, and the private HF repos a volume-less pod reads its code
from and writes everything it produces to. Neither the pipeline nor any reproduction
path imports this -- it exists only to drive our own sweeps onto rented GPUs, and every
credential it reads is operator-side (gitignored .env), never in .env.example.

Pods mount no network volume: the code snapshot rides in $HF_RESULTS_DATASET under
code/, each board's data is built on the pod from its published source, and results,
kept ckpts and logs leave through HF before the pod deletes itself.

One-time setup: RunPod console -> Settings -> API Keys -> a REST key -> RUNPOD_API_KEY.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
API = "https://api.runpod.io/v2"              # REST v2; v1 (rest.runpod.io) retired 2026-11-15


def env_or_die(key: str) -> str:
    load_dotenv()
    val = os.environ.get(key, "")
    if not val:
        sys.exit(f"{key} not set -- add it to .env at the repo root (see deploy/pod/_runpod.py)")
    return val


class ApiError(RuntimeError):
    def __init__(self, method, path, code, detail):
        super().__init__(f"RunPod API {method} {path}: {code} {detail}")
        self.code = code


def api(method: str, path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path, method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {env_or_die('RUNPOD_API_KEY')}",
                 "Content-Type": "application/json",
                 # RunPod's REST edge is behind Cloudflare, which began 1010-banning
                 # urllib's default User-Agent (2026-09-07: curl and a browser UA get
                 # 200, bare urllib gets 403). Send a browser UA so the POST passes.
                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/125.0.0.0 Safari/537.36"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode()
    except urllib.error.HTTPError as e:
        raise ApiError(method, path, e.code, e.read().decode()) from e
    return json.loads(text) if text.strip() else {}


def require_private(repo_id: str, repo_type: str) -> None:
    """Everything a pod produces leaves through these repos, including ckpts trained on
    private pages and the code snapshot -- refuse to launch onto a public one."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = HfApi(token=env_or_die("HF_TOKEN")).repo_info(repo_id, repo_type=repo_type)
    except RepositoryNotFoundError:
        sys.exit(f"{repo_type} repo {repo_id} does not exist (or HF_TOKEN cannot read it) "
                 f"-- create it, private, before launching")
    if not info.private:
        sys.exit(f"{repo_type} repo {repo_id} is public -- pods write to it; make it private")


def upload_snapshot(blob: bytes, name: str) -> None:
    """The code tarball into $HF_RESULTS_DATASET at code/<name>; content-addressed, so a
    re-upload of unchanged code is a no-op commit."""
    from huggingface_hub import HfApi
    HfApi(token=env_or_die("HF_TOKEN")).upload_file(
        path_or_fileobj=blob, path_in_repo=f"code/{name}",
        repo_id=env_or_die("HF_RESULTS_DATASET"), repo_type="dataset",
        commit_message=f"pod code snapshot {name}")
