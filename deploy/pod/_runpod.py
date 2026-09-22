"""Shared RunPod plumbing for the two operator tools (upload_data.py, launch.py):
.env loading, the network-volume S3 client, and the REST call for pod launch.
Neither the pipeline nor any reproduction path imports this -- it exists only to
drive our own sweeps onto rented GPUs, and every credential it reads is
operator-side (gitignored .env), never in .env.example.

One-time setup (RunPod console, EU-RO-1 -- the DC that has 4090s AND the S3 API):
  Storage        -> the network volume id                    -> RUNPOD_VOLUME_KEY
  Settings       -> API Keys    -> a REST key                 -> RUNPOD_API_KEY
  Settings       -> S3 API Keys -> a key (secret shown once)  -> RUNPOD_S3_SECRET_ACCESS_KEY
  the access-key id is the account user id (not shown there)  -> RUNPOD_S3_ACCESS_KEY
  RUNPOD_DC=EU-RO-1   (endpoint derives from it; override with RUNPOD_ENDPOINT_URL)
"""

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
API = "https://rest.runpod.io/v1/pods"

# Per-board raw inputs uploaded to the volume under data/<board>/. Platform-neutral
# only: the derived blocks/feats/labels are libxml2-version-specific and built ON the
# pod (pod/pod_prep.sh), never uploaded. Primary raw is always <board>.jsonl (the
# pod scripts rely on that convention); wmb also carries the domain splits.
BOARDS = {
    "wmb":    ["wmb.jsonl", "splits.json"],
    "wcxb":   ["wcxb.jsonl"],
    "daniel": ["daniel.jsonl"],
}


def env_or_die(key: str) -> str:
    load_dotenv()
    val = os.environ.get(key, "")
    if not val:
        sys.exit(f"{key} not set -- add it to .env at the repo root (see deploy/pod/_runpod.py)")
    return val


def bucket() -> str:
    return env_or_die("RUNPOD_VOLUME_KEY")


def s3_client():
    """boto3 client for the volume's S3 endpoint. boto3 is an operator-only dep
    (pip install boto3); imported lazily so launch.py, which is REST-only, needs
    neither it nor the S3 credentials."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        sys.exit("boto3 missing -- pip install boto3")
    dc = env_or_die("RUNPOD_DC")
    endpoint = os.environ.get("RUNPOD_ENDPOINT_URL", f"https://s3api-{dc.lower()}.runpod.io")
    return boto3.client(
        "s3", endpoint_url=endpoint, region_name=dc.lower(),
        aws_access_key_id=env_or_die("RUNPOD_S3_ACCESS_KEY"),
        aws_secret_access_key=env_or_die("RUNPOD_S3_SECRET_ACCESS_KEY"),
        # boto3 >=1.36 attaches CRC32 checksum trailers by default; RunPod's S3
        # layer rejects them with SignatureDoesNotMatch -- disable unless required.
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"},
                      request_checksum_calculation="when_required",
                      response_checksum_validation="when_required"))


def api(method: str, path: str = "", body: dict | None = None) -> dict:
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
        raise RuntimeError(f"RunPod API {method} {path}: {e.code} {e.read().decode()}") from e
    return json.loads(text) if text.strip() else {}
