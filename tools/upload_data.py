"""
Push the built board data to the RunPod network volume over its S3 endpoint --
straight from the laptop, no pod running (uploads are free of pod billing). Pods
mount the volume at /workspace and read each board's files at vendors/<board>/data
through a per-board symlink, so the pipeline's hardcoded paths resolve unchanged.
Only the platform-neutral raw inputs ship (_runpod.BOARDS): the derived
blocks/feats/labels come out of the lxml render, whose byte output is libxml2-
version-specific, so they are built ON the pod (Linux libxml2) via pod_prep.sh and
must never cross from the laptop's libxml2. Idempotent: a re-run uploads only files
whose remote size differs.

    python tools/upload_data.py --stage list          # what's on the volume now
    python tools/upload_data.py --stage wipe           # dry run: manifest + sizes, deletes nothing
    python tools/upload_data.py --stage wipe --yes     # delete EVERYTHING (clean slate)
    python tools/upload_data.py                         # upload every board's raw, then verify
    python tools/upload_data.py --board daniel          # just one board's raw
    python tools/upload_data.py --stage verify          # remote sizes vs local

Needs boto3 and the RUNPOD_S3_* / RUNPOD_VOLUME_KEY / RUNPOD_DC keys in .env
(pip install .[runpod]).
"""

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from deploy.pod._runpod import BOARDS, ROOT, bucket, s3_client

# RunPod's S3 layer caps a single PutObject at 500 MB and loses parts uploaded in
# parallel (boto3's auto-multipart fails CompleteMultipartUpload). Above the cap
# we upload parts sequentially, per their own guidance.
BIG_FILE = 450 * 2**20
PART_SIZE = 128 * 2**20


def pairs(boards):
    """(local path, remote key data/<board>/<name>) for each board's raw inputs."""
    out = []
    for board in boards:
        data = ROOT / "vendors" / board / "data"
        for name in BOARDS[board]:
            p = data / name
            if not p.exists():
                sys.exit(f"missing local input: {p} -- build it before uploading")
            out.append((p, f"data/{board}/{name}"))
    return out


def list_remote(s3, prefix=""):
    """key -> size for everything under prefix (paginated)."""
    out, token = {}, None
    while True:
        kw = {"Bucket": bucket(), "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for obj in resp.get("Contents", []):
            out[obj["Key"]] = obj["Size"]
        if not resp.get("IsTruncated"):
            return out
        token = resp["NextContinuationToken"]


def upload_big(s3, path, key):
    stale = s3.list_multipart_uploads(Bucket=bucket(), Prefix=key)
    for up in stale.get("Uploads", []):
        s3.abort_multipart_upload(Bucket=bucket(), Key=up["Key"], UploadId=up["UploadId"])
        print(f"  aborted stale multipart for {up['Key']}")
    size = path.stat().st_size
    n = -(-size // PART_SIZE)
    print(f"  multipart {key}: {size / 2**30:.2f} GB in {n} sequential parts")
    upload_id = s3.create_multipart_upload(Bucket=bucket(), Key=key)["UploadId"]
    try:
        parts = []
        with open(path, "rb") as f:
            for i in range(1, n + 1):
                chunk = f.read(PART_SIZE)
                for attempt in range(3):
                    try:
                        r = s3.upload_part(Bucket=bucket(), Key=key, PartNumber=i,
                                           UploadId=upload_id, Body=chunk)
                        break
                    except Exception as e:
                        if attempt == 2:
                            raise
                        print(f"  part {i} attempt {attempt + 1} failed ({e}), retrying")
                        time.sleep(5)
                parts.append({"PartNumber": i, "ETag": r["ETag"]})
                print(f"  part {i}/{n} up", flush=True)
        s3.complete_multipart_upload(Bucket=bucket(), Key=key, UploadId=upload_id,
                                     MultipartUpload={"Parts": parts})
    except Exception:
        s3.abort_multipart_upload(Bucket=bucket(), Key=key, UploadId=upload_id)
        raise


def stage_upload(s3, boards):
    ps = pairs(boards)
    remote = list_remote(s3, "data/")
    todo = [(p, k) for p, k in ps if remote.get(k) != p.stat().st_size]
    print(f"{len(todo)} to upload, {len(ps) - len(todo)} already current")
    for p, k in todo:
        if p.stat().st_size > BIG_FILE:
            upload_big(s3, p, k)
        else:
            print(f"  {k} ({p.stat().st_size / 2**20:.0f} MB)", flush=True)
            s3.upload_file(str(p), bucket(), k)
    stage_verify(s3, boards)


def stage_verify(s3, boards):
    ps = pairs(boards)
    remote = list_remote(s3, "data/")
    bad = [(k, p.stat().st_size, remote.get(k))
           for p, k in ps if remote.get(k) != p.stat().st_size]
    for k, loc, rem in bad:
        print(f"  MISMATCH {k}: local {loc}, remote {rem}")
    if bad:
        sys.exit(f"verify FAILED: {len(bad)} of {len(ps)} files missing/mismatched")
    print(f"verify OK: {len(ps)} files match local sizes")


def _manifest(remote):
    by_top = defaultdict(lambda: [0, 0])
    for k, sz in remote.items():
        top = by_top[k.split("/", 1)[0] + "/" if "/" in k else k]
        top[0] += 1
        top[1] += sz
    for top, (n, sz) in sorted(by_top.items()):
        print(f"  {top:<20} {n:>6} objects  {sz / 2**30:>8.2f} GB")


def stage_wipe(s3, yes):
    remote = list_remote(s3)
    mp = s3.list_multipart_uploads(Bucket=bucket()).get("Uploads", [])
    if not remote and not mp:
        print("volume already empty")
        return
    print(f"volume {bucket()}: {len(remote)} objects, "
          f"{sum(remote.values()) / 2**30:.2f} GB, {len(mp)} in-progress multipart")
    _manifest(remote)
    if not yes:
        sys.exit("dry run -- re-run with --yes to DELETE ALL of the above")
    for up in mp:
        s3.abort_multipart_upload(Bucket=bucket(), Key=up["Key"], UploadId=up["UploadId"])
    # RunPod's S3 answers the batch DeleteObjects call with 307 Temporary Redirect
    # (archive, 2026-08-24); single-object delete works. Thread it -- 18k serial
    # round-trips are slow.
    keys = list(remote)
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for _ in pool.map(lambda k: s3.delete_object(Bucket=bucket(), Key=k), keys):
            done += 1
            if done % 1000 == 0 or done == len(keys):
                print(f"  deleted {done}/{len(keys)}", flush=True)
    print("wiped")


def stage_logs(s3, run):
    """Read a pod's logs back off the volume -- they persist there after the pod
    self-terminates. No --run lists the runs; --run <name> dumps its files."""
    try:
        sys.stdout.reconfigure(errors="replace")   # pod logs carry non-cp1252 glyphs
    except (AttributeError, ValueError):
        pass
    remote = list_remote(s3, "outputs/")
    if not remote:
        print("no outputs/ on the volume")
        return
    if not run:
        runs = defaultdict(lambda: [0, 0])
        for k, sz in remote.items():
            name = k.split("/")[1] if k.count("/") >= 2 else k
            runs[name][0] += 1
            runs[name][1] += sz
        for name, (n, sz) in sorted(runs.items()):
            print(f"  {name:<28} {n} files  {sz / 1024:.0f} KB")
        print("\npass --run <name> to dump its logs")
        return
    keys = sorted(k for k in remote if k.startswith(f"outputs/{run}/"))
    if not keys:
        print(f"no logs for run '{run}'")
        return
    for k in keys:
        body = s3.get_object(Bucket=bucket(), Key=k)["Body"].read().decode("utf-8", "replace")
        print(f"\n===== {k} ({len(body)} B) =====\n{body}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["upload", "list", "verify", "wipe", "logs"], default="upload")
    ap.add_argument("--board", choices=[*BOARDS, "all"], default="all",
                    help="which board's raw to upload/verify (default: all)")
    ap.add_argument("--yes", action="store_true", help="wipe: confirm deletion")
    ap.add_argument("--run", help="logs: dump this run's files (default: list runs)")
    args = ap.parse_args()
    boards = list(BOARDS) if args.board == "all" else [args.board]
    s3 = s3_client()
    if args.stage == "list":
        _manifest(list_remote(s3))
    elif args.stage == "wipe":
        stage_wipe(s3, args.yes)
    elif args.stage == "verify":
        stage_verify(s3, boards)
    elif args.stage == "logs":
        stage_logs(s3, args.run)
    else:
        stage_upload(s3, boards)


if __name__ == "__main__":
    main()
