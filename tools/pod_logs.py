"""
Read a pod's logs back after it self-terminates. Pods mount no volume; on exit each
pushes its log folder to $HF_RESULTS_DATASET under logs/<run>/ (bootstrap.log, run.log,
run_meta.json). No --run lists the runs; --run <name> dumps its files.

    python tools/pod_logs.py                 # list runs with logs
    python tools/pod_logs.py --run thr311    # dump one run's logs
"""
import argparse
import sys
from collections import defaultdict

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

from deploy.pod._runpod import env_or_die


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="dump this run's files (default: list runs)")
    args = ap.parse_args()
    load_dotenv()
    repo, token = env_or_die("HF_RESULTS_DATASET"), env_or_die("HF_TOKEN")
    files = [f for f in HfApi(token=token).list_repo_files(repo, repo_type="dataset")
             if f.startswith("logs/")]
    if not args.run:
        runs = defaultdict(int)
        for f in files:
            runs[f.split("/")[1]] += 1
        for name, n in sorted(runs.items()):
            print(f"  {name:<28} {n} files")
        print("\npass --run <name> to dump its logs" if runs else "no logs/ in the dataset")
        return
    keys = sorted(f for f in files if f.startswith(f"logs/{args.run}/"))
    if not keys:
        sys.exit(f"no logs for run '{args.run}'")
    try:
        sys.stdout.reconfigure(errors="replace")   # pod logs carry non-cp1252 glyphs
    except (AttributeError, ValueError):
        pass
    for k in keys:
        path = hf_hub_download(repo, k, repo_type="dataset", token=token)
        with open(path, encoding="utf-8", errors="replace") as f:
            body = f.read()
        print(f"\n===== {k} ({len(body)} B) =====\n{body}")


if __name__ == "__main__":
    main()
