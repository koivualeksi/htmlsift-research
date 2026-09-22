"""
Spawn one RunPod GPU pod that runs a WMB trainer and self-terminates. Snapshots the
repo code to the volume's code/ (S3), then REST-creates a pod with the volume mounted
at /workspace and the pod shape's entrypoint (pod_train.sh / pod_bench.sh / pod_prep.sh)
delivered base64. --trainer picks the entry
point (trainers/<name>.py); --args is that trainer's own sweep args, passed through
verbatim (this launcher never parses them -- run the trainer with --help for its args).
Results stream to $HF_RESULTS_DATASET when --args carries --push; the pod deletes itself
on exit (entrypoint EXIT trap + MAX_RUNTIME watchdog).

    python pod/launch.py --trainer frozen --print-payload   # dry run: nothing uploaded
    python pod/launch.py --trainer finetune --smoke         # tiny end-to-end pod (no push)
    python pod/launch.py --trainer finetune --name wmb-ft-311m \
        --args "--models 311m --layers 9,10,11,12,13,22 --heads bigru --feats none --seeds 1 --push"

Metered -- launches a paid pod (§2: human-run). Needs RUNPOD_API_KEY + the
S3/volume keys + HF_TOKEN + HF_RESULTS_DATASET in .env (plus HF_MODELS_REPO when the
sweep carries --keep), and boto3 (.[runpod]).
"""

import argparse
import base64
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

from dotenv import load_dotenv

from _runpod import ROOT, api, bucket, env_or_die, s3_client

IMAGE = "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204"
# Dripper's pod boots vLLM's own image (route B): vLLM 0.11.1 + torch 2.9 + CUDA pre-matched,
# so we never fight torch/driver versions on a metered pod. The tag must match mineru-html's
# vllm==0.11.1 pin exactly -- a newer image would make pip downgrade vLLM at install.
DRIPPER_IMAGE = "vllm/vllm-openai:v0.11.1"

# Container command: decode the base64 entrypoint script and run it. Sent as dockerStartCmd
# (CMD) for images with no ENTRYPOINT (our pytorch base). The vLLM image sets ENTRYPOINT
# `vllm`, and RunPod APPENDS dockerStartCmd as args to an image ENTRYPOINT rather than
# replacing it (that fed `vllm /bin/bash -c ...` and the bootstrap never ran -- so its EXIT
# trap never armed and the pod would not self-terminate). So there it goes in dockerEntrypoint.
START_CMD = ["/bin/bash", "-c", 'echo "$ENTRYPOINT_SCRIPT" | base64 -d | /bin/bash']
# Preference order; launch falls through on a stock-out (EU-RO-1 churns). All
# >=24 GB; the Blackwells run the cu128 torch the entrypoint installs anyway.
GPU_FALLBACK = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA RTX PRO 4000 Blackwell",
]
MIN_DOWNLOAD_MBPS = 1000                 # avoids hours-slow image pulls


def build_code_tar():
    """Repo code (core + vendors + pyproject) as a gz tarball, named by its sha8.
    Excludes data, caches, and backups -- only what the pod needs to import."""
    def keep(ti):
        parts = ti.name.split("/")
        if "__pycache__" in parts or ti.name.endswith((".pyc", ".bak")):
            return None
        if parts[0] == "vendors" and "data" in parts[1:]:      # vendors/<board>/data
            return None
        return ti
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(ROOT / "pyproject.toml", arcname="pyproject.toml")
        tf.add(ROOT / "core", arcname="core", filter=keep)
        tf.add(ROOT / "vendors", arcname="vendors", filter=keep)
        tf.add(ROOT / "trainers", arcname="trainers", filter=keep)
        tf.add(ROOT / "bench", arcname="bench", filter=keep)
    blob = buf.getvalue()
    return blob, f"wmb-{hashlib.sha256(blob).hexdigest()[:8]}.tar.gz"


def ssh_pubkey():
    for name in ("id_ed25519.pub", "id_rsa.pub"):
        p = Path.home() / ".ssh" / name
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def launch(args):
    load_dotenv()
    blob, snapshot = build_code_tar()
    image, disk = IMAGE, 20                    # per-shape; the Dripper pod overrides both

    if args.bench:
        # Run a bench harness on the pod (bench/speed/<name>.py, or bench/accuracy/predict.py)
        # via the generic entrypoint; RUN_MODE + SWEEP_ARGS drive it exactly as a trainer.
        # speed prints to the pod log; predict --push streams to the results dataset.
        # HF_MODELS_REPO is always set -- resolve_ckpt pulls the keeper from it.
        sweep = args.args or ""
        run_name = args.name or f"bench-{args.bench}"
        max_runtime = args.max_runtime or "2h"
        entrypoint = ROOT / "deploy" / "pod" / "pod_bench.sh"
        env_vars = {
            "ENTRYPOINT_SCRIPT": base64.b64encode(entrypoint.read_bytes()).decode(),
            "RUNPOD_TERMINATE_API_KEY": env_or_die("RUNPOD_API_KEY"),
            "RUN_NAME": run_name,
            "CODE_SNAPSHOT": snapshot,
            "RUN_MODE": (f"bench/accuracy/{args.bench}.py" if args.bench == "predict"
                         else f"bench/speed/{args.bench}.py"),
            "SWEEP_ARGS": sweep,
            "MAX_RUNTIME": max_runtime,
            "HF_TOKEN": env_or_die("HF_TOKEN"),
            "HF_MODELS_REPO": env_or_die("HF_MODELS_REPO"),
            "HF_RESULTS_DATASET": env_or_die("HF_RESULTS_DATASET"),
        }
    elif args.dripper:
        # Dripper (bench/speed/gpu_dripper.py) on its OWN pod: route B boots vLLM's image (vLLM + torch
        # 2.9 pre-matched), pod_dripper.sh adds mineru-html + .[repro]. No model or results repo
        # -- dripper.py prints its report to the pod log and pushes nothing. Bigger container
        # disk for the ~14 GB vLLM image.
        sweep = args.args or ""
        run_name = args.name or "dripper"
        max_runtime = args.max_runtime or "2h"
        image, disk = DRIPPER_IMAGE, 40
        entrypoint = ROOT / "deploy" / "pod" / "pod_dripper.sh"
        env_vars = {
            "ENTRYPOINT_SCRIPT": base64.b64encode(entrypoint.read_bytes()).decode(),
            "RUNPOD_TERMINATE_API_KEY": env_or_die("RUNPOD_API_KEY"),
            "RUN_NAME": run_name,
            "CODE_SNAPSHOT": snapshot,
            "RUN_MODE": "bench/speed/gpu_dripper.py",
            "SWEEP_ARGS": sweep,
            "MAX_RUNTIME": max_runtime,
            "HF_TOKEN": env_or_die("HF_TOKEN"),
        }
    elif args.prep:
        # Build each --boards board's derived data (blocks/feats, + labels where derived)
        # on the pod's Linux libxml2 (no HF, no sweep). BOARDS drives pod_prep.sh's loop.
        boards = args.boards.replace(",", " ")
        run_name = args.name or "prep"
        max_runtime = args.max_runtime or "45m"
        entrypoint = ROOT / "deploy" / "pod" / "pod_prep.sh"
        env_vars = {
            "ENTRYPOINT_SCRIPT": base64.b64encode(entrypoint.read_bytes()).decode(),
            "RUNPOD_TERMINATE_API_KEY": env_or_die("RUNPOD_API_KEY"),
            "RUN_NAME": run_name,
            "CODE_SNAPSHOT": snapshot,
            "BOARDS": boards,
            "MAX_RUNTIME": max_runtime,
        }
    else:
        sweep = args.args or ""
        if args.smoke:
            if "--push" in sweep:
                sys.exit("--smoke never touches the results repo: drop --push from --args")
            sweep = (sweep + " --limit 20 --epochs 1").strip()
        base = args.name or f"wmb-{args.trainer}"
        run_name = base + ("-smoke" if args.smoke else "")
        max_runtime = args.max_runtime or ("60m" if args.smoke else "5h")
        entrypoint = ROOT / "deploy" / "pod" / "pod_train.sh"
        env_vars = {
            "ENTRYPOINT_SCRIPT": base64.b64encode(entrypoint.read_bytes()).decode(),
            "RUNPOD_TERMINATE_API_KEY": env_or_die("RUNPOD_API_KEY"),
            "RUN_NAME": run_name,
            "CODE_SNAPSHOT": snapshot,
            "RUN_MODE": f"trainers/{args.trainer}.py",
            "SWEEP_ARGS": sweep,
            "MAX_RUNTIME": max_runtime,
            "HF_TOKEN": env_or_die("HF_TOKEN"),
            "HF_RESULTS_DATASET": env_or_die("HF_RESULTS_DATASET"),
        }
        # A --keep run pushes keeper ckpts to the model repo (distinct from the
        # results dataset); the trainer sys.exits at push time without it. Resolve
        # here so the failure is at launch, not after hours of training.
        if "--keep" in sweep:
            env_vars["HF_MODELS_REPO"] = env_or_die("HF_MODELS_REPO")
    pub = ssh_pubkey()
    if pub:
        env_vars["PUBLIC_KEY"] = pub

    gpus = [args.gpu] if args.gpu else GPU_FALLBACK
    payload = {
        "name": f"htmlsift-{run_name}",
        "imageName": image,
        "gpuTypeIds": [gpus[0]],
        "gpuCount": 1,
        # membership list, not a floor -- omitting 13.0 once excluded the Blackwell
        # hosts that actually had stock (archive, 2026-08-18).
        "allowedCudaVersions": ["12.8", "12.9", "13.0"],
        "minDownloadMbps": MIN_DOWNLOAD_MBPS,
        "containerDiskInGb": disk,
        "networkVolumeId": env_or_die("RUNPOD_VOLUME_KEY"),
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "env": env_vars,
        "dockerStartCmd": START_CMD,
    }
    if args.dripper:
        # vLLM image ENTRYPOINT is `vllm`; override it so our bootstrap runs (and its EXIT
        # trap arms -- otherwise the pod never self-terminates).
        payload["dockerEntrypoint"] = START_CMD
        payload["dockerStartCmd"] = []

    if args.print_payload:
        shown = {k: (v if k in ("RUN_NAME", "CODE_SNAPSHOT", "RUN_MODE", "SWEEP_ARGS", "BOARDS", "MAX_RUNTIME")
                     else f"<{len(v)} chars>") for k, v in env_vars.items()}
        print(json.dumps(dict(payload, env=shown), indent=2))
        print(f"\n(dry run) would upload code/{snapshot} ({len(blob) / 1024:.0f} KB) "
              f"and launch {run_name}")
        return

    s3 = s3_client()
    s3.put_object(Bucket=bucket(), Key=f"code/{snapshot}", Body=blob)
    print(f"code snapshot uploaded: code/{snapshot} ({len(blob) / 1024:.0f} KB)")

    detail = (f'prep {env_vars["BOARDS"]} (build blocks/feats/labels)' if args.prep
              else f'bench/{args.bench} "{env_vars["SWEEP_ARGS"]}"' if args.bench
              else f'dripper "{env_vars["SWEEP_ARGS"]}"' if args.dripper
              else f'sweep "{env_vars["SWEEP_ARGS"]}"')
    result = None
    for gpu in gpus:
        payload["gpuTypeIds"] = [gpu]
        print(f"launching {run_name}: {gpu}, watchdog {max_runtime}, {detail}")
        try:
            result = api("POST", body=payload)
            break
        except RuntimeError as e:
            if "no instances currently available" not in str(e):
                raise
            print(f"  no {gpu} available, falling through")
    if result is None:
        sys.exit(f"no instances for any of {gpus} right now -- retry in a few minutes")
    pod_id = result["id"]
    tail = (f'builds blocks/feats/labels onto the volume for: {env_vars["BOARDS"]}' if args.prep
            else "speed -> pod log; predict --push -> $HF_RESULTS_DATASET" if args.bench
            else "report -> pod log (retrieve with tools/upload_data.py --stage logs)" if args.dripper
            else "results -> $HF_RESULTS_DATASET via --push")
    print(f"launched pod {pod_id} ({run_name}). Self-terminates on exit; {tail}.")
    print(f"  console: https://console.runpod.io/pods?id={pod_id}")
    print(f"  status:  runpodctl get pod {pod_id}")
    print(f"  logs:    python tools/upload_data.py --stage logs --run {run_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer", choices=["finetune", "frozen", "features_only",
                                          "embedding_table"],
                    help="which trainer to run (trainers/<name>.py); required unless --prep")
    ap.add_argument("--args", help="the trainer's sweep args, passed through verbatim "
                                   "(omit to use the trainer's own defaults; see its --help)")
    ap.add_argument("--name", default=None, help="pod/run name (default wmb-<trainer>)")
    ap.add_argument("--gpu", help="pin ONE gpuTypeId (default: 4090 -> Blackwell fallback)")
    ap.add_argument("--max-runtime", help="watchdog, e.g. 5h/90m (default 5h; 60m for --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end pod (97m-6, 20 pages, 1 epoch); does NOT push to HF")
    ap.add_argument("--prep", action="store_true",
                    help="data-prep pod: build each --boards board's blocks/feats/labels on "
                         "the volume under the pod's Linux libxml2 (run once before sweeps)")
    ap.add_argument("--boards", default="wmb,wcxb,daniel",
                    help="--prep: comma-separated boards to build (default: all three)")
    ap.add_argument("--bench", choices=["stages", "predict", "gpu_throughput"],
                    help="run a bench harness on the pod (bench/speed/<name>.py, or "
                         "bench/accuracy/predict.py) instead of a trainer; --args are that "
                         "harness's own args (--model ... --device cuda)")
    ap.add_argument("--dripper", action="store_true",
                    help="run Dripper (bench/speed/gpu_dripper.py) on its own vLLM-image pod (route B); "
                         "--args are gpu_dripper.py's own args (--device cuda [--fold ...])")
    ap.add_argument("--print-payload", action="store_true",
                    help="dry run: print the pod payload, upload and launch nothing")
    args = ap.parse_args()
    if not args.prep and not args.trainer and not args.bench and not args.dripper:
        ap.error("one of --trainer, --bench, --dripper, or --prep is required")
    launch(args)


if __name__ == "__main__":
    main()
