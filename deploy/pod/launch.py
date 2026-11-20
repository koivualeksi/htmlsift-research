"""
Spawn one RunPod GPU pod that runs a training spec or a bench harness and self-terminates.
No network volume: the code snapshot goes to $HF_RESULTS_DATASET under code/, the pod pulls
it over HTTPS, builds each board it needs from the board's published source (the same
prepare.py a reproducer runs, on the pod's own Linux render), runs, and pushes its logs
back to the dataset under logs/<run>/ on exit. --spec runs `python -m trainers.train` on
that spec: the boards to build come from its configs, every run folder is pushed as it
ends, and a restarted container resumes the same launch. --bench runs a harness with
--args passed through verbatim. The pod deletes itself on exit (entrypoint EXIT trap +
MAX_RUNTIME watchdog).

Placement: European data centres only, Nordic first, read from RunPod's catalog; then
one unpinned pass per GPU across all of them, since the catalog has listed stock that a
create then refused.

    python deploy/pod/launch.py --spec experiments/wmb-mini.yaml --print-payload  # dry run
    python deploy/pod/launch.py --spec experiments/wmb-lite.yaml --smoke     # 1-epoch pod check
    python deploy/pod/launch.py --spec experiments/wmb-mini.yaml --queue 12  # wait for a GPU
    python deploy/pod/launch.py --queue-status <id>                # waiting / deployed / expired
    python deploy/pod/launch.py --bench predict --args "--model wmb-311m-10 --push"

Metered -- launches a paid pod (§2: human-run). Needs RUNPOD_API_KEY, HF_TOKEN and a
private HF_RESULTS_DATASET in .env, plus a private HF_MODELS_REPO for a bench run.
"""

import argparse
import base64
import hashlib
import io
import json
import shlex
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from _runpod import ROOT, ApiError, api, env_or_die, require_private, upload_snapshot

sys.path.insert(0, str(ROOT))
from trainers import runs, spec as specs  # noqa: E402 -- needs ROOT on the path

IMAGE = "runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204"
# Dripper's pod boots vLLM's own image (route B): vLLM 0.11.1 + torch 2.9 + CUDA pre-matched,
# so we never fight torch/driver versions on a metered pod. The tag must match mineru-html's
# vllm==0.11.1 pin exactly -- a newer image would make pip downgrade vLLM at install.
DRIPPER_IMAGE = "vllm/vllm-openai:v0.11.1"

# Container command: decode the base64 entrypoint script and run it. Sent as `cmd` (exec-form
# CMD) for images with no ENTRYPOINT (our pytorch base). The vLLM image sets ENTRYPOINT
# `vllm`, and RunPod APPENDS the CMD as args to an image ENTRYPOINT rather than replacing
# it (that fed `vllm /bin/bash -c ...` and the bootstrap never ran -- so its EXIT trap never
# armed and the pod would not self-terminate). So there it goes in `entrypoint`.
START_CMD = ["/bin/bash", "-c", 'echo "$ENTRYPOINT_SCRIPT" | base64 -d | /bin/bash']
# Preference order. All >=24 GB; the Blackwells run the cu128 torch the entrypoint installs.
GPU_FALLBACK = [
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX PRO 4500 Blackwell",
    "NVIDIA RTX PRO 4000 Blackwell",
    "NVIDIA A40",
]
# Membership list, not a floor -- omitting 13.0 once excluded the Blackwell hosts that had
# stock (archive, 2026-08-18); v2 matches exactly, and on 2026-11-17 the free hosts
# reported 13.2 only. The cu128 torch runs on all.
CUDA = ["12.8", "12.9", "13.0", "13.1", "13.2"]
NORDIC = ("EU-SE-", "EUR-IS-", "EUR-NO-")
ALL_BOARDS = "wmb,wcxb,daniel"


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
        tf.add(ROOT / "experiments", arcname="experiments", filter=keep)
    blob = buf.getvalue()
    return blob, f"wmb-{hashlib.sha256(blob).hexdigest()[:8]}.tar.gz"


def ssh_pubkey():
    for name in ("id_ed25519.pub", "id_rsa.pub"):
        p = Path.home() / ".ssh" / name
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return None


def _flag(sweep, name):
    """The value after `name` in a harness's arg string, or None."""
    toks = shlex.split(sweep)
    return toks[toks.index(name) + 1] if name in toks[:-1] else None


def train_env(path, smoke, launch, git):
    """A training spec -> (run name, boards, pod env). The spec is loaded and expanded here,
    so a bad spec fails on the laptop; the board checks need htmlsift and run on the pod
    (CHECK) before the board build."""
    path = Path(path).resolve()
    if path.parent != ROOT / "experiments":
        raise SystemExit(f"{path}: a pod runs specs from experiments/ (shipped in the snapshot)")
    s = specs.load(path)
    configs = specs.expand(s)
    entry = f"-m trainers.train experiments/{path.name}"
    env = {"ENTRY": entry + (" --smoke" if smoke else ""), "CHECK": entry + " --check",
           "LAUNCH": launch, "GIT_SHA": git[0], "GIT_DIRTY": "1" if git[1] else "0"}
    extras = {c["extra"] for c in configs if c.get("extra")}
    if extras:
        (env["EXTRA"],) = extras                 # one extra source exists: custom
    name = f"{s.name}-{launch}" + ("-smoke" if smoke else "")
    return name, ",".join(sorted({c["board"] for c in configs})), env


def candidates(gpus):
    """(gpu, [dc]) attempts: every European DC the catalog lists with stock for a GPU,
    Nordic first, in GPU preference order; then each GPU unpinned across all of them."""
    r = api("GET", "/catalog/datacenters?regions=EUROPE&include=GPU_AVAILABILITY")
    dcs = sorted(r.get("dataCenters", r) if isinstance(r, dict) else r,
                 key=lambda d: (not d["id"].startswith(NORDIC), d["id"]))
    listed = [(g, [d["id"]]) for g in gpus for d in dcs
              if any(a["id"] == g and a["availability"] != "NONE"
                     for a in d.get("gpuAvailability", []))]
    return listed + [(g, [d["id"] for d in dcs]) for g in gpus]


def launch(args):
    load_dotenv()
    blob, snapshot = build_code_tar()
    image, disk = IMAGE, 50                    # no volume: data, venv and caches all on disk
    sweep = args.args or ""
    entrypoint = "pod.sh"
    common = {"RUNPOD_TERMINATE_API_KEY": env_or_die("RUNPOD_API_KEY"),
              "CODE_SNAPSHOT": snapshot,
              "HF_TOKEN": env_or_die("HF_TOKEN"),
              "HF_RESULTS_DATASET": env_or_die("HF_RESULTS_DATASET")}

    if args.bench:
        # A bench harness (bench/speed/<name>.py, or bench/accuracy/predict.py) via the
        # generic entrypoint. predict scores every board in its --boards (or the one
        # --benchmark in --ckpt mode); the speed harnesses are WMB-only.
        run_name = args.name or f"bench-{args.bench}"
        max_runtime = args.max_runtime or "2h"
        boards = ((_flag(sweep, "--boards") or _flag(sweep, "--benchmark") or ALL_BOARDS)
                  if args.bench == "predict" else "wmb")
        harness = (f"bench/accuracy/{args.bench}.py" if args.bench == "predict"
                   else f"bench/speed/{args.bench}.py")
        env_vars = {"ENTRY": f"{harness} {sweep}".strip(),
                    "HF_MODELS_REPO": env_or_die("HF_MODELS_REPO")}
    elif args.dripper:
        # Dripper (bench/speed/gpu_dripper.py) on its OWN pod: route B boots vLLM's image (vLLM + torch
        # 2.9 pre-matched), pod_dripper.sh adds mineru-html and the scorer's leaf deps. WMB raw only
        # -- Dripper reads raw HTML, not our render. Bigger disk for the ~14 GB vLLM image.
        run_name = args.name or "dripper"
        max_runtime = args.max_runtime or "2h"
        image, disk = DRIPPER_IMAGE, 60
        entrypoint = "pod_dripper.sh"
        boards = "wmb"
        env_vars = {"RUN_MODE": "bench/speed/gpu_dripper.py", "SWEEP_ARGS": sweep}
    else:
        name, boards, env_vars = train_env(args.spec, args.smoke, runs.new_id(),
                                           runs.git_provenance())
        run_name = args.name or name
        max_runtime = args.max_runtime or ("60m" if args.smoke else "5h")
    env_vars = {"ENTRYPOINT_SCRIPT": base64.b64encode(
                    (ROOT / "deploy" / "pod" / entrypoint).read_bytes()).decode(),
                **common, **env_vars, "RUN_NAME": run_name,
                "MAX_RUNTIME": max_runtime, "BOARDS": boards.replace(",", " ")}
    pub = ssh_pubkey()
    if pub:
        env_vars["PUBLIC_KEY"] = pub

    payload = {
        "name": f"htmlsift-{run_name}",
        "image": image,
        "gpu": {"id": None, "count": 1, "allowedCudaVersions": CUDA},
        "dataCenterIds": None,
        "disk": disk,
        "ports": ["22/tcp"],
        "env": env_vars,
        "cmd": START_CMD,
    }
    if args.dripper:
        # vLLM image ENTRYPOINT is `vllm`; override it so our bootstrap runs (and its EXIT
        # trap arms -- otherwise the pod never self-terminates).
        payload["entrypoint"] = START_CMD
        payload["cmd"] = []

    gpus = [args.gpu] if args.gpu else GPU_FALLBACK
    tries = candidates(gpus)
    if args.print_payload:
        shown = {k: (v if k in ("RUN_NAME", "CODE_SNAPSHOT", "ENTRY", "CHECK", "LAUNCH",
                                "GIT_SHA", "GIT_DIRTY", "EXTRA", "RUN_MODE", "SWEEP_ARGS",
                                "BOARDS", "MAX_RUNTIME") else f"<{len(v)} chars>")
                 for k, v in env_vars.items()}
        print(json.dumps(dict(payload, env=shown), indent=2))
        print("\nattempt order:")
        for g, dcs in tries:
            print(f"  {g} @ {', '.join(dcs)}")
        print(f"\n(dry run) would upload code/{snapshot} ({len(blob) / 1024:.0f} KB) "
              f"and launch {run_name}")
        return

    require_private(env_or_die("HF_RESULTS_DATASET"), "dataset")
    if args.spec:
        runs.preflight()                         # writable results dataset, provenance
    push = _flag(sweep, "--push")                # --push <repo> overrides the results dataset
    if push and not push.startswith("--"):
        require_private(push, "dataset")
    if "HF_MODELS_REPO" in env_vars:
        require_private(env_vars["HF_MODELS_REPO"], "model")
    upload_snapshot(blob, snapshot)
    print(f"code snapshot uploaded: code/{snapshot} ({len(blob) / 1024:.0f} KB)")

    # v2 returns a stock-out as a plain 400 with only a human-readable detail, the same code
    # as a malformed request. So any 400 falls through to the next attempt; a 400 on every one
    # is a request problem as likely as a stock-out, and the last detail is printed to tell.
    result, last = None, None
    print(f"launching {run_name}: boards {boards}, watchdog {max_runtime}, "
          f"\"{env_vars.get('ENTRY') or sweep}\"")
    for gpu, dcs in tries:
        payload["gpu"]["id"], payload["dataCenterIds"] = gpu, dcs
        try:
            result = api("POST", "/pods", body=payload)
            print(f"  {gpu} @ {result.get('dataCenterId') or ', '.join(dcs)}: created")
            break
        except ApiError as e:
            if e.code != 400:
                raise
            last = e
            print(f"  {gpu} @ {dcs[0] if len(dcs) == 1 else 'any EU'}: refused (400)")
    if result is None and args.queue:
        # One queued pod for the first-choice GPU, any European DC: RunPod deploys it when a
        # machine frees up, before expiresAt. One only, so a capacity burst cannot start
        # duplicate runs. A queued create still tries to deploy first, so 201 is possible.
        payload["gpu"]["id"], payload["dataCenterIds"] = gpus[0], tries[-1][1]
        expires = datetime.now(timezone.utc) + timedelta(hours=args.queue)
        payload["queue"] = {"expiresAt": expires.isoformat(timespec="seconds")}
        result = api("POST", "/pods", body=payload)
        if "podId" in result:                                  # 202: the queued-pod record
            print(f"queued {result['id']} ({run_name}): {gpus[0]} in any EU DC, status "
                  f"{result['status']}, waits until {result['queue']['expiresAt']} UTC.")
            print(f"  status: python deploy/pod/launch.py --queue-status {result['id']}")
            print(f"  cancel: python deploy/pod/launch.py --queue-cancel {result['id']}")
            return
    if result is None:
        sys.exit(f"every attempt refused -- a stock-out or a bad request; last: {last}"
                 + ("" if args.queue else "\n  --queue <hours> waits for capacity instead"))
    pod_id = result["id"]
    print(f"launched pod {pod_id} ({run_name}). Self-terminates on exit; logs -> "
          f"{common['HF_RESULTS_DATASET']} logs/{run_name}/.")
    print(f"  console: https://console.runpod.io/pods?id={pod_id}")
    print(f"  logs:    python tools/pod_logs.py --run {run_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", help="experiments/<name>.yaml to train on the pod")
    ap.add_argument("--args", help="a bench harness's own args, passed through verbatim")
    ap.add_argument("--name", default=None,
                    help="pod/run name (default <spec name>-<launch id>, or bench-<harness>)")
    ap.add_argument("--gpu", help="pin ONE gpu id (default: the GPU_FALLBACK order)")
    ap.add_argument("--max-runtime", help="watchdog, e.g. 5h/90m (default 5h; 60m for --smoke)")
    ap.add_argument("--smoke", action="store_true",
                    help="train the spec as a smoke (20 pages, 1 epoch, seed 0); pushes no runs")
    ap.add_argument("--bench", choices=["stages", "predict", "gpu_throughput"],
                    help="run a bench harness on the pod (bench/speed/<name>.py, or "
                         "bench/accuracy/predict.py) instead of a trainer; --args are that "
                         "harness's own args (--model ... --device cuda)")
    ap.add_argument("--dripper", action="store_true",
                    help="run Dripper (bench/speed/gpu_dripper.py) on its own vLLM-image pod (route B); "
                         "--args are gpu_dripper.py's own args (--device cuda [--fold ...])")
    ap.add_argument("--print-payload", action="store_true",
                    help="dry run: print the pod payload and attempt order, upload and launch nothing")
    ap.add_argument("--queue", type=float, metavar="HOURS",
                    help="when every attempt is refused, queue one pod for the first-choice GPU "
                         "that RunPod deploys once capacity frees up, waiting at most HOURS")
    ap.add_argument("--queue-status", metavar="ID", help="show a queued pod and exit")
    ap.add_argument("--queue-cancel", metavar="ID", help="cancel a queued pod and exit")
    args = ap.parse_args()
    if args.queue_status or args.queue_cancel:
        qid = args.queue_status or args.queue_cancel
        if args.queue_cancel:
            api("DELETE", f"/queued-pods/{qid}")
        q = api("GET", f"/queued-pods/{qid}")
        print(f"{q['id']} {q['name']}: {q['status']}, gpu {q['gpu']['id']}, "
              f"expires {q['queue']['expiresAt']}"
              + (f", pod {q['podId']}" if q.get("podId") else "")
              + (f", failed: {q['failureReason']}" if q.get("failureReason") else ""))
        return
    if sum(map(bool, (args.spec, args.bench, args.dripper))) != 1:
        ap.error("exactly one of --spec, --bench or --dripper is required")
    if args.smoke and not args.spec:
        ap.error("--smoke trains a spec")
    launch(args)


if __name__ == "__main__":
    main()
