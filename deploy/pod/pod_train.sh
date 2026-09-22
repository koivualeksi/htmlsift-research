#!/usr/bin/env bash
# Pod-side bootstrap for one WMB trainer run. deploy/pod/launch.py delivers this
# base64-encoded in ENTRYPOINT_SCRIPT; the pod's dockerStartCmd decodes and runs
# it, so the pod never touches git or a registry. The network volume mounts at
# /workspace: data/<board>/ (each prepped board's files), code/ (repo tarballs),
# outputs/$RUN_NAME/ (logs -- results themselves go to HF via the trainer's --push).
# Every board present on the volume is symlinked; the trainer's --benchmark picks one.
#
# ALWAYS self-terminates via the EXIT trap, including on failure: a pod that
# can't clean up restarts and bills forever (archive, 2026-08-18).
#
# Env from launch.py: RUN_NAME, CODE_SNAPSHOT, RUN_MODE, SWEEP_ARGS, MAX_RUNTIME,
#   HF_TOKEN, HF_RESULTS_DATASET, RUNPOD_TERMINATE_API_KEY; HF_MODELS_REPO when the
#   sweep carries --keep; RUNPOD_POD_ID from RunPod; optional PUBLIC_KEY for the ssh
#   debug hatch.
set -euo pipefail

VOL=/workspace
OUT="$VOL/outputs/$RUN_NAME"
mkdir -p "$OUT"
date -u +'entrypoint started %Y-%m-%dT%H:%M:%SZ' > "$OUT/STARTED"   # first write: proves it booted
exec > >(tee -a "$OUT/bootstrap.log") 2>&1
START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cleanup() {
    status=$?
    set +e +u   # nothing may abort this before the terminate call
    printf '{"run_name":"%s","sweep_args":"%s","code_snapshot":"%s","exit_code":%d,"started":"%s","ended":"%s","gpu":"%s"}\n' \
        "${RUN_NAME:-}" "${SWEEP_ARGS:-}" "${CODE_SNAPSHOT:-}" "$status" "${START_TS:-}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)" \
        > "$OUT/run_meta.json" || true
    echo "exit status $status -- terminating pod ${RUNPOD_POD_ID}..."
    for i in 1 2 3; do
        code=$(curl -s -o /tmp/term.txt -w '%{http_code}' -X DELETE \
            "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" \
            -H "Authorization: Bearer ${RUNPOD_TERMINATE_API_KEY}") || code=000
        echo "terminate attempt ${i}: HTTP ${code}"
        case "$code" in 2??) break ;; esac
        sleep 5
    done
    runpodctl remove pod "$RUNPOD_POD_ID" || true   # fallback: pod-scoped key RunPod injects
    sleep 300   # hold the container so RunPod doesn't restart it before termination lands
}
trap cleanup EXIT

# Our dockerStartCmd replaces the image's default start script (which launches
# sshd), so start sshd ourselves or the debug hatch is dead.
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
    ssh-keygen -A 2>/dev/null || true
    service ssh start || /usr/sbin/sshd || echo "WARN: sshd failed to start"
fi

echo "=== $RUN_NAME -- code $CODE_SNAPSHOT ==="
test -f "$VOL/code/$CODE_SNAPSHOT" || { echo "ERROR: code snapshot missing -- launch.py uploads it"; exit 1; }

mkdir -p /root/repo
tar xzf "$VOL/code/$CODE_SNAPSHOT" -C /root/repo
# volume data -> the pipeline's hardcoded path, one symlink per prepped board
# (mirrors pod_prep.sh / pod_bench.sh; the trainer's --benchmark picks which it reads).
for d in "$VOL"/data/*/; do
    board=$(basename "$d")
    [ -d "/root/repo/vendors/$board" ] && ln -sfn "$VOL/data/$board" "/root/repo/vendors/$board/data"
done

# granite caches on the volume: downloads once, survives pod death. The base image
# exports HF_HUB_ENABLE_HF_TRANSFER=1, but a fresh uv venv has no hf_transfer, which
# then makes every download raise -- turn it off (archive, 2026-08-26).
export HF_HOME="$VOL/hf"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$HF_HOME"

# py3.13 venv: matches the interpreter the render gates were validated on (py3.12
# rendered 1/730 val pages one line off). torch from the cu128 index (exclusive)
# so we get the CUDA build, not PyPI's cpu wheel; the [train] extra then sees
# torch==2.11.0 already satisfied.
cd /root/repo
pip install -q --no-cache-dir uv
uv venv --clear /root/venv --python 3.13
PYBIN=/root/venv/bin/python
uv pip install -q --python "$PYBIN" --no-cache torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
# -e (editable): imports must resolve to /root/repo, not a copy under site-packages.
# eval.py reads its data at Path(__file__)/../data, and the data symlink lives on the
# /root/repo tree -- a non-editable install copies vendors/ into site-packages, so
# __file__ lands there and the symlink is bypassed (splits.json FileNotFoundError).
uv pip install -q --python "$PYBIN" --no-cache -e ".[repro,train]"
"$PYBIN" -c "import sys, torch; assert torch.cuda.is_available(), 'no CUDA'; print('python', sys.version.split()[0], '| torch', torch.__version__, torch.cuda.get_device_name(0))"

# WMB trainer run. RUN_MODE (from launch.py) is the entry point -- trainers/<mode>.py.
# SWEEP_ARGS carries the axes AND --push: a real run streams to $HF_RESULTS_DATASET and
# resumes on restart; a smoke omits --push so it never touches the results repo. --device
# defaults to cuda.
# shellcheck disable=SC2086 -- SWEEP_ARGS is intentionally word-split.
timeout "$MAX_RUNTIME" "$PYBIN" -u "$RUN_MODE" ${SWEEP_ARGS:-} \
    2>&1 | tee "$OUT/run.log"

echo "=== sweep finished cleanly ==="
