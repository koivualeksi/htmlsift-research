#!/usr/bin/env bash
# Pod-side bootstrap for one bench-harness run (bench/<name>.py). deploy/pod/launch.py
# delivers this base64-encoded in ENTRYPOINT_SCRIPT; the pod's dockerStartCmd decodes
# and runs it, so the pod never touches git or a registry. Bench is multi-board:
# predict.py scores a keeper over every board in --args --boards, so this mounts every
# board present on the volume (data/<board>/), unlike pod_train.sh (WMB only). The
# keeper is pulled from $HF_MODELS_REPO by resolve_ckpt.
#
# ALWAYS self-terminates via the EXIT trap, including on failure: a pod that
# can't clean up restarts and bills forever (archive, 2026-08-18).
#
# Env from launch.py: RUN_NAME, CODE_SNAPSHOT, RUN_MODE, SWEEP_ARGS, MAX_RUNTIME,
#   HF_TOKEN, HF_MODELS_REPO, HF_RESULTS_DATASET, RUNPOD_TERMINATE_API_KEY;
#   RUNPOD_POD_ID from RunPod; optional PUBLIC_KEY for the ssh debug hatch.
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
# Mount every board present on the volume. A board dir with no blocks.jsonl is not
# prepped -- warn but do not fail: this run may not score it (that's in --args --boards),
# and predict.py raises on a board it needs but cannot read. Fail only if nothing is
# prepped. The hf cache lives at $VOL/hf, not under data/, so it is never a board dir.
mounted=""
for d in "$VOL"/data/*/; do
    board=$(basename "$d")
    [ -d "/root/repo/vendors/$board" ] || continue
    ln -sfn "$d" "/root/repo/vendors/$board/data"
    if [ -f "$d/blocks.jsonl" ]; then
        mounted="$mounted $board"
    else
        echo "WARN: $board mounted but not prepped (no blocks.jsonl) -- launch.py --prep --boards $board"
    fi
done
[ -n "$mounted" ] || { echo "ERROR: no prepped boards under $VOL/data -- run upload_data.py + launch.py --prep first"; exit 1; }
echo "bench boards mounted:$mounted"

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
# -e (editable): imports must resolve to /root/repo, not a copy under site-packages,
# so each board's vendors/<board>/adapter reads its data through the symlink above.
uv pip install -q --python "$PYBIN" --no-cache -e ".[repro,train]"
"$PYBIN" -c "import sys, torch; assert torch.cuda.is_available(), 'no CUDA'; print('python', sys.version.split()[0], '| torch', torch.__version__, torch.cuda.get_device_name(0))"

# Bench run. RUN_MODE (from launch.py) is the entry point -- bench/<name>.py.
# SWEEP_ARGS carries that harness's own args (--model ... --device cuda [--push]):
# speed prints to the pod log; predict --push streams to $HF_RESULTS_DATASET.
# shellcheck disable=SC2086 -- SWEEP_ARGS is intentionally word-split.
timeout "$MAX_RUNTIME" "$PYBIN" -u "$RUN_MODE" ${SWEEP_ARGS:-} \
    2>&1 | tee "$OUT/run.log"

echo "=== bench finished cleanly ==="
