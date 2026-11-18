#!/usr/bin/env bash
# Pod-side bootstrap for one bench-harness run (bench/<name>.py). deploy/pod/launch.py
# delivers this base64-encoded in ENTRYPOINT_SCRIPT; the pod's start command decodes and
# runs it. No network volume: the code snapshot comes from $HF_RESULTS_DATASET (code/),
# every board in BOARDS (predict's --boards; WMB for the speed harnesses) is built here
# from its published source on this pod's Linux render, and the logs go back to the
# dataset (logs/$RUN_NAME/) on exit. The keeper is pulled from $HF_MODELS_REPO by
# resolve_ckpt; predict --push streams to $HF_RESULTS_DATASET, speed prints to the log.
#
# ALWAYS self-terminates via the EXIT trap, including on failure: a pod that
# can't clean up restarts and bills forever (archive, 2026-08-18).
#
# Env from launch.py: RUN_NAME, CODE_SNAPSHOT, RUN_MODE, SWEEP_ARGS, MAX_RUNTIME, BOARDS,
#   HF_TOKEN, HF_MODELS_REPO, HF_RESULTS_DATASET, RUNPOD_TERMINATE_API_KEY;
#   RUNPOD_POD_ID from RunPod; optional PUBLIC_KEY for the ssh debug hatch.
set -euo pipefail

export OUT=/root/out
mkdir -p "$OUT"
exec > >(tee -a "$OUT/bootstrap.log") 2>&1
START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Logs leave through HF's commit API with curl alone: the image's own python is not a safe
# place to depend on (its pip and python3 can be different interpreters).
push_logs() {
    local nd=/tmp/logs.ndjson f
    printf '{"key":"header","value":{"summary":"pod logs %s"}}\n' "$RUN_NAME" > "$nd"
    for f in "$OUT"/*; do
        printf '{"key":"file","value":{"path":"logs/%s/%s","encoding":"base64","content":"%s"}}\n' \
            "$RUN_NAME" "$(basename "$f")" "$(base64 -w0 "$f")" >> "$nd"
    done
    curl -sS --max-time 120 -X POST -H "Authorization: Bearer $HF_TOKEN" \
        -H "Content-Type: application/x-ndjson" --data-binary @"$nd" \
        "https://huggingface.co/api/datasets/$HF_RESULTS_DATASET/commit/main" -o /dev/null -w 'log push: HTTP %{http_code}\n'
}

cleanup() {
    status=$?
    set +e +u   # nothing may abort this before the terminate call
    printf '{"run_name":"%s","sweep_args":"%s","code_snapshot":"%s","exit_code":%d,"started":"%s","ended":"%s","gpu":"%s"}\n' \
        "${RUN_NAME:-}" "${SWEEP_ARGS:-}" "${CODE_SNAPSHOT:-}" "$status" "${START_TS:-}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo unknown)" \
        > "$OUT/run_meta.json"
    push_logs || echo "WARN: log push failed"
    echo "exit status $status -- terminating pod ${RUNPOD_POD_ID}..."
    for i in 1 2 3; do
        code=$(curl -s -o /tmp/term.txt -w '%{http_code}' -X DELETE \
            "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" \
            -H "Authorization: Bearer ${RUNPOD_TERMINATE_API_KEY}") || code=000
        echo "terminate attempt ${i}: HTTP ${code}"
        case "$code" in 2??) break ;; esac
        sleep 5
    done
    runpodctl remove pod "$RUNPOD_POD_ID" || true   # fallback: pod-scoped key RunPod injects
    sleep 300   # hold the container so RunPod doesn't restart it before termination lands
}
trap cleanup EXIT

# Our start command replaces the image's default start script (which launches
# sshd), so start sshd ourselves or the debug hatch is dead.
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
    ssh-keygen -A 2>/dev/null || true
    service ssh start || /usr/sbin/sshd || echo "WARN: sshd failed to start"
fi

echo "=== $RUN_NAME -- code $CODE_SNAPSHOT, boards $BOARDS ==="
mkdir -p /root/repo
curl -fsSL -H "Authorization: Bearer $HF_TOKEN" -o /root/code.tar.gz \
    "https://huggingface.co/datasets/$HF_RESULTS_DATASET/resolve/main/code/$CODE_SNAPSHOT"
tar xzf /root/code.tar.gz -C /root/repo

# The base image exports HF_HUB_ENABLE_HF_TRANSFER=1, but a fresh uv venv has no
# hf_transfer, which then makes every download raise -- turn it off (archive, 2026-08-26).
export HF_HUB_ENABLE_HF_TRANSFER=0

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
# eval.py reads its data at Path(__file__)/../data -- a non-editable install copies
# vendors/ into site-packages, so __file__ lands there and the data is not found.
uv pip install -q --python "$PYBIN" --no-cache -e ".[repro,train]"
"$PYBIN" -c "import sys, torch, lxml.etree as e; assert torch.cuda.is_available(), 'no CUDA'; print('python', sys.version.split()[0], '| torch', torch.__version__, torch.cuda.get_device_name(0), '| libxml2', e.LIBXML_VERSION)"

# Board data from the published source: prepare.py acquires (pinned + sha256-checked) and
# builds the board; blocks.py --feats ABC adds the structural features on the same render,
# and labels.py re-derives labels where the board derives them from the render.
for board in $BOARDS; do
    echo "--- prep $board ---"
    timeout "$MAX_RUNTIME" "$PYBIN" "vendors/$board/adapter/prepare.py"
    timeout "$MAX_RUNTIME" "$PYBIN" "vendors/$board/adapter/blocks.py" --feats ABC
    if [ -f "vendors/$board/adapter/labels.py" ]; then
        timeout "$MAX_RUNTIME" "$PYBIN" "vendors/$board/adapter/labels.py"
    fi
done

# wmb: prove serialize's render agrees with the freshly built block counts over val+test
# -- the exact invariant that failed cross-platform.
case " $BOARDS " in *" wmb "*)
timeout "$MAX_RUNTIME" "$PYBIN" - <<'PY'
import sys
sys.path.insert(0, "vendors/wmb/adapter")
import serialize, eval as ev
n = bad = 0
for fold in ("val", "test"):
    ids = ev.fold_ids(fold)
    blk, pages = ev.load_blocks(ids), ev.load_pages(ids)
    for tid, b in blk.items():
        if not b["blocks"]:
            continue
        n += 1
        try:
            serialize.prepare(pages[tid]["html"], pages[tid].get("url", ""), len(b["blocks"]))
        except AssertionError as e:
            bad += 1
            print("MISMATCH", tid, e)
print(f"serialize-vs-blocks: {n - bad}/{n} match (val+test)")
assert bad == 0, f"{bad} pages diverge -- Linux build inconsistent, do NOT run the sweep"
PY
;; esac

# Bench run. RUN_MODE (from launch.py) is the entry point -- bench/<name>.py.
# SWEEP_ARGS carries that harness's own args (--model ... --device cuda [--push]):
# speed prints to the pod log; predict --push streams to $HF_RESULTS_DATASET.
# shellcheck disable=SC2086 -- SWEEP_ARGS is intentionally word-split.
timeout "$MAX_RUNTIME" "$PYBIN" -u "$RUN_MODE" ${SWEEP_ARGS:-}     2>&1 | tee "$OUT/run.log"

echo "=== bench finished cleanly ==="
