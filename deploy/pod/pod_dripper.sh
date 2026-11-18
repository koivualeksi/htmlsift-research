#!/usr/bin/env bash
# Pod-side bootstrap for the Dripper GPU comparison (bench/speed/gpu_dripper.py). Route B: the pod
# boots from vLLM's OWN image (vllm/vllm-openai @ the 0.11.1 tag, set by deploy/pod/launch.py), so
# vLLM + torch==2.9.0 + CUDA arrive pre-matched -- we do NOT build a torch venv here (that is
# pod_bench.sh, torch 2.11, which collides with vLLM). We add only mineru-html (the Dripper
# API) and .[repro] (the torch-free WMB scorer) into the image's python, then run the harness.
# Dripper is WMB-only (§8) and reads raw HTML, so this builds only WMB's raw pages and folds
# from the published source -- no render. No network volume: the code snapshot comes from
# $HF_RESULTS_DATASET (code/) and the logs go back to it (logs/$RUN_NAME/) on exit.
#
# ALWAYS self-terminates via the EXIT trap, even on failure: a pod that can't clean up
# restarts and bills forever (archive, 2026-08-18).
#
# Env from launch.py: RUN_NAME, CODE_SNAPSHOT, RUN_MODE, SWEEP_ARGS, MAX_RUNTIME, HF_TOKEN,
#   HF_RESULTS_DATASET, RUNPOD_TERMINATE_API_KEY; RUNPOD_POD_ID from RunPod; optional PUBLIC_KEY for the ssh hatch.
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

# Our dockerStartCmd replaces the image's default start script, so start sshd ourselves or
# the debug hatch is dead.
if [ -n "${PUBLIC_KEY:-}" ]; then
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
    ssh-keygen -A 2>/dev/null || true
    service ssh start || /usr/sbin/sshd || echo "WARN: sshd failed to start"
fi

echo "=== $RUN_NAME -- code $CODE_SNAPSHOT ==="
mkdir -p /root/repo
curl -fsSL -H "Authorization: Bearer $HF_TOKEN" -o /root/code.tar.gz     "https://huggingface.co/datasets/$HF_RESULTS_DATASET/resolve/main/code/$CODE_SNAPSHOT"
tar xzf /root/code.tar.gz -C /root/repo
# The vllm image ships hf_transfer, so leave HF_HUB_ENABLE_HF_TRANSFER as the image sets it
# (unlike the uv-venv pods).

# Route B: install into the IMAGE's python (vLLM + torch 2.9 already present). mineru-html
# (imports as mineru_html) without its [vllm] extra never touches the vLLM/torch/CUDA stack.
# We do NOT install .[repro] -- it pins huggingface-hub==1.12.2, which the image's transformers
# 4.x refuses (<1.0). Instead install the WMB scorer's torch-free leaf deps directly (lxml and
# numpy are already in the image) and put our repo in editable --no-deps so imports resolve
# without pulling that pin. The import check fails the pod fast if a dep disturbed vLLM.
cd /root/repo
PYBIN=$(command -v python3)
pip install -q --no-cache-dir mineru-html==1.1.2
pip install -q --no-cache-dir html2text==2025.4.15 rouge_score==0.1.2 jieba==0.42.1 tldextract==5.3.2
pip install -q --no-cache-dir -e . --no-deps
"$PYBIN" -c "import torch, vllm, mineru_html; from mineru_html import MinerUHTMLConfig, MinerUHTMLGeneric, create_vllm_backend; from vendors.wmb.adapter.benchmark import WMBBenchmark; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, '| vllm', vllm.__version__, '|', torch.cuda.get_device_name(0))"

# WMB raw pages + folds from the published source: every prepare.py stage but the render
# (acquire -> collapse -> repopulate -> split), which Dripper does not read.
timeout "$MAX_RUNTIME" "$PYBIN" -c "from vendors.wmb.adapter.prepare import STAGES; [print('===', n, '===', flush=True) or run() for n, run in STAGES if n != 'blocks']"

# The harness. RUN_MODE = bench/speed/gpu_dripper.py; SWEEP_ARGS carries its args (--device cuda [--fold]).
# dripper.py prints the report to the pod log (retrieved via tools/pod_logs.py),
# like pod_bench.sh's speed run.
# shellcheck disable=SC2086 -- SWEEP_ARGS is intentionally word-split.
timeout "$MAX_RUNTIME" "$PYBIN" -u "$RUN_MODE" ${SWEEP_ARGS:-} 2>&1 | tee "$OUT/run.log"

echo "=== dripper finished cleanly ==="
