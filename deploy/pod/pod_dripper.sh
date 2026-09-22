#!/usr/bin/env bash
# Pod-side bootstrap for the Dripper GPU comparison (bench/speed/gpu_dripper.py). Route B: the pod
# boots from vLLM's OWN image (vllm/vllm-openai @ the 0.11.1 tag, set by deploy/pod/launch.py), so
# vLLM + torch==2.9.0 + CUDA arrive pre-matched -- we do NOT build a torch venv here (that is
# pod_bench.sh, torch 2.11, which collides with vLLM). We add only mineru-html (the Dripper
# API) and .[repro] (the torch-free WMB scorer) into the image's python, then run the harness.
# Dripper is WMB-only (§8), so this mounts just data/wmb.
#
# ALWAYS self-terminates via the EXIT trap, even on failure: a pod that can't clean up
# restarts and bills forever (archive, 2026-08-18).
#
# Env from launch.py: RUN_NAME, CODE_SNAPSHOT, RUN_MODE, SWEEP_ARGS, MAX_RUNTIME, HF_TOKEN,
#   RUNPOD_TERMINATE_API_KEY; RUNPOD_POD_ID from RunPod; optional PUBLIC_KEY for the ssh hatch.
set -euo pipefail

VOL=/workspace
OUT="$VOL/outputs/$RUN_NAME"
mkdir -p "$OUT"
date -u +'entrypoint started %Y-%m-%dT%H:%M:%SZ' > "$OUT/STARTED"
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
test -f "$VOL/code/$CODE_SNAPSHOT" || { echo "ERROR: code snapshot missing -- launch.py uploads it"; exit 1; }
mkdir -p /root/repo
tar xzf "$VOL/code/$CODE_SNAPSHOT" -C /root/repo

# Dripper is WMB-only: mount data/wmb (raw pages + gold for the scorer). Unlike pod_bench, no
# blocks.jsonl is needed -- Dripper reads raw HTML, not our lxml render.
test -d "$VOL/data/wmb" || { echo "ERROR: no data/wmb on the volume -- upload_data.py --board wmb first"; exit 1; }
ln -sfn "$VOL/data/wmb" /root/repo/vendors/wmb/data
echo "wmb data mounted"

# Dripper weights cache on the volume (survive pod death). The vllm image ships hf_transfer,
# so leave HF_HUB_ENABLE_HF_TRANSFER as the image sets it (unlike the uv-venv pods).
export HF_HOME="$VOL/hf"
mkdir -p "$HF_HOME"

# Route B: install into the IMAGE's python (vLLM + torch 2.9 already present). mineru-html
# (imports as mineru_html) without its [vllm] extra never touches the vLLM/torch/CUDA stack.
# We do NOT install .[repro] -- it pins huggingface-hub==1.12.2, which the image's transformers
# 4.x refuses (<1.0). Instead install the WMB scorer's torch-free leaf deps directly (lxml and
# numpy are already in the image) and put our repo in editable --no-deps so imports resolve
# without pulling that pin. The import check fails the pod fast if a dep disturbed vLLM.
cd /root/repo
PYBIN=$(command -v python3)
pip install -q --no-cache-dir mineru-html==1.1.2
pip install -q --no-cache-dir html2text==2025.4.15 rouge_score==0.1.2 jieba==0.42.1
pip install -q --no-cache-dir -e . --no-deps
"$PYBIN" -c "import torch, vllm, mineru_html; from mineru_html import MinerUHTMLConfig, MinerUHTMLGeneric, create_vllm_backend; from vendors.wmb.adapter.benchmark import WMBBenchmark; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, '| vllm', vllm.__version__, '|', torch.cuda.get_device_name(0))"

# The harness. RUN_MODE = bench/speed/gpu_dripper.py; SWEEP_ARGS carries its args (--device cuda [--fold]).
# dripper.py prints the report to the pod log (retrieved via tools/upload_data.py --stage logs),
# like pod_bench.sh's speed run.
# shellcheck disable=SC2086 -- SWEEP_ARGS is intentionally word-split.
timeout "$MAX_RUNTIME" "$PYBIN" -u "$RUN_MODE" ${SWEEP_ARGS:-} 2>&1 | tee "$OUT/run.log"

echo "=== dripper finished cleanly ==="
