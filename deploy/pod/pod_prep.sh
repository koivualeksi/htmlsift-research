#!/usr/bin/env bash
# Data-prep pod: build each board's derived data -- blocks.jsonl, feats.jsonl, and
# labels.jsonl for boards that derive labels -- on the volume under the POD's Linux
# libxml2, from each board's platform-neutral <board>.jsonl. The render's byte output
# is libxml2-version-specific, so the derived files MUST be built on the same libxml2
# inference uses -- building them on the laptop (Windows libxml2) and shipping them is
# exactly what desynced serialize from blocks (527 vs 528). Delivered base64 in
# ENTRYPOINT_SCRIPT by launch.py --prep. Self-terminates via the EXIT trap.
#
# Env: BOARDS (space-separated), RUN_NAME, CODE_SNAPSHOT, MAX_RUNTIME,
#   RUNPOD_TERMINATE_API_KEY, RUNPOD_POD_ID.
set -euo pipefail

VOL=/workspace
OUT="$VOL/outputs/$RUN_NAME"
mkdir -p "$OUT"
date -u +'prep started %Y-%m-%dT%H:%M:%SZ' > "$OUT/STARTED"
exec > >(tee -a "$OUT/bootstrap.log") 2>&1
START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cleanup() {
    status=$?
    set +e +u
    printf '{"run_name":"%s","stage":"prep","boards":"%s","code_snapshot":"%s","exit_code":%d,"started":"%s","ended":"%s"}\n' \
        "${RUN_NAME:-}" "${BOARDS:-}" "${CODE_SNAPSHOT:-}" "$status" "${START_TS:-}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/run_meta.json" || true
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

echo "=== $RUN_NAME (data prep: $BOARDS) -- code $CODE_SNAPSHOT ==="
test -f "$VOL/code/$CODE_SNAPSHOT" || { echo "ERROR: code snapshot missing"; exit 1; }
for board in $BOARDS; do
    test -f "$VOL/data/$board/$board.jsonl" \
        || { echo "ERROR: $VOL/data/$board/$board.jsonl missing -- run tools/upload_data.py --board $board first"; exit 1; }
done

mkdir -p /root/repo
tar xzf "$VOL/code/$CODE_SNAPSHOT" -C /root/repo
for board in $BOARDS; do
    ln -sfn "$VOL/data/$board" "/root/repo/vendors/$board/data"
done
cd /root/repo

# lxml + numpy only -- the render/label/feature chain needs no torch. .[repro] carries
# lxml, jieba and rouge_score, covering every board's blocks.py and labels.py.
pip install -q --no-cache-dir uv
uv venv --clear /root/venv --python 3.13
PYBIN=/root/venv/bin/python
uv pip install -q --python "$PYBIN" --no-cache ".[repro]" numpy==2.2.6
"$PYBIN" -c "import lxml.etree as e; print('libxml2', e.LIBXML_VERSION)"

for board in $BOARDS; do
    echo "--- prep $board ---"
    timeout "$MAX_RUNTIME" "$PYBIN" "vendors/$board/adapter/blocks.py" --feats ABC
    # wcxb/daniel derive labels from the render (blocks.jsonl); the frozen ceiling
    # assert in labels.py is the cross-platform render gate. wmb has no labels.py --
    # its labels are DOM provenance, built inside blocks.py.
    if [ -f "vendors/$board/adapter/labels.py" ]; then
        timeout "$MAX_RUNTIME" "$PYBIN" "vendors/$board/adapter/labels.py"
    fi
done

# wmb-only: prove serialize's render agrees with the freshly built block counts over
# val+test -- the exact invariant that failed cross-platform.
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

echo "=== prep finished cleanly: derived files built on the volume for: $BOARDS ==="
