#!/bin/sh
# The published CPU speed + RAM runs (§7): the mini through the htmlsift package against
# trafilatura, readability and resiliparse on WMB test545, in two containers of one image,
# 2 vCPU / 4 GB and 1 vCPU / 2 GB. onnxruntime runs single-threaded in both, as trafilatura
# does. Run from the repo root on Linux, after
#
#   docker build -f deploy/docker/Dockerfile -t htmlsift-bench .
#   python tools/release.py mini          # the bundle, under data/bundles/mini
#
# WMB data and the bundle mount read-only; nothing is fetched (HF_HUB_OFFLINE). The reports
# land in results/.
set -eu

run() {  # cpus cpuset memory report
    docker run --rm --cpus="$1" --cpuset-cpus="$2" --memory="$3" --memory-swap="$3" \
        -e HF_HUB_OFFLINE=1 -e OMP_NUM_THREADS=1 \
        -v "$PWD/vendors/wmb/data:/repo/vendors/wmb/data:ro" \
        -v "$PWD/data/bundles/mini:/bundle:ro" \
        -v "$PWD/results:/out" \
        htmlsift-bench --bundle /bundle --fold test --threads 1 --memory --out "/out/$4"
}

run 2 0,1 4g speed-wmb.md
run 1 0 2g speed-wmb-1vcpu.md
