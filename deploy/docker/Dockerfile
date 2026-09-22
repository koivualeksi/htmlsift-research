# Reproducible CPU environment for the §7 speed+F1 comparison (bench/speed/cpu_compare.py).
#
# The point is a pinned stack. Python 3.13 + lxml 6.1.1 are the render-gate versions (§4;
# 3.12 rendered 1/730 val pages one line off), so this locks BOTH the timing and the F1
# column, and OMP/BLAS threads are fixed so a run is reproducible on a given CPU. Absolute
# ms still varies by host CPU -- publish the RATIO and the CPU string the report prints,
# never an absolute (§7).
#
# Data and keeper ckpts are NOT baked (data/ is gitignored, §2): mount them at run time.
# Build from the repo root so the context is the whole tree (.dockerignore drops data/):
#
#   docker build -f deploy/docker/Dockerfile -t htmlsift-bench .
#   docker run --rm --cpus=2 --cpuset-cpus=0,1 \
#       -v "$PWD/vendors/wmb/data:/repo/vendors/wmb/data" \
#       -v "$PWD/data/runs/wcxb/ckpt/97m-6_s0.pt:/repo/ckpt.pt" \
#       htmlsift-bench --ckpt /repo/ckpt.pt --fold test --threads 2
#
# --cpuset-cpus pins which cores (kills scheduler-migration jitter); --cpus caps how many.
# The vCPU count is part of the claim, not just hygiene -- our encoder is multithreaded and
# trafilatura is not, so the ratio grows in our favour with more cores. 2 is the conservative
# pinned choice.
FROM python:3.13-slim-bookworm

ENV OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1 HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /repo
COPY pyproject.toml ./
COPY core core
COPY vendors vendors
COPY bench bench
COPY trainers trainers
COPY export export

# CPU torch from the cpu index (PyPI's default wheel pulls CUDA on Linux); a separate layer so
# a rebuild that only touches the line below reuses this ~200 MB download. torch==2.11.0+cpu
# then satisfies the [train] `torch==2.11.0` pin (PEP 440 ignores the local +cpu label), exactly
# as pod_train.sh relies on.
RUN pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
# The model stack the numbers were validated under: [train] (transformers) + the scorer ([repro])
# + the third-party extractors ([bench]). -e so imports resolve to /repo and the mounted
# vendors/wmb/data is read in place.
RUN pip install --no-cache-dir -e ".[repro,bench,train]"

ENTRYPOINT ["python", "bench/speed/cpu_compare.py"]
