"""Assemble and publish the shipped model *bundles* to the public release repo
(koivualeksi/htmlsift-release).

Torch-side and human-run: the production package never runs this, it downloads the pinned
bundle and (for mini) runs onnxruntime. A "bundle" is exactly the directory the package's
runtime loads -- the layout htmlsift/_artifacts.py hard-codes as <mode>/<file>:

    mini/  mini-table-int8.onnx  mini-head-fABC.onnx  tokenizer.json  manifest.json
    base/  config.json           weights.pt           tokenizer.json  manifest.json

    python tools/release.py mini            # build + gate the mini bundle under data/bundles/mini
    python tools/release.py mini --push     # + upload it to mini/
    python tools/release.py base            # bake + self-check the base bundle
    python tools/release.py base --push     # + upload it to base/

The manifest schema is intentionally duplicated here rather than imported from the package:
the two repos stay decoupled (research imports nothing from htmlsift). Keep the fields in sync
with htmlsift.Manifest by hand -- they are few and stable (mode, feats, hidden, window, cap,
zscore, source). `source` stamps which keeper + research commit a bundle came from, so a
published htmlsift version is traceable back to a trained model; the package ignores it.

mini (torch-free int8 table + BiGRU head) is gated by export/gate.py before any push -- the
built graphs must reproduce torch bit-for-bit. base ships the native-torch flagship, baked
into a self-contained bundle so the runtime never pulls granite. Not a runtime import.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, upload_folder
from transformers import AutoConfig

from core.features import feature_dim
from core.loader import resolve_ckpt
from core.model import build_encoder, build_head
from export.build import build_mini
from export.gate import parity

RELEASE_REPO = "koivualeksi/htmlsift-release"
MINI_ARM, MINI_SEED = "311m-table-bigru-fABC", 1   # ship the best-val seed (val732 0.8623)
BASE_ARM, BASE_SEED = "311m-10", 1                  # seeds within noise; s1 matches the mini
WINDOW = 8192                                       # training window (arms.WINDOW); manifest schema
ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _provenance(keeper_ref):
    """{keeper, research_commit} stamped into the manifest -- the promotion record. Never
    fails a build: an unknown commit (e.g. not a git checkout) is recorded as None."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        sha = None
    return {"keeper": keeper_ref, "research_commit": sha}


def _push_bundle(local_dir, mode):
    """Upload the whole bundle dir to <mode>/ in the release repo (one commit). upload_folder
    sends every file under local_dir, so local_dir must hold exactly the bundle -- cmd_* wipe
    and rebuild it first."""
    arm = f"{MINI_ARM} s{MINI_SEED}" if mode == "mini" else f"{BASE_ARM} s{BASE_SEED}"
    upload_folder(folder_path=str(local_dir), path_in_repo=mode, repo_id=RELEASE_REPO,
                  repo_type="model", commit_message=f"{mode} bundle: {arm}")
    print(f"pushed {mode}/ bundle", flush=True)


def cmd_mini(push):
    keeper_ref = f"wmb/ckpt/{MINI_ARM}_s{MINI_SEED}.pt"
    ck = resolve_ckpt(keeper_ref, torch.device("cpu"))
    assert ck.get("kind") == "table" and not ck.get("int8") and ck.get("feats") == "ABC"

    out = ROOT / "data" / "bundles" / "mini"          # data/ gitignored
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    table_p, head_p, emb, head, d_in = build_mini(ck, out)   # writes the 2 onnx into out/
    print(f"built {table_p.name} {table_p.stat().st_size / 1e6:.0f} MB | "
          f"{head_p.name} {head_p.stat().st_size / 1e6:.1f} MB (d_in {d_in})", flush=True)

    flips, n = parity(table_p, head_p, emb, head, d_in)      # refuse to ship on drift
    if flips:
        raise SystemExit(f"ONNX parity gate FAILED: {flips}/{n} flips -- built graph drifts from torch")
    print(f"gate OK: 0/{n} flips over random inputs", flush=True)

    shutil.copyfile(hf_hub_download(ck["model"], "tokenizer.json"), out / "tokenizer.json")
    _idx, mean, std = ck["zscore"]                    # keeper stores (idx, mean, std)
    _write_json(out / "manifest.json", {
        "mode": "mini",
        "feats": ck["feats"],                         # "ABC"
        "window": WINDOW,                             # unused by mini (no windowing); schema
        "cap": 0,
        "zscore": {"mean": [float(x) for x in mean], "std": [float(x) for x in std]},
        "source": _provenance(keeper_ref),
    })
    _report(out)
    if push:
        _push_bundle(out, "mini")


def cmd_base(push):
    keeper_ref = f"wmb/ckpt/{BASE_ARM}_s{BASE_SEED}.pt"
    ck = resolve_ckpt(keeper_ref, torch.device("cpu"))
    assert ck.get("kind") != "table", "base wants an encoder keeper, not a table keeper"

    out = ROOT / "data" / "bundles" / "base"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)

    n = ck["layers"]
    config = AutoConfig.from_pretrained(ck["model"])
    config.num_hidden_layers = n                      # truncate to the trained depth
    if getattr(config, "layer_types", None):          # ModernBERT: one entry per layer --
        config.layer_types = config.layer_types[:n]   # keep in sync or validate() fails
    config.save_pretrained(out)                        # writes config.json

    torch.save({"encoder": ck["encoder"], "head_state": ck["head_state"]}, out / "weights.pt")
    shutil.copyfile(hf_hub_download(ck["model"], "tokenizer.json"), out / "tokenizer.json")
    _write_json(out / "manifest.json", {
        "mode": "base",
        "feats": ck.get("feats"),                     # None for the text-only encoder
        "hidden": ck.get("hidden", 256),              # BiGRU head hidden
        "window": WINDOW,
        "cap": ck.get("cap", 0),
        "source": _provenance(keeper_ref),
    })

    # self-check: the saved config.json must re-read to the trained depth, and the runtime
    # arch must strict-load the keeper's weights (catches a bad bake -- e.g. layer_types drift).
    saved = AutoConfig.from_pretrained(out)
    assert saved.num_hidden_layers == n, f"config.json depth {saved.num_hidden_layers} != {n}"
    if getattr(saved, "layer_types", None):
        assert len(saved.layer_types) == n, "config.json layer_types out of sync with num_hidden_layers"
    enc = build_encoder(ck["model"], n)
    enc.load_state_dict(ck["encoder"])                 # strict: keys must match exactly
    head = build_head("bigru", d_in=enc.config.hidden_size + feature_dim(ck.get("feats")),
                      hidden=ck.get("hidden", 256))
    head.load_state_dict(ck["head_state"])
    print("self-check OK: config re-reads, encoder + head strict-load", flush=True)

    _report(out)
    if push:
        _push_bundle(out, "base")


def _report(out):
    print(f"assembled {out.parent.name}/{out.name} bundle:", flush=True)
    for p in sorted(out.iterdir()):
        print(f"  {p.name:22} {p.stat().st_size / 1e6:8.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", choices=["mini", "base"])
    ap.add_argument("--push", action="store_true", help=f"upload the bundle to {RELEASE_REPO}")
    args = ap.parse_args()
    if args.push:
        load_dotenv(ROOT / ".env")
    (cmd_mini if args.artifact == "mini" else cmd_base)(args.push)


if __name__ == "__main__":
    main()
