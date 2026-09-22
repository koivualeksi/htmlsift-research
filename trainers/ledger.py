"""Result ledger: arm identity, resume, and HF transport for the trainers.

One Ledger owns a (local jsonl, remote HF dataset repo, remote path) triple -- plus the
keeper-ckpt model repo (push_ckpt), distinct from the dataset. resume seeds the skip-set
from the remote, append writes records locally, merge_push ships them. It is a no-op when there is no repo, so a local no-push smoke and a
pod run share one code path. Sharding across parallel pods is the caller's choice -- one
explicit shard per pod writes a distinct file (sharded_path) and never races;
merge_push only narrows the window for the accidental same-shard case.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.errors import EntryNotFoundError


def arm_key(model, layers, head, feats, cap, seed):
    return (model, layers, head, feats or "none", cap, seed)


def record_key(rec):
    base = arm_key(rec["model"], rec["layers"], rec["head"], rec["feats"],
                   rec.get("cap", 0), rec["seed"])
    if "train_limit" in rec:
        base = base + (rec["train_limit"],)
    if "holdout" in rec:
        base = base + (rec["holdout"],)
    return base


def resolve_repo(push_arg):
    """--push's three states: None (absent) -> no push; "" (bare --push) -> the
    $HF_RESULTS_DATASET output repo; else the given repo. A bare --push with no env
    set is a config error, not a silent local run."""
    if push_arg is None:
        return None
    if push_arg == "":
        repo = os.environ.get("HF_RESULTS_DATASET")
        if not repo:
            sys.exit("--push needs a repo: pass one or set HF_RESULTS_DATASET")
        return repo
    return push_arg


def resolve_models_repo(push_arg):
    """The keeper-ckpt model repo (distinct from the results dataset), from
    $HF_MODELS_REPO -- resolved only when --push is active. None otherwise; the presence
    check is at push time, so a --push run without --keep needs no model repo."""
    return os.environ.get("HF_MODELS_REPO") if push_arg is not None else None


def sharded_path(base, shard):
    """base.<shard>.jsonl for a per-pod shard; base unchanged when unsharded."""
    return base.replace(".jsonl", f".{shard}.jsonl") if shard else base


def _read_jsonl(path):
    return [json.loads(l) for l in
            Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


class Ledger:
    def __init__(self, local, repo, path, models_repo=None, key_fn=record_key):
        self.local = Path(local)
        self.repo = repo
        self.path = path
        self.models_repo = models_repo   # keeper-ckpt model repo (push_ckpt); distinct from repo
        self.key_fn = key_fn             # record -> dedup/skip key; default the training arm-key
        self.done = set()                # already-done arm keys; populated by resume()

    def resume(self):
        """Seed self.done with the already-done arm keys. With a repo, pull the remote
        jsonl first (a fresh pod has none); the set is a startup snapshot, so a late pod
        may redo an arm -- the union on push keeps both, no data loss. No repo -> no
        resume. Returns self.done for the caller's convenience."""
        if self.repo is None:
            return self.done
        if not self.local.exists():
            try:
                cached = hf_hub_download(self.repo, self.path, repo_type="dataset")
                self.local.write_bytes(Path(cached).read_bytes())
            except EntryNotFoundError:
                return self.done
        self.done = {self.key_fn(r) for r in _read_jsonl(self.local)}
        return self.done

    def is_done(self, key):
        return key in self.done

    def all_done(self, keys):
        """True if every arm key in keys is already recorded -- the group-skip guard."""
        return all(k in self.done for k in keys)

    def push_ckpt(self, local, path_in_repo, msg):
        """Upload a keeper ckpt to the model repo (repo_type='model'), not the results
        dataset. No-op without --push (self.repo None); errors if --push is on but no model
        repo was resolved (a kept ckpt with nowhere to go)."""
        if self.repo is None:
            return
        if self.models_repo is None:
            sys.exit("--push with a kept ckpt needs a model repo: set HF_MODELS_REPO")
        upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                    repo_id=self.models_repo, repo_type="model", commit_message=msg)

    def append(self, rec):
        rec.setdefault("ts", datetime.now(timezone.utc).isoformat())
        with open(self.local, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def merge_push(self, msg):
        """Reload the remote shard and union it with local (latest-ts per arm key)
        before uploading, so a push never drops a record a same-shard pod appended
        since our last read."""
        if self.repo is None:
            return
        remote = []
        try:
            cached = hf_hub_download(self.repo, self.path, repo_type="dataset")
            remote = _read_jsonl(cached)
        except EntryNotFoundError:
            pass
        latest = {}
        for r in remote + _read_jsonl(self.local):
            k = self.key_fn(r)
            if k not in latest or r["ts"] > latest[k]["ts"]:
                latest[k] = r
        merged = sorted(latest.values(), key=lambda r: r["ts"])
        self.local.write_text("".join(json.dumps(r) + "\n" for r in merged),
                              encoding="utf-8")
        upload_file(path_or_fileobj=str(self.local), path_in_repo=self.path,
                    repo_id=self.repo, repo_type="dataset", commit_message=msg)
