"""Download IO shared by the board acquire/collapse scripts: integrity hashing -- a
per-file sha256 (WMB pins each file) and a manifest sha256 over a set of files (WCXB/DAnIEL
pin a whole tree by the sha256 of its sorted 'sha256  relpath' lines) -- and read_html for
wild HTML that is not guaranteed UTF-8. The caller owns which paths go into the manifest --
scoping it to the download, not the derived outputs, is a board-specific choice made at the
call site.
"""
import hashlib
from pathlib import Path


def sha256(path) -> str:                 # str from hf_hub_download, Path from the others
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_sha256(paths, data_dir) -> str:
    lines = [f"{sha256(p)}  {p.relative_to(data_dir).as_posix()}" for p in sorted(paths)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def read_html(path: Path) -> tuple[str, bool]:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8"), False
    except UnicodeDecodeError:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        return (str(best) if best is not None else raw.decode("utf-8", "replace")), True
