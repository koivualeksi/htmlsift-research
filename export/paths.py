"""The mini artifact's graph filenames, in one place: the builder, the bench runtime, and
the publisher all read the names from here, so a rename touches one line."""
from pathlib import Path

TABLE = "mini-table-int8.onnx"
HEAD = "mini-head-fABC.onnx"


def mini_paths(root):
    """(table_path, head_path) under a local dir `root` (e.g. data/onnx)."""
    root = Path(root)
    return root / TABLE, root / HEAD
