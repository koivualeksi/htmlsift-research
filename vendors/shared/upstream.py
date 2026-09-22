"""Load a vendored upstream/*.py by file path. upstream/ deliberately carries no
__init__.py (their code, unmodified -- CLAUDE.md §9), so it cannot be imported as a
package; it is loaded verbatim under a board-prefixed name."""
import importlib.util
from pathlib import Path


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
