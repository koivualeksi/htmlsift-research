# vendors/

One directory per benchmark. Each holds three things.

| | |
|---|---|
| `upstream/` | their code, vendored verbatim. Diff it against the commit named in that board's `README.md` and the diff must come back empty. Their licence, their headers, no edits. |
| `adapter/` | our code. Fits their data, format and conventions to this repo's pipeline. GPL-3.0-or-later, like the rest of the repo. |
| `data/` | the corpus landing site. Gitignored, created by the acquire step, never committed. |

Each board's `README.md` records provenance: upstream repository, path, commit,
licence, and the checksum of every file the acquire step downloads. A stranger
cloning this repo gets the bytes we measured, or a raised error.

Acquisition always starts from the published source. No step reads a
pre-existing local copy, because a new user does not have one.

`core/` holds nothing board-specific and names no benchmark.
