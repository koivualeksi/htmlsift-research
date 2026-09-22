"""
End-to-end WCXB data preparation: acquire -> collapse -> blocks -> labels.
Turns the published snapshot into the training data and the label ceiling
(data/wcxb.jsonl, data/splits.json, data/blocks.jsonl, data/labels.jsonl). Each
stage asserts its own invariant and raises on failure, stopping the chain.

    python vendors/wcxb/adapter/prepare.py

No split stage, unlike WMB: WCXB's dev/test split is used verbatim (dev = train,
test = eval board, never labeled), so there is no fold to carve. blocks renders
with render_hidden=True (WCXB's reference is captured from a hydrated DOM); labels
are dev-only. Individual stages stay runnable for partial rebuilds. acquire pulls
the snapshot into data/ and keeps it -- their evaluate.py reads the tree -- so a
re-run re-verifies against the cache rather than re-downloading.
"""

from vendors.wcxb.adapter import (
    acquire,
    blocks,
    collapse,
    labels
)

STAGES = [
    ("acquire", acquire.acquire),
    ("collapse", collapse.main),
    ("blocks", blocks.main),
    ("labels", labels.main),
]


def main():
    for name, run in STAGES:
        print(f"\n=== {name} ===", flush=True)
        run()
    print("\nwcxb data prep complete: data/wcxb.jsonl, data/splits.json, "
          "data/blocks.jsonl, data/labels.jsonl")


if __name__ == "__main__":
    main()
