"""
End-to-end WMB data preparation: acquire -> collapse -> repopulate -> split
-> blocks. Turns the published source into the training data
(data/wmb.jsonl, data/splits.json, data/blocks.jsonl). Each stage asserts its
own invariant and raises on failure, stopping the chain.

    python vendors/wmb/adapter/prepare.py

Individual stages stay runnable on their own for partial rebuilds; this runs the
full path from the published source. acquire downloads ~1.3 GB and collapse then
deletes the two source files, so a full re-run re-downloads -- to rebuild only
the blocks off an existing wmb.jsonl, run blocks.py directly.
"""

from vendors.wmb.adapter import (
    acquire,
    blocks,
    collapse,
    repopulate,
    split
)

STAGES = [
    ("acquire", acquire.acquire),
    ("collapse", collapse.main),
    ("repopulate", repopulate.main),
    ("split", split.main),
    ("blocks", blocks.main),
]


def main():
    for name, run in STAGES:
        print(f"\n=== {name} ===", flush=True)
        run()
    print("\nwmb data prep complete: data/wmb.jsonl, data/splits.json, data/blocks.jsonl")


if __name__ == "__main__":
    main()
