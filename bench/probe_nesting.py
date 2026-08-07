"""How many sharding levels does the plate actually have?

`prefetch_subchunk_indexes` and the inner-index cache only do anything when an
inner chunk is itself a shard. If the array is flat-sharded, eager and lazy are
the same code path and comparing them measures nothing.
"""

from __future__ import annotations

import json
import sys

BASE = "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_0/X"


def levels(codecs, depth=0):
    found = 0
    for c in codecs:
        if "sharding" in str(c.get("name", "")):
            cfg = c.get("configuration", {})
            print(f"    level {depth}: inner chunk {cfg.get('chunk_shape')}")
            found = 1 + levels(cfg.get("codecs", []), depth + 1)
    return found


for name in ("data", "indices", "indptr"):
    meta = json.load(open(f"{BASE}/{name}/zarr.json"))
    print(f"  {name}: outer chunk {meta.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape')}")
    n = levels(meta.get("codecs", []))
    print(f"    -> {n} sharding level(s)")
    if n < 2:
        print("    -> NOT nested: inner-index cache is inactive, eager == lazy")
sys.exit(0)
