"""Cold whole-shard reads from a real plate. Run once per installed wheel.

Every rep reads a *different* shard-aligned window, so nothing is served from
the page cache -- the earlier run measured a 16 MB array it had just written,
which had no latency left to hide and so showed nothing.

Shard-aligned means every chunk item is a whole chunk, which is the only path
the fetch pool touches. `shards_per_read` sets how many reads are in flight,
which is the variable that matters: main caps at the rayon width, this branch
caps at fetch_threads.

Usage: bench_cold_shards.py <label> <fetch_threads|-> <shards_per_read>
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import zarr

import zarrs  # noqa: F401

LABEL, FETCH_ARG, SPR = sys.argv[1], sys.argv[2], int(sys.argv[3])
FETCH = None if FETCH_ARG == "-" else int(FETCH_ARG)
PATH = "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_0/X/indices"
REPS = 3

cfg: dict = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
if FETCH is not None:
    cfg["codec_pipeline.fetch_threads"] = FETCH
zarr.config.set(cfg)

probe = zarr.open_array(PATH, mode="r")
shard = int(probe.shards[0])
n_shards = probe.shape[0] // shard
itemsize = probe.dtype.itemsize

# Disjoint windows, one per rep, so each read is cold.
windows = []
for r in range(REPS):
    first = r * SPR
    if first + SPR > n_shards:
        break
    windows.append((first * shard, (first + SPR) * shard))

ts, checks = [], []
for start, stop in windows:
    a = zarr.open_array(PATH, mode="r")  # fresh pipeline per read
    t = time.perf_counter()
    buf = a[start:stop]
    ts.append(time.perf_counter() - t)
    checks.append(int(buf[:: max(1, buf.size // 512)].astype(np.int64).sum()))

gib = SPR * shard * itemsize / 1024**3
med = float(np.median(ts)) * 1e3
out = {
    "label": LABEL,
    "fetch_threads": FETCH,
    "shards_per_read": SPR,
    "gib_per_read": round(gib, 2),
    "median_ms": round(med, 1),
    "all_ms": [round(t * 1e3, 1) for t in ts],
    "gib_per_s": round(gib / (med / 1000.0), 2),
    "checksums": checks,
    "host": os.uname().nodename,
}
print(json.dumps(out))
print(
    f"{LABEL:<18} fetch={str(FETCH):<5} shards={SPR:<3} "
    f"{gib:>5.2f} GiB  median={med:>8.1f} ms  {out['gib_per_s']:>6.2f} GiB/s",
    file=sys.stderr,
)
