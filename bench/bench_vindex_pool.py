"""Random-row gather across the whole collection, through the fetch pool.

`dataset[rows]` on a backed CSR reads a few elements out of each touched
shard, so almost every chunk item is a partial read -- exactly what
`read_plan` describes and what the fetch pool issues together.

Rows are drawn uniformly at random and left **unsorted**, across **every**
dataset in the collection rather than one plate. Sorting rows lets the
underlying reads come out roughly sequential, which is the case Lustre is
already good at and which a fetch pool cannot improve; the workload this is
meant to serve -- minibatch sampling -- has no such order. Spreading over all
datasets keeps the working set far larger than page cache, so each rep is a
genuine cold read rather than a measurement of RAM.

Two knobs, both read from the environment once per process, so comparing
settings means separate runs:
  ZARRS_PYTHON_FETCH_THREADS      pool size; 0 disables planning entirely
  ZARRS_PYTHON_FILE_HANDLE_CACHE  open file handles kept; 0 is upstream default

Each run prints a checksum of what it read beside the timing. The checksum
must not move when the knobs do -- that is the correctness check, and it makes
the first runs of a sweep double as the smoke test.

The seed matters more than it looks. With a fixed seed every process reads
the *same* rows, so the first run pays the cold reads and every run after it
is served from page cache -- on a node with more RAM than the working set,
that turns a storage benchmark into a memory benchmark and every knob looks
inert. The seed therefore defaults to fresh entropy per run.

Comparing two builds needs the opposite: pass both the *same* explicit seed
so they read identical rows, and the checksums then have to agree. Whichever
runs second still benefits from the first one's cache, so alternate which
goes first across pairs and give each pair a fresh random seed -- that is
what bench_pair.sh does.

Usage: bench_vindex_pool.py <label> <rows_per_batch> [reps] [warmup] [seed]
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import time

import numpy as np
import zarr

import zarrs  # noqa: F401
from anndata._core.sparse_dataset import sparse_dataset

LABEL = sys.argv[1]
NROWS = int(sys.argv[2])
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
WARMUP = int(sys.argv[4]) if len(sys.argv) > 4 else 2
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else secrets.randbits(63)

COLLECTION = "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr"
FETCH = os.environ.get("ZARRS_PYTHON_FETCH_THREADS", "0")
FDCACHE = os.environ.get("ZARRS_PYTHON_FILE_HANDLE_CACHE", "0")

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})

root = zarr.open_group(COLLECTION, mode="r")
names = sorted(k for k in root.keys() if k.startswith("dataset_"))
datasets = [sparse_dataset(root[f"{name}/X"]) for name in names]
n_obs = [d.shape[0] for d in datasets]
rng = np.random.default_rng(SEED)

ts: list[float] = []
digest = hashlib.sha256()
nnz = 0
for i in range(WARMUP + REPS):
    # Uniform over the whole collection, then split per dataset. Rows stay
    # unsorted: sorting would hand the storage a near-sequential access
    # pattern that this workload never has.
    which = rng.integers(0, len(datasets), NROWS)
    picks = [
        rng.integers(0, n_obs[j], int((which == j).sum())) for j in range(len(datasets))
    ]
    t = time.perf_counter()
    outs = [d[rows] for d, rows in zip(datasets, picks) if len(rows)]
    dt = time.perf_counter() - t
    if i >= WARMUP:
        ts.append(dt)
        nnz += sum(int(o.nnz) for o in outs)
        # Same bytes regardless of how they were fetched.
        for o in outs:
            digest.update(np.ascontiguousarray(o.data))
            digest.update(np.ascontiguousarray(o.indices))
            digest.update(np.ascontiguousarray(o.indptr))

med = float(np.median(ts)) * 1e3
res = {
    "label": LABEL,
    "fetch_threads": int(FETCH),
    "fd_cache": int(FDCACHE),
    "seed": SEED,
    "datasets": names,
    "rows": NROWS,
    "reps": REPS,
    "median_ms": round(med, 1),
    "p10_ms": round(float(np.percentile(ts, 10)) * 1e3, 1),
    "p90_ms": round(float(np.percentile(ts, 90)) * 1e3, 1),
    "us_per_row": round(med * 1000.0 / NROWS, 1),
    "rows_per_s": round(NROWS / (med / 1e3), 1),
    "mean_nnz": nnz // len(ts),
    "checksum": digest.hexdigest()[:16],
    "host": os.uname().nodename,
}
print(json.dumps(res))
print(
    f"{LABEL:<14} fetch={FETCH:<5} fd={FDCACHE:<5} seed={SEED:<4} rows={NROWS:<6} "
    f"median={med:>9.1f} ms  {res['rows_per_s']:>9.1f} rows/s  "
    f"checksum={res['checksum']}",
    file=sys.stderr,
)
