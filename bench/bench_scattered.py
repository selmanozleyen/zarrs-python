"""Scattered-row gather on a real plate: the negative control.

`dataset[rows]` on a backed CSR reads a few elements out of each touched shard,
so almost every chunk item is a *partial* read. The fetch pool only handles
whole chunks, so this should show no difference between the two wheels. Run it
to establish that, rather than assuming it.

Usage: bench_scattered.py <label> <fetch_threads|-> <rows_per_batch>
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import zarr

import zarrs  # noqa: F401
from anndata._core.sparse_dataset import sparse_dataset

LABEL, FETCH_ARG, NROWS = sys.argv[1], sys.argv[2], int(sys.argv[3])
FETCH = None if FETCH_ARG == "-" else int(FETCH_ARG)
PLATE = "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_0/X"
REPS, WARMUP = 8, 2

cfg: dict = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
if FETCH is not None:
    cfg["codec_pipeline.fetch_threads"] = FETCH
zarr.config.set(cfg)

d = sparse_dataset(zarr.open(PLATE, mode="r"))
n_obs = d.shape[0]
rng = np.random.default_rng(0)

ts, nnz = [], 0
for i in range(WARMUP + REPS):
    rows = np.sort(rng.choice(n_obs, NROWS, replace=False))
    t = time.perf_counter()
    out = d[rows]
    dt = time.perf_counter() - t
    if i >= WARMUP:
        ts.append(dt)
        nnz += int(out.nnz)

med = float(np.median(ts)) * 1e3
res = {
    "label": LABEL,
    "mode": "scattered_rows",
    "fetch_threads": FETCH,
    "rows": NROWS,
    "median_ms": round(med, 1),
    "p10_ms": round(float(np.percentile(ts, 10)) * 1e3, 1),
    "p90_ms": round(float(np.percentile(ts, 90)) * 1e3, 1),
    "us_per_row": round(med * 1000.0 / NROWS, 1),
    "mean_nnz": nnz // len(ts),
    "host": os.uname().nodename,
}
print(json.dumps(res))
print(
    f"{LABEL:<18} scattered fetch={str(FETCH):<5} rows={NROWS:<5} "
    f"median={med:>9.1f} ms  {res['us_per_row']:>7.1f} us/row  nnz={res['mean_nnz']:,}",
    file=sys.stderr,
)
