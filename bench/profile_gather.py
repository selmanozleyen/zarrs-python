"""Where does one CSR gather actually spend its time?

Splits a gather into the phases that can each be fixed independently:

  indexing   zarr working out which chunks and offsets are involved
  prep       describing that to Rust -- make_chunk_info, including split_1d_runs
  rust       retrieve_chunks_and_apply_index: build decoders, plan, fetch, decode
  assembly   anndata turning the raw data/indices/indptr into a CSR matrix

`rust` is opaque here by design; if it dominates, the next step is instrumenting
inside it rather than guessing. cProfile output follows for the Python side.

Usage: profile_gather.py <rows> [reps]
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from collections import defaultdict

import numpy as np
import zarr

import zarrs
from anndata._core.sparse_dataset import sparse_dataset
from zarrs import pipeline as zp
from zarrs import utils as zu

COLLECTION = "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr"
NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 9192
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

spent: dict[str, float] = defaultdict(float)
calls: dict[str, int] = defaultdict(int)


def timed(bucket, fn):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            spent[bucket] += time.perf_counter() - start
            calls[bucket] += 1

    return wrapper


zu.split_1d_runs = timed("  of which split_1d_runs", zu.split_1d_runs)
zp.split_1d_runs = zu.split_1d_runs
zp.make_chunk_info_for_rust_with_indices = timed(
    "prep (describe to rust)", zp.make_chunk_info_for_rust_with_indices
)

zarr.config.set(
    {
        "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
        "codec_pipeline.file_handle_cache_size": 512,
    }
)

root = zarr.open_group(COLLECTION, mode="r")
names = sorted(k for k in root.keys() if k.startswith("dataset_"))
datasets = [sparse_dataset(root[f"{n}/X"]) for n in names]
n_obs = [d.shape[0] for d in datasets]

# The pyclass has no __dict__, so the Rust entry point cannot be wrapped
# directly. Time the whole pipeline read instead and take Rust as the
# remainder once prep is subtracted.
_pipeline_read = zp.ZarrsCodecPipeline.read


async def timed_read(self, *args, **kwargs):
    start = time.perf_counter()
    try:
        return await _pipeline_read(self, *args, **kwargs)
    finally:
        spent["pipeline.read (prep + rust)"] += time.perf_counter() - start
        calls["pipeline.read (prep + rust)"] += 1


zp.ZarrsCodecPipeline.read = timed_read
zarrs.ZarrsCodecPipeline.read = timed_read

rng = np.random.default_rng()


def one_batch():
    which = rng.integers(0, len(datasets), NROWS)
    picks = [
        np.sort(rng.integers(0, n_obs[j], int((which == j).sum())))
        for j in range(len(datasets))
    ]
    return [d[r] for d, r in zip(datasets, picks) if len(r)]


one_batch()  # warm up
spent.clear()
calls.clear()

wall = time.perf_counter()
for _ in range(REPS):
    one_batch()
wall = time.perf_counter() - wall

print(f"{NROWS} sorted rows x {REPS} reps: {wall * 1e3:.0f} ms total\n")
read = spent["pipeline.read (prep + rust)"]
prep = spent["prep (describe to rust)"]
rows = [
    ("pipeline.read (prep + rust)", read, calls["pipeline.read (prep + rust)"]),
    ("  prep: describe to rust", prep, calls["prep (describe to rust)"]),
    ("    of which split_1d_runs", spent["  of which split_1d_runs"],
     calls["  of which split_1d_runs"]),
    ("  rust: fetch + decode", read - prep, 0),
    ("outside the pipeline (zarr indexing + anndata assembly)", wall - read, 0),
]
print(f"{'phase':<56} {'ms':>9} {'% wall':>7} {'calls':>7}")
for name, seconds, n in rows:
    print(
        f"{name:<56} {seconds * 1e3:>9.1f} {100 * seconds / wall:>6.1f}% "
        f"{n if n else '':>7}"
    )

print("\n--- cProfile, top 20 by cumulative (Python side only) ---")
prof = cProfile.Profile()
prof.enable()
one_batch()
prof.disable()
buf = io.StringIO()
pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(20)
print("\n".join(buf.getvalue().splitlines()[4:32]))
