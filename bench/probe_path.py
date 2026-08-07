"""Which read path does a CSR row gather actually take?

ZarrsCodecPipeline.read falls back to zarr-python's BatchedCodecPipeline when
the selection is one it cannot describe for Rust, and it does so silently. If
that is what a row gather hits, then nothing in the Rust pipeline -- fetch
pool, file handle cache, read plan -- is on the path at all, and every knob
measures nothing.

Counts both routes over one gather and prints the split.
"""

from __future__ import annotations

import sys

import numpy as np
import zarr
from zarr.core import BatchedCodecPipeline

import zarrs
from anndata._core.sparse_dataset import sparse_dataset

PLATE = "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_0/X"
NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 64

counts = {"python": 0, "rust": 0}

_py_read = BatchedCodecPipeline.read


async def counted_py(self, *args, **kwargs):
    counts["python"] += 1
    return await _py_read(self, *args, **kwargs)


BatchedCodecPipeline.read = counted_py

_zarrs_read = zarrs.ZarrsCodecPipeline.read


async def counted_zarrs(self, *args, **kwargs):
    before = counts["python"]
    result = await _zarrs_read(self, *args, **kwargs)
    if counts["python"] == before:
        counts["rust"] += 1
    return result


zarrs.ZarrsCodecPipeline.read = counted_zarrs

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
d = sparse_dataset(zarr.open(PLATE, mode="r"))
rng = np.random.default_rng()
rows = rng.integers(0, d.shape[0], NROWS)

out = d[rows]

total = counts["python"] + counts["rust"]
print(f"rows={NROWS} nnz={out.nnz}")
print(f"  rust pipeline : {counts['rust']}")
print(f"  python fallback: {counts['python']}")
if total:
    print(f"  -> {100 * counts['rust'] // total}% of reads reach the Rust path")
if counts["python"]:
    print("FALLBACK IS ACTIVE: Rust-side knobs cannot affect this workload")
