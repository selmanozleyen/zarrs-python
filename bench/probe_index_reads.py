"""How many shard indexes does one batch read, and how many are re-reads?

`partial_decoder_cache` is a local in `retrieve_chunks_and_apply_index`, so a
decoder is built per shard per call and each construction reads that shard's
index from storage. Nothing is remembered between batches even though a
read-only array's indexes never change.

Counts, per batch: distinct shard keys touched (= decoders built = index
reads), and how many of those were already touched by an earlier batch (=
reads a cross-batch cache would remove).
"""

from __future__ import annotations

import sys

import numpy as np
import zarr

import zarrs
from anndata._core.sparse_dataset import sparse_dataset
from zarrs import pipeline as zp

COLLECTION = "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr"
NROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
BATCHES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
INDEX_BYTES = 920 * 2 * 8  # 920 inner chunks per shard, (offset, length) u64

batch_keys: set[str] = set()
seen_ever: set[str] = set()
_read = zp.ZarrsCodecPipeline.read


async def traced(self, batch_info, out, drop_axes=()):
    batch_info = list(batch_info)
    for byte_getter, _spec, _cs, _os, _ in batch_info:
        batch_keys.add(str(byte_getter.path))
    return await _read(self, batch_info, out, drop_axes)


zp.ZarrsCodecPipeline.read = traced
zarrs.ZarrsCodecPipeline.read = traced

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
root = zarr.open_group(COLLECTION, mode="r")
names = sorted(k for k in root.keys() if k.startswith("dataset_"))
datasets = [sparse_dataset(root[f"{n}/X"]) for n in names]
n_obs = [d.shape[0] for d in datasets]
rng = np.random.default_rng()

print(f"{NROWS} sorted rows over {len(datasets)} datasets, {BATCHES} batches")
print(f"assuming {INDEX_BYTES / 1024:.1f} KiB of index per shard\n")
print(f"{'batch':>6} {'shards':>8} {'repeat':>8} {'index MiB':>10} {'saved MiB':>10}")

total, repeat_total = 0, 0
for b in range(BATCHES):
    batch_keys.clear()
    which = rng.integers(0, len(datasets), NROWS)
    for j, d in enumerate(datasets):
        rows = rng.integers(0, n_obs[j], int((which == j).sum()))
        if len(rows):
            d[np.sort(rows)]
    repeat = len(batch_keys & seen_ever)
    seen_ever |= batch_keys
    total += len(batch_keys)
    repeat_total += repeat
    print(
        f"{b:>6} {len(batch_keys):>8} {repeat:>8} "
        f"{len(batch_keys) * INDEX_BYTES / 2**20:>10.1f} {repeat * INDEX_BYTES / 2**20:>10.1f}"
    )

print(
    f"\ntotal index reads: {total}   of which repeats: {repeat_total} "
    f"({100 * repeat_total // max(total, 1)}%)"
)
print(f"distinct shards ever touched: {len(seen_ever)} "
      f"({len(seen_ever) * INDEX_BYTES / 2**20:.1f} MiB resident to cache them all)")
