"""Selection structure of a CSR gather, sorted vs unsorted, side by side.

Answers the question that decides whether splitting into runs is worth doing:
how long is a run, on average? A run is the unit `split_1d_runs` turns into a
ChunkItem, so short runs mean many tiny reads.
"""

from __future__ import annotations

import numpy as np
import zarr

import zarrs
from anndata._core.sparse_dataset import sparse_dataset
from zarrs import pipeline as zp

PLATE = "/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_collection.zarr/dataset_0/X"

rec: list[tuple[str, str, int | None, int | None]] = []
_read = zp.ZarrsCodecPipeline.read


def kind(s) -> str:
    if isinstance(s, tuple):
        return "tuple[" + ",".join(kind(x) for x in s) + "]"
    if isinstance(s, slice):
        return "slice"
    if isinstance(s, np.ndarray):
        return f"ndarray[{s.size}]"
    return type(s).__name__


async def traced(self, batch_info, out, drop_axes=()):
    batch_info = list(batch_info)
    for _bg, _spec, cs, os_, _ in batch_info:
        c = cs[0] if isinstance(cs, tuple) else cs
        o = os_[0] if isinstance(os_, tuple) else os_
        runs = None
        if isinstance(c, np.ndarray) and c.size > 1:
            if isinstance(o, np.ndarray) and o.size == c.size:
                runs = int(np.flatnonzero((np.diff(c) != 1) | (np.diff(o) != 1)).size) + 1
            else:
                runs = int(np.flatnonzero(np.diff(c) != 1).size) + 1
        rec.append((kind(cs), kind(os_), getattr(c, "size", None), runs))
    return await _read(self, batch_info, out, drop_axes)


zp.ZarrsCodecPipeline.read = traced
zarrs.ZarrsCodecPipeline.read = traced

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
d = sparse_dataset(zarr.open(PLATE, mode="r"))
rng = np.random.default_rng(0)
base = rng.integers(0, d.shape[0], 512)

for label, rows in (("UNSORTED", base), ("SORTED", np.sort(base))):
    rec.clear()
    d[rows]
    fancy = [x for x in rec if x[3] is not None]
    coords = sum(x[2] for x in fancy)
    runs = sum(x[3] for x in fancy)
    print(f"{label}: {len(rec)} selections, {len(fancy)} fancy")
    print(f"   coords={coords} runs={runs} coords_per_run={coords / max(runs, 1):.1f}")
    for ck, ok, size, rn in rec[:3]:
        print(f"   chunk_sel={ck:18s} out_sel={ok:18s} size={size} runs={rn}")
