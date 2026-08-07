"""What does 'random row access' actually look like at the storage level?

A CSR row is data[indptr[i]:indptr[i+1]] -- a contiguous span by construction.
So picking random *rows* is not the same as picking random *elements*, and the
per-row contiguity is a property of the format, not of anything the benchmark
or the pipeline does.

This builds a small synthetic CSR and reports, for sorted and unsorted random
row picks, how fragmented the chunk side and the output side each are.
Deliberately synthetic and local: the question is about mechanism, not scale.
"""

from __future__ import annotations

import tempfile

import numpy as np
import zarr
from scipy import sparse

import zarrs
from anndata._core.sparse_dataset import sparse_dataset
from zarrs import pipeline as zp

N_ROWS, N_COLS, PER_ROW = 2_000, 500, 40
CHUNK = 20_000

rec: list[tuple[np.ndarray | slice, np.ndarray | slice]] = []
_read = zp.ZarrsCodecPipeline.read


async def traced(self, batch_info, out, drop_axes=()):
    batch_info = list(batch_info)
    for _bg, _spec, cs, os_, _ in batch_info:
        c = cs[0] if isinstance(cs, tuple) else cs
        o = os_[0] if isinstance(os_, tuple) else os_
        rec.append((c, o))
    return await _read(self, batch_info, out, drop_axes)


zp.ZarrsCodecPipeline.read = traced
zarrs.ZarrsCodecPipeline.read = traced


def runs(a) -> int | None:
    if not isinstance(a, np.ndarray) or a.size < 2:
        return None
    return int(np.flatnonzero(np.diff(a) != 1).size) + 1


tmp = tempfile.mkdtemp()
rng = np.random.default_rng(0)
indptr = np.arange(0, (N_ROWS + 1) * PER_ROW, PER_ROW, dtype=np.int64)
nnz = indptr[-1]
m = sparse.csr_matrix(
    (
        rng.random(nnz, dtype=np.float32),
        rng.integers(0, N_COLS, nnz).astype(np.int32),
        indptr,
    ),
    shape=(N_ROWS, N_COLS),
)

g = zarr.open_group(tmp, mode="w")
g.attrs["encoding-type"] = "csr_matrix"
g.attrs["encoding-version"] = "0.1.0"
g.attrs["shape"] = [N_ROWS, N_COLS]
for name, arr in (("data", m.data), ("indices", m.indices), ("indptr", m.indptr)):
    g.create_array(name, shape=arr.shape, chunks=(CHUNK,), dtype=arr.dtype)[:] = arr

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
d = sparse_dataset(zarr.open_group(tmp, mode="r"))

picks = rng.choice(N_ROWS, 64, replace=False)
print(f"CSR: {N_ROWS} rows x {PER_ROW} nnz each, chunk={CHUNK}")
print(f"64 random rows -> {64 * PER_ROW} elements\n")

for label, rows in (("unsorted", picks), ("sorted", np.sort(picks))):
    rec.clear()
    d[rows]
    fancy = [(c, o) for c, o in rec if isinstance(c, np.ndarray) and c.size > 1]
    if not fancy:
        print(f"{label:9s}: no fancy selections (all slices)")
        continue
    coords = sum(c.size for c, _ in fancy)
    chunk_runs = sum(runs(c) for c, _ in fancy)
    both = sum(
        int(np.flatnonzero((np.diff(c) != 1) | (np.diff(o) != 1)).size) + 1
        if isinstance(o, np.ndarray) and o.size == c.size
        else runs(c)
        for c, o in fancy
    )
    out_kind = "ndarray" if isinstance(fancy[0][1], np.ndarray) else "slice"
    print(f"{label:9s}: coords={coords} out_selection={out_kind}")
    print(f"           chunk-side runs = {chunk_runs:5d}  ({coords / chunk_runs:7.1f} coords/run)")
    print(f"           both-side  runs = {both:5d}  ({coords / both:7.1f} coords/run)")
