"""Reading sampled rows of a CSR matrix, the way AnnData stores one.

`X` is not a 2-D array here: it is three 1-D arrays, `indptr`, `indices` and `data`. Sampling
rows means gathering `indptr` at scattered positions, then reading `indices[start:stop]` and
`data[start:stop]` for each sampled row. Batched into one read that is a sorted integer array
whose runs are the rows' nnz spans -- variable length, and at offsets with no relation to the
chunk grid.

That differs from dense row sampling in every way that matters to the read path: one dimension
instead of two, long runs instead of single rows, and unaligned boundaries. Correctness has to
hold with and without a compressor, because the two decide whether a run can be read on its own
or only as the inner chunk containing it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

N_OBS = 400
N_VAR = 5000
CHUNK = 4096
SHARD = 16384

CONFIGS = {
    "unplanned": {"codec_pipeline.integer_array_indexing": True},
    "planned": {
        "codec_pipeline.integer_array_indexing": True,
        "codec_pipeline.plan_reads": True,
    },
}


@pytest.fixture(params=["zstd", "none"])
def compressors(request):
    return "auto" if request.param == "zstd" else None


@pytest.fixture
def csr(tmp_path: Path, compressors) -> tuple[Path, dict[str, np.ndarray]]:
    """A CSR matrix with variable nnz per row, stored as AnnData stores one."""
    rng = np.random.default_rng(0)
    nnz_per_row = rng.integers(20, 400, size=N_OBS)
    indptr = np.concatenate([[0], np.cumsum(nnz_per_row)]).astype(np.int64)
    nnz = int(indptr[-1])
    indices = np.concatenate(
        [np.sort(rng.choice(N_VAR, size=n, replace=False)) for n in nnz_per_row]
    ).astype(np.int32)
    data = rng.random(nnz, dtype=np.float32)

    path = tmp_path / "X"
    kwargs = {} if compressors == "auto" else {"compressors": compressors}
    for name, array in (("indptr", indptr), ("indices", indices), ("data", data)):
        # indptr is small enough to live in one chunk, as it does in practice.
        chunks = (CHUNK,) if name != "indptr" else (len(indptr),)
        shards = (SHARD,) if name != "indptr" else (len(indptr),)
        z = zarr.create_array(
            path / name,
            dtype=array.dtype,
            shape=array.shape,
            chunks=chunks,
            shards=shards,
            **kwargs,
        )
        z[:] = array
    return path, {"indptr": indptr, "indices": indices, "data": data}


def row_span_selection(indptr: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The element positions of `rows`, batched into one sorted selection."""
    return np.concatenate([np.arange(indptr[r], indptr[r + 1]) for r in rows])


@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
@pytest.mark.parametrize(
    "n_rows",
    # 1 row is a single run; 256 of 400 rows is dense enough that runs share inner chunks.
    [1, 8, 64, 256],
)
def test_sampled_rows_match(
    csr: tuple[Path, dict[str, np.ndarray]], config: str, n_rows: int
) -> None:
    path, truth = csr
    rng = np.random.default_rng(n_rows)
    rows = np.sort(rng.choice(N_OBS, size=n_rows, replace=False))
    selection = row_span_selection(truth["indptr"], rows)

    with zarr.config.set(
        {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"} | CONFIGS[config]
    ):
        for name in ("indices", "data"):
            got = zarr.open_array(path / name, mode="r")[selection]
            np.testing.assert_array_equal(got, truth[name][selection])


@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
def test_indptr_gather(csr: tuple[Path, dict[str, np.ndarray]], config: str) -> None:
    """The row boundaries are a scattered gather of single elements, and consecutive sampled
    rows share a boundary, so the positions must be deduplicated to stay strictly increasing."""
    path, truth = csr
    rng = np.random.default_rng(1)
    rows = np.sort(rng.choice(N_OBS, size=32, replace=False))
    wanted = np.unique(np.concatenate([rows, rows + 1]))

    with zarr.config.set(
        {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"} | CONFIGS[config]
    ):
        got = zarr.open_array(path / "indptr", mode="r")[wanted]
    np.testing.assert_array_equal(got, truth["indptr"][wanted])


@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
def test_a_single_row_spanning_a_chunk_boundary(
    csr: tuple[Path, dict[str, np.ndarray]], config: str
) -> None:
    """A row's span has nothing to do with the chunk grid, so some rows straddle a boundary and
    some straddle a shard boundary. Those are the cases where a run becomes several reads."""
    path, truth = csr
    indptr = truth["indptr"]
    straddling = [
        r
        for r in range(N_OBS)
        if indptr[r] // CHUNK != indptr[r + 1] // CHUNK  # crosses an inner chunk
    ]
    assert straddling, "expected some rows to cross a chunk boundary"

    with zarr.config.set(
        {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"} | CONFIGS[config]
    ):
        array = zarr.open_array(path / "data", mode="r")
        for row in straddling[:5]:
            selection = np.arange(indptr[row], indptr[row + 1])
            np.testing.assert_array_equal(array[selection], truth["data"][selection])
