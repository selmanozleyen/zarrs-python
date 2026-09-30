from __future__ import annotations

from math import prod
from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

# Not a multiple of SHARD, so the last shard is only partly covered by the array.
N = 40_000
CHUNK = 4_096
SHARD = 16_384

CHUNK_UNIT = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}


@pytest.fixture(params=["zstd", "none"])
def compressors(request):
    return "auto" if request.param == "zstd" else None


@pytest.fixture
def array(tmp_path: Path, compressors) -> tuple[Path, np.ndarray]:
    values = np.arange(N, dtype=np.float32)
    kwargs = {} if compressors == "auto" else {"compressors": compressors}
    path = tmp_path / "a"
    z = zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=(CHUNK,),
        shards=(SHARD,),
        **kwargs,
    )
    z[:] = values
    return path, values


def selections() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "one whole chunk": np.arange(CHUNK, 2 * CHUNK),
        "scattered": np.sort(rng.choice(N, size=2_000, replace=False)),
        # Duplicates must be kept.
        "with duplicates": np.repeat(
            np.sort(rng.choice(N, size=500, replace=False)), 3
        ),
        "partly covered last shard": np.arange(N - 100, N),
        "single element": np.array([N - 1]),
        "every second": np.arange(0, N, 2),
    }


@pytest.mark.parametrize("name", list(selections()))
def test_selection_matches_and_takes_the_handle(
    array: tuple[Path, np.ndarray], entries: dict[str, int], name: str
) -> None:
    path, truth = array
    selection = selections()[name]

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] > 0


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(np.array([9_000, 40, 8_000, 39]), id="decreasing"),
        pytest.param(slice(0, 4 * CHUNK, 3), id="strided slice"),
    ],
)
def test_ineligible_selections_decline_and_are_still_right(
    array: tuple[Path, np.ndarray], entries: dict[str, int], selection
) -> None:
    path, truth = array

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] == 0


@pytest.mark.parametrize(
    ("written", "name"),
    [
        # Some inner chunks of the shard exist, others never do.
        (slice(0, CHUNK), "partial shard"),
        # Whole shards written, whole shards missing.
        (slice(0, SHARD), "whole shard"),
    ],
)
def test_unwritten_chunks_read_as_fill(
    tmp_path: Path, entries: dict[str, int], written: slice, name: str
) -> None:
    """`rows` straddles the boundary: written chunks, unwritten ones, and the chunk with it."""
    values = np.arange(N, dtype=np.float32)
    path = tmp_path / f"sparse-{name.replace(' ', '-')}"
    z = zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=(CHUNK,),
        shards=(SHARD,),
        fill_value=np.float32(-7),
    )
    z[written] = values[written]

    expected = np.full(N, -7, dtype=np.float32)
    expected[written] = values[written]

    rows = np.sort(
        np.concatenate(
            [
                np.arange(written.stop - 200, written.stop + 200),
                np.arange(N - 300, N - 100),
            ]
        )
    )
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows]

    np.testing.assert_array_equal(got, expected[rows])
    assert entries["handle"] > 0, "this selection should have taken the chunk-unit path"


def test_a_column_split_inner_chunk_is_served(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """A 2-D array whose inner chunk is narrower than the shard is served, one item per band."""
    values = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    path = tmp_path / "two_d"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(16, 16), shards=(32, 32)
    )
    z[:] = values

    rows = np.array([1, 5, 5, 40])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, :]

    np.testing.assert_array_equal(got, values[rows, :])
    assert entries["handle"] > 0, (
        "a column-split 2-D selection should reach the chunk-unit path"
    )


@pytest.fixture
def full_width(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """A 2-D array whose inner chunk spans every column, as `annbatch.write_sharded` writes."""
    values = np.arange(256 * 48, dtype=np.float32).reshape(256, 48)
    path = tmp_path / "full_width"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 48), shards=(64, 48)
    )
    z[:] = values
    return path, values


def test_full_width_two_dimensional_takes_the_path(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """Rows grouped by inner chunk, one item per chunk."""
    path, values = full_width
    rows = np.array([1, 3, 3, 9, 60, 61, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, :]

    np.testing.assert_array_equal(got, values[rows, :])
    assert entries["handle"] > 0, (
        "a full-width 2-D selection did not take the chunk-unit path"
    )


def test_a_partial_column_slice_takes_the_path(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """`X[rows, 8:24]` of a full-width inner chunk."""
    path, values = full_width
    rows = np.array([1, 3, 3, 9, 60, 61, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, 8:24]

    np.testing.assert_array_equal(got, values[rows, 8:24])
    assert entries["handle"] > 0, (
        "a partial column slice did not take the chunk-unit path"
    )


def test_a_strided_column_slice_still_falls_back(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """Step 2 is not one run per row."""
    path, values = full_width
    rows = np.array([1, 3, 9, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, 8:24:2]

    np.testing.assert_array_equal(got, values[rows, 8:24:2])
    assert entries["handle"] == 0, "a strided column slice reached the chunk-unit path"


def test_a_column_slice_matches_zarr_python(
    full_width: tuple[Path, np.ndarray],
) -> None:
    """Byte for byte against the reference pipeline, on the same store."""
    path, _ = full_width
    rows = np.sort(np.random.default_rng(0).choice(256, size=64, replace=False))
    with zarr.config.set(CHUNK_UNIT):
        mine = zarr.open_array(path, mode="r")[rows, 8:24]
    with zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    ):
        theirs = zarr.open_array(path, mode="r")[rows, 8:24]
    np.testing.assert_array_equal(mine, theirs)


def test_the_contiguity_rule() -> None:
    """One run only when every axis before the last partial one takes a single element."""
    from zarrs.utils import _is_one_run

    assert _is_one_run([16], (48,))
    assert _is_one_run([48], (48,))
    assert _is_one_run([3, 10], (5, 10))
    assert not _is_one_run([5, 4], (5, 10))
    assert _is_one_run([1, 4], (5, 10))


def test_full_width_matches_zarr_python(
    full_width: tuple[Path, np.ndarray],
) -> None:
    """Byte for byte against the reference pipeline, on the same store."""
    path, _ = full_width
    rows = np.sort(np.random.default_rng(0).choice(256, size=64, replace=False))
    with zarr.config.set(CHUNK_UNIT):
        mine = zarr.open_array(path, mode="r")[rows, :]
    with zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    ):
        theirs = zarr.open_array(path, mode="r")[rows, :]
    np.testing.assert_array_equal(mine, theirs)


# zarr warns that this layout disables partial reads. That is the layout under test.
@pytest.mark.filterwarnings("ignore:Combining a `sharding_indexed` codec")
def test_a_codec_after_sharding_is_refused(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """A codec after the sharding codec compresses the shard, so its index stops addressing it."""
    from zarr.codecs import BytesCodec, ShardingCodec

    values = np.arange(N, dtype=np.float32)
    path = tmp_path / "outer_compressed"
    z = zarr.create_array(
        path,
        shape=values.shape,
        chunks=(SHARD,),
        dtype="float32",
        # An explicit sharding serializer leaves the default compressor outside it.
        serializer=ShardingCodec(chunk_shape=(CHUNK,), codecs=[BytesCodec()]),
    )
    z[:] = values

    selection = np.sort(np.random.default_rng(0).choice(N, size=200, replace=False))
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]
    np.testing.assert_array_equal(got, values[selection])
    assert entries["handle"] == 0, "an outer codec reached the chunk-unit path"


def test_a_shard_holding_one_inner_chunk(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """`shards == chunks`."""
    values = np.arange(4096, dtype=np.float32)
    path = tmp_path / "one_per_shard"
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(1024,), shards=(1024,)
    )[:] = values

    rows = np.array([1, 5, 9, 300, 1025, 4095])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows]

    np.testing.assert_array_equal(got, values[rows])
    assert entries["handle"] > 0, "this selection should have taken the chunk-unit path"


def test_an_array_narrower_than_its_chunk_takes_the_path(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    """The array is 30 columns wide but its chunk is 48, so every chunk ends in fill."""
    values = np.arange(256 * 30, dtype=np.float32).reshape(256, 30)
    path = tmp_path / "narrow"
    z = zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 48), shards=(64, 48)
    )
    z[:] = values
    rows = np.array([1, 3, 3, 9, 60, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, :]

    np.testing.assert_array_equal(got, values[rows, :])
    assert entries["handle"] > 0, "a narrow array did not take the chunk-unit path"


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param(slice(CHUNK - 10, 2 * CHUNK + 10), id="spanning chunks"),
        pytest.param(slice(None), id="everything"),
        pytest.param(slice(N - 100, N), id="partly covered last shard"),
    ],
)
def test_a_contiguous_slice_takes_the_path(
    array: tuple[Path, np.ndarray], entries: dict[str, int], selection
) -> None:
    path, truth = array

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, truth[selection])
    assert entries["handle"] > 0, "a contiguous slice did not take the chunk-unit path"


def test_a_single_column_takes_the_path(
    full_width: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    """`X[rows, 5]` arrives as a point selection with a constant column."""
    path, values = full_width
    rows = np.array([1, 3, 3, 9, 60, 61, 200])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows, 5]

    np.testing.assert_array_equal(got, values[rows, 5])
    assert entries["handle"] > 0, "a single column did not take the chunk-unit path"


@pytest.fixture(params=["1d", "2d"])
def unsharded(tmp_path: Path, request) -> tuple[Path, np.ndarray]:
    """A plain chunked array with no sharding codec at all."""
    if request.param == "1d":
        values = np.arange(4_000, dtype=np.float32)
        chunks = (256,)
    else:
        values = np.arange(1_024 * 24, dtype=np.float32).reshape(1_024, 24)
        chunks = (32, 24)
    path = tmp_path / f"plain_{request.param}"
    z = zarr.create_array(path, dtype=values.dtype, shape=values.shape, chunks=chunks)
    z[:] = values
    return path, values


def test_an_unsharded_array_takes_the_path(
    unsharded: tuple[Path, np.ndarray], entries: dict[str, int]
) -> None:
    path, values = unsharded
    rows = np.array([1, 3, 3, 40, 41, 300, 999])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows]

    np.testing.assert_array_equal(got, values[rows])
    assert entries["handle"] > 0, "an unsharded array did not take the chunk-unit path"


def test_an_unsharded_array_matches_zarr_python(
    unsharded: tuple[Path, np.ndarray],
) -> None:
    path, _ = unsharded
    rng = np.random.default_rng(0)
    with zarr.config.set(CHUNK_UNIT):
        arr = zarr.open_array(path, mode="r")
        rows = np.sort(rng.choice(arr.shape[0], size=200, replace=False))
        mine = arr[rows]
    with zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    ):
        theirs = zarr.open_array(path, mode="r")[rows]
    np.testing.assert_array_equal(mine, theirs)


def test_an_unsharded_array_reads_unwritten_chunks_as_fill(
    tmp_path: Path, entries: dict[str, int]
) -> None:
    path = tmp_path / "sparse_plain"
    z = zarr.create_array(
        path, dtype=np.float32, shape=(4_000,), chunks=(256,), fill_value=np.float32(-7)
    )
    z[0:256] = np.arange(256, dtype=np.float32)

    rows = np.array([1, 5, 2_000, 3_999])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[rows]

    np.testing.assert_array_equal(got, np.array([1, 5, -7, -7], dtype=np.float32))
    assert entries["handle"] > 0, "an unsharded array did not take the chunk-unit path"


@pytest.fixture
def nested(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Two levels of sharding: 256-row shard -> 32-row subshard -> 8-row inner chunk."""
    from zarr.codecs import BytesCodec, ShardingCodec

    values = np.arange(1024 * 48, dtype=np.float32).reshape(1024, 48)
    path = tmp_path / "nested2d"
    z = zarr.create_array(
        path,
        shape=values.shape,
        chunks=(256, 48),
        dtype="float32",
        compressors=None,
        serializer=ShardingCodec(
            chunk_shape=(32, 48),
            codecs=[ShardingCodec(chunk_shape=(8, 48), codecs=[BytesCodec()])],
        ),
    )
    z[:] = values
    return path, values


@pytest.mark.parametrize(
    ("name", "read"),
    [
        ("whole rows", lambda a, r: a.oindex[r, :]),
        ("column sub-box", lambda a, r: a.oindex[r, 8:24]),
        ("contiguous slice", lambda a, r: a.oindex[10:200, :]),
    ],
)
def test_every_shape_works_through_two_shard_levels(
    nested: tuple[Path, np.ndarray], entries: dict[str, int], name: str, read
) -> None:
    path, _ = nested
    rows = np.array([1, 3, 3, 40, 300, 900])
    with zarr.config.set(CHUNK_UNIT):
        got = read(zarr.open_array(path, mode="r"), rows)
    with zarr.config.set(
        {"codec_pipeline.path": "zarr.core.codec_pipeline.BatchedCodecPipeline"}
    ):
        theirs = read(zarr.open_array(path, mode="r"), rows)

    np.testing.assert_array_equal(got, theirs)
    assert entries["handle"] > 0, (
        f"{name} did not take the chunk-unit path through two levels"
    )


# --- The description, checked against the bytes it claims, with no store and no Rust ----


def _fold(starts, extents) -> int:
    """A coordinate as a row-major element offset."""
    offset, stride = 0, 1
    for axis in reversed(range(len(starts))):
        offset += int(starts[axis]) * stride
        stride *= int(extents[axis])
    return offset


def _batch(shard: tuple[int, ...], rows, boxes: list[tuple[int, int]]):
    """The entries zarr builds for `X[rows, lo:hi, ...]`, and the output shape."""
    import itertools
    import types

    sliced = isinstance(rows, tuple)
    row_list = list(range(*rows)) if sliced else list(rows)
    out_shape = (len(row_list), *[hi - lo for lo, hi in boxes])
    grids = [
        range(row_list[0] // shard[0], row_list[-1] // shard[0] + 1),
        *[range(lo // s, (hi - 1) // s + 1) for (lo, hi), s in zip(boxes, shard[1:])],
    ]
    entries = []
    for coord in itertools.product(*grids):
        take = [i for i, r in enumerate(row_list) if r // shard[0] == coord[0]]
        if not take:
            continue
        local = [row_list[i] - coord[0] * shard[0] for i in take]
        chunk_sel = [
            slice(local[0], local[-1] + 1)
            if sliced
            else np.array(local, dtype=np.int64)
        ]
        out_sel = [slice(take[0], take[-1] + 1)]
        for axis, ((lo, hi), extent) in enumerate(zip(boxes, shard[1:]), start=1):
            base = coord[axis] * extent
            a, b = max(lo, base), min(hi, base + extent)
            if a >= b:
                break
            chunk_sel.append(slice(a - base, b - base))
            out_sel.append(slice(a - lo, b - lo))
        else:
            entries.append(
                (
                    types.SimpleNamespace(path="c/" + "/".join(map(str, coord))),
                    types.SimpleNamespace(shape=tuple(shard)),
                    tuple(chunk_sel),
                    tuple(out_sel),
                    False,
                )
            )
    return entries, out_shape


def _wanted(entry, out_shape) -> list[tuple[int, int]]:
    """(shard element, output element) pairs the selection asks for, paired in order."""
    import itertools

    _, spec, chunk_sel, out_sel, _ = entry
    axes = [
        list(s) if isinstance(s, np.ndarray) else list(range(s.start, s.stop))
        for s in chunk_sel
    ]
    src = [_fold(c, spec.shape) for c in itertools.product(*axes)]
    dst = [
        _fold(c, out_shape)
        for c in itertools.product(*[range(s.start, s.stop) for s in out_sel])
    ]
    assert len(src) == len(dst)
    return list(zip(src, dst))


def _described(args) -> list[tuple[int, int]]:
    """The same pairs, expanded the way Rust reads the description back."""
    if args[0] == "span":
        _, _key, chunk_shape, shape, first, count, out_start, _inner = args
        row, out_row = prod(chunk_shape[1:]), prod(shape[1:])
        return [
            ((first + k) * row + j, (out_start + k) * out_row + j)
            for k in range(count)
            for j in range(row)
        ]
    _, _key, chunk_shape, shape, indices, out_starts, out_widths, _inner, starts = args
    row, out_row = prod(chunk_shape[1:]), prod(shape[1:])
    src_offset = _fold(starts, chunk_shape[1:])
    dst_offset = _fold(out_starts[1:], shape[1:])
    run = prod(out_widths[1:])
    return [
        (
            int(index) * row + src_offset + j,
            (out_starts[0] + i) * out_row + dst_offset + j,
        )
        for i, index in enumerate(indices)
        for j in range(run)
    ]


@pytest.mark.parametrize(
    ("shard", "inner"),
    [
        ((4, 5), (4, 5)),
        ((2, 3, 4), (2, 3, 4)),
        # Two inner chunks across the shard, so a column range can cross into both.
        ((4, 12), (4, 6)),
        ((16, 12), (8, 6)),
        ((4, 12, 4), (4, 6, 4)),
        # An inner chunk that does not divide the shard, so the last band is short.
        ((4, 12), (4, 5)),
    ],
    ids=[
        "rank-2",
        "rank-3",
        "divided",
        "divided-tall",
        "divided-rank-3",
        "ragged-inner",
    ],
)
@pytest.mark.parametrize(
    "rows",
    [(0, 8), (1, 9), [0, 1, 3, 6], [5]],
    ids=["slice", "unaligned-slice", "list", "one"],
)
def test_a_description_names_exactly_its_own_bytes(shard, inner, rows) -> None:
    """Every column range over a shard grid, described and expanded back."""
    import itertools

    from zarrs.utils import _chunk_unit_args

    width = shard[1] * 3
    checked = 0
    for lo, hi in itertools.combinations(range(width + 1), 2):
        boxes = [(lo, hi), *[(0, shard[axis]) for axis in range(2, len(shard))]]
        entries, out_shape = _batch(shard, rows, boxes)
        pairs: list[tuple[int, int]] = []
        for entry in entries:
            pushes = _chunk_unit_args(entry, out_shape, (), tuple(inner))
            if pushes is None:
                break
            said = [pair for args in pushes for pair in _described(args)]
            assert sorted(said) == sorted(_wanted(entry, out_shape)), (
                f"{[a[0] for a in pushes]} for {entry[0].path} names bytes the selection did "
                f"not ask for: {entry[2]} -> {entry[3]} of {out_shape}"
            )
            pairs += said
        else:
            got = sorted(d for _, d in pairs)
            assert got == list(range(prod(out_shape))), (
                f"X[{rows}, {lo}:{hi}] of {shard}: the descriptions do not tile {out_shape}"
            )
            checked += 1
    assert checked, "every column range declined; this asserted nothing"


def test_a_strided_output_box_is_declined() -> None:
    """A partial last axis behind a wider one is not one run per index, so it is declined."""
    import types

    from zarrs.utils import _chunk_unit_args

    entry = (
        types.SimpleNamespace(path="c/0/0/0"),
        types.SimpleNamespace(shape=(4, 5, 10)),
        (slice(0, 4), slice(0, 5), slice(0, 4)),
        (slice(0, 4), slice(0, 5), slice(0, 4)),
        False,
    )
    assert _chunk_unit_args(entry, (4, 5, 10), (), (4, 5, 10)) is None


@pytest.mark.parametrize(
    "selection",
    [
        pytest.param((5, 5), id="both-axes-scalar"),
        pytest.param((0, 0), id="both-axes-scalar-at-the-origin"),
        pytest.param((31, 23), id="both-axes-scalar-at-the-end"),
        pytest.param((slice(None), 3), id="scalar-on-a-trailing-axis"),
        pytest.param((5, slice(None)), id="scalar-row"),
        pytest.param((5, slice(6, 18)), id="scalar-row-and-a-column-band"),
        # A CoordinateIndexer with a constant column.
        pytest.param(
            (np.array([1, 5, 20]), 7), id="a-constant-column-is-a-scalar-axis"
        ),
    ],
)
def test_a_scalar_axis_is_served(
    tmp_path: Path, entries: dict[str, int], selection
) -> None:
    """zarr drops a scalar axis from the output without saying so in `drop_axes`."""
    values = np.arange(32 * 24, dtype=np.float64).reshape(32, 24)
    path = tmp_path / "scalar.zarr"
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 6), shards=(16, 12)
    )[:] = values

    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]

    np.testing.assert_array_equal(got, values[selection])
    assert entries["handle"] > 0, (
        "a scalar axis should not send the batch to zarr-python"
    )


@pytest.mark.parametrize(
    "cols",
    [
        pytest.param(np.array([7]), id="one-column"),
        pytest.param(np.array([7, 7]), id="the-same-column-twice"),
    ],
)
def test_a_kept_constant_column_axis_is_not_rebuilt(tmp_path: Path, cols) -> None:
    """`oindex[rows, [7]]` keeps the column axis, so an extent of one would be a lie."""
    values = np.arange(32 * 24, dtype=np.float64).reshape(32, 24)
    path = tmp_path / "kept.zarr"
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 6), shards=(16, 12)
    )[:] = values

    rows = np.array([4, 5, 6])
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r").oindex[rows, cols]

    np.testing.assert_array_equal(got, values[np.ix_(rows, cols)])


def test_a_bare_ndarray_selector_is_declined_not_mis_described(tmp_path: Path) -> None:
    """An out-of-order index on a rank-1 array arrives as a bare ndarray, not a tuple."""
    from zarrs.utils import DiscontiguousArrayError

    values = np.arange(8, dtype=np.float32)
    path = tmp_path / "bare.zarr"
    zarr.create_array(path, dtype=values.dtype, shape=values.shape, chunks=(4,))[:] = (
        values
    )
    idx = np.array([0, 4, 5, 1])

    with zarr.config.set(CHUNK_UNIT):
        np.testing.assert_array_equal(zarr.open_array(path, mode="r")[idx], values[idx])

    with (
        zarr.config.set({**CHUNK_UNIT, "codec_pipeline.strict": True}),
        pytest.raises(DiscontiguousArrayError),
    ):
        zarr.open_array(path, mode="r")[idx]


def test_a_row_of_an_array_wider_than_one_shard(tmp_path: Path) -> None:
    """Two entries in one batch: `drop_axes` must not leak from the first into the second."""
    values = np.arange(32 * 40, dtype=np.float64).reshape(32, 40)
    path = tmp_path / "wide.zarr"
    zarr.create_array(
        path, dtype=values.dtype, shape=values.shape, chunks=(8, 10), shards=(16, 20)
    )[:] = values

    with zarr.config.set({**CHUNK_UNIT, "codec_pipeline.strict": True}):
        got = zarr.open_array(path, mode="r")[5, :]

    np.testing.assert_array_equal(got, values[5, :])
