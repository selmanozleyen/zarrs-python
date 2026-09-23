"""Inflating part of a blosc chunk, rather than all of it.

Values alone cannot test this: a partial inflate and a full one return the same bytes, so a
gate that silently never fires passes every correctness test and only the throughput moves.
Asserted through the counter, as the raw read unit is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import zarr

from zarrs._internal import block_path_stats

if TYPE_CHECKING:
    from pathlib import Path

CHUNK_UNIT = {"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"}
SHAPE = (256, 64)
CHUNKS = (8, 64)
SHARDS = (32, 64)


def _write(path: Path, *, compressors):
    values = np.arange(SHAPE[0] * SHAPE[1], dtype=np.float32).reshape(SHAPE)
    zarr.create_array(
        path,
        dtype=values.dtype,
        shape=values.shape,
        chunks=CHUNKS,
        shards=SHARDS,
        compressors=compressors,
    )[:] = values
    return values


def _read(path: Path, selection) -> tuple[np.ndarray, int]:
    before = block_path_stats()
    with zarr.config.set(CHUNK_UNIT):
        got = zarr.open_array(path, mode="r")[selection]
    return got, block_path_stats() - before


def test_a_blosc_chunk_is_inflated_in_part(tmp_path: Path) -> None:
    """The rows wanted, not the chunk holding them -- and the same values either way."""
    from zarr.codecs import BloscCodec

    path = tmp_path / "blosc.zarr"
    values = _write(path, compressors=BloscCodec(cname="lz4", clevel=5))
    rows = np.arange(0, SHAPE[0], 8)  # one row per inner chunk
    got, blocks = _read(path, (rows, slice(None)))

    np.testing.assert_array_equal(got, values[rows, :])
    assert blocks > 0, "the block path never ran on a blosc array"


def test_an_uncompressed_chunk_does_not_take_the_block_path(tmp_path: Path) -> None:
    """No blosc, nothing to inflate in part -- the raw path owns that case."""
    path = tmp_path / "plain.zarr"
    values = _write(path, compressors=None)
    rows = np.arange(0, SHAPE[0], 8)
    got, blocks = _read(path, (rows, slice(None)))

    np.testing.assert_array_equal(got, values[rows, :])
    assert blocks == 0, f"the block path ran without a blosc codec: {blocks} jobs"


def test_a_checksummed_chunk_does_not_take_the_block_path(tmp_path: Path) -> None:
    """`crc32c` beside blosc means the decompressed bytes are not the chunk's bytes."""
    from zarr.codecs import BloscCodec, Crc32cCodec

    path = tmp_path / "crc.zarr"
    values = _write(path, compressors=[BloscCodec(cname="lz4", clevel=5), Crc32cCodec()])
    rows = np.arange(0, SHAPE[0], 8)
    got, blocks = _read(path, (rows, slice(None)))

    np.testing.assert_array_equal(got, values[rows, :])
    assert blocks == 0, f"the block path ran with a codec beside blosc: {blocks} jobs"
