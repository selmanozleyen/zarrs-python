"""Which mechanism issues the reads must not change what comes back.

`plan_reads_io` picks between blocking reads on plain OS threads and an io_uring ring. They differ
in how many reads can be outstanding and what that costs, and in nothing else -- so the same
selection through either one has to return identical bytes.

The ring is not reachable everywhere: it is Linux-only, and even on Linux a cluster can forbid it
(`kernel.io_uring_disabled=2` makes `io_uring_setup` return EPERM). So `uring` is a request, not an
assertion, and asking for it where it cannot be had must fall back rather than fail -- otherwise the
same code would have to be configured differently per site, which is how a benchmark ends up
measuring a different backend than its config claims.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import zarr

if TYPE_CHECKING:
    from pathlib import Path

PLANNED = {
    "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
    "codec_pipeline.integer_array_indexing": True,
    "codec_pipeline.plan_reads": True,
    "codec_pipeline.shard_index_cache_size": 8,
    # No fallback to hide behind: if a backend cannot serve the read, that must raise here rather
    # than be served correctly by zarr-python and look like a passing test.
    "codec_pipeline.strict": True,
}


@pytest.fixture
def sharded(tmp_path: Path) -> tuple[Path, np.ndarray]:
    """Compressed, so inner chunks are atomic and the read goes through the whole-unit path."""
    path = tmp_path / "a.zarr"
    values = np.arange(32 * 40, dtype="float32").reshape(32, 40)
    array = zarr.create_array(
        path,
        shape=values.shape,
        dtype="float32",
        chunks=(4, 5),
        shards=(16, 20),
        compressors=[zarr.codecs.BloscCodec(cname="lz4")],
    )
    array[:] = values
    return path, values


def read(path: Path, rows, **overrides) -> np.ndarray:
    with zarr.config.set(PLANNED | overrides):
        return zarr.open_array(path, mode="r")[rows, :]


@pytest.mark.parametrize("backend", ["auto", "threads", "uring"])
def test_backends_agree(sharded, backend):
    """Every backend returns the truth, including one that had to fall back to get there."""
    path, values = sharded
    rows = [0, 1, 5, 17, 18, 31]
    got = read(path, rows, **{"codec_pipeline.plan_reads_io": backend})
    np.testing.assert_array_equal(got, values[rows, :])


def test_depth_does_not_change_the_answer(sharded):
    """Queue depth is a throughput knob. A depth of one still has to be correct."""
    path, values = sharded
    rows = [2, 3, 9, 20, 30]
    for depth in (1, 4, 4096):
        got = read(
            path,
            rows,
            **{
                "codec_pipeline.plan_reads_io": "auto",
                "codec_pipeline.plan_reads_fetch_depth": depth,
            },
        )
        np.testing.assert_array_equal(got, values[rows, :], err_msg=f"{depth=}")


def test_hint_lookahead_does_not_change_the_answer(sharded):
    """Hints are advice, so they may be ignored, declined, or raced -- never believed.

    FADVISE(WILLNEED) only asks the kernel to start readahead; it does not promise the pages arrive,
    and a hint that fails is not an error. So every lookahead has to return the same bytes as no
    lookahead at all, including a window far longer than the read list, which is what exercises the
    hint cursor running off the end.
    """
    path, values = sharded
    rows = [1, 4, 6, 19, 25, 31]
    baseline = read(path, rows, **{"codec_pipeline.plan_reads_hint_lookahead": 0})
    np.testing.assert_array_equal(baseline, values[rows, :])
    for lookahead in (1, 8, 4096):
        got = read(
            path,
            rows,
            **{
                "codec_pipeline.plan_reads_io": "auto",
                "codec_pipeline.plan_reads_hint_lookahead": lookahead,
            },
        )
        np.testing.assert_array_equal(got, values[rows, :], err_msg=f"{lookahead=}")


def test_hints_share_the_ring_with_reads(sharded):
    """A hint occupies a queue entry, so hinting must not starve the reads it exists to help.

    With depth 1 there is room for exactly one entry at a time, and a lookahead far past that would
    wedge the loop if hints were pushed without counting against depth. The answer being right is
    the evidence that submission and hinting stayed in step.
    """
    path, values = sharded
    rows = [0, 7, 13, 28]
    got = read(
        path,
        rows,
        **{
            "codec_pipeline.plan_reads_io": "auto",
            "codec_pipeline.plan_reads_fetch_depth": 1,
            "codec_pipeline.plan_reads_hint_lookahead": 64,
        },
    )
    np.testing.assert_array_equal(got, values[rows, :])


def test_unknown_backend_is_rejected(sharded):
    """A typo must not quietly measure whatever the default happens to be."""
    path, _ = sharded
    with pytest.raises(ValueError, match="unknown plan_reads_io"):
        read(path, [0, 1], **{"codec_pipeline.plan_reads_io": "iouring"})
