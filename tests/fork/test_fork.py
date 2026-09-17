from __future__ import annotations

import os

import numpy as np
import pytest
import zarr

from .conftest import SELECTIONS, ChildResult, run_in_child, seeded

pytestmark = pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX only")


def assert_refused_for_tokio(result: ChildResult) -> None:
    assert result.code == 3, f"expected a refusal, got exit {result.code}: {result.error}"
    assert "RuntimeError" in result.error, result.error
    # Names this store's runtime, so a blanket refusal would not satisfy it.
    assert "this store's tokio runtime" in result.error, result.error
    assert "forkserver" in result.error, result.error


@pytest.mark.parametrize("selection", SELECTIONS.values(), ids=SELECTIONS.keys())
@pytest.mark.parametrize("shards", [False, True], ids=["chunks", "sharded"])
def test_a_forked_child_can_read(tmp_path, selection, *, shards: bool) -> None:
    path = str(tmp_path / "a.zarr")
    seeded(path, shards=shards)
    expected = np.asarray(zarr.open_array(path, mode="r")[selection])

    def read() -> None:
        got = np.asarray(zarr.open_array(path, mode="r")[selection])
        np.testing.assert_array_equal(got, expected)

    result = run_in_child(read)
    assert result.code == 0, result.error


def test_a_forked_child_can_write(tmp_path) -> None:
    path = str(tmp_path / "a.zarr")
    seeded(path)

    def write() -> None:
        # Not the fill value: writing zeros takes the erase branch, which "reads back as
        # zeros" would accept from a write that never happened.
        zarr.open_array(path, mode="r+")[:64, :64] = np.full((64, 64), 7, dtype="int16")

    result = run_in_child(write)
    assert result.code == 0, result.error
    np.testing.assert_array_equal(
        np.asarray(zarr.open_array(path, mode="r")[:64, :64]),
        np.full((64, 64), 7, dtype="int16"),
    )


def test_each_forked_child_rebuilds_its_own_pool(tmp_path) -> None:
    path = str(tmp_path / "a.zarr")
    seeded(path)
    expected = np.asarray(zarr.open_array(path, mode="r")[:])

    def read() -> None:
        np.testing.assert_array_equal(
            np.asarray(zarr.open_array(path, mode="r")[:]), expected
        )

    for _ in range(2):
        result = run_in_child(read)
        assert result.code == 0, result.error


def test_a_forked_child_is_refused_an_inherited_remote_store(http_url) -> None:
    array = zarr.open_array(http_url, mode="r")
    array[:8, :8]

    assert_refused_for_tokio(run_in_child(lambda: array[:]))


def test_a_forked_child_is_refused_a_remote_store_it_opens_itself(http_url) -> None:
    # A new store in the child is still the parent's runtime, so it is refused too.
    zarr.open_array(http_url, mode="r")[:8, :8]

    assert_refused_for_tokio(run_in_child(lambda: zarr.open_array(http_url, mode="r")[:]))


def test_an_unforked_process_can_still_use_a_remote_store(http_url) -> None:
    expected = np.arange(256 * 256, dtype="int16").reshape(256, 256)
    np.testing.assert_array_equal(np.asarray(zarr.open_array(http_url, mode="r")[:]), expected)
