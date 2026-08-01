"""Ablation matrix for the sparse-read benchmark.

Drop into ``~/projects/sparse-read-bench/scripts/`` alongside ``bench_sparse.py``.

Everything here is DATA. Adding or removing an ablation must never require
editing the runner.

Fixed sampling regime
---------------------
The sampling config is FIXED, not swept::

    chunk_size      = 1       pure random, every row drawn independently
    preload_nchunks = 8192    rows materialised per fetch
    batch_size      = 1024    rows handed to the consumer

``_in_memory_size = chunk_size * preload_nchunks = 8192`` rows per fetch, so
one fetch feeds **8 batches**.

*** THE RUNNER MUST ASSERT THIS. ***
Earlier runs used roughly one fetch per batch, which is a DIFFERENT workload:
it issues ~2,080 reads at a time instead of ~16,600, and this workload is
depth-limited, so the two are not comparable. Any result produced without
these exact values is invalid and must not be compared against one that was.

*** METHODOLOGY CONSEQUENCE - do not report a per-batch median. ***
Only 1 batch in 8 performs I/O; the other 7 slice an in-memory buffer. A
per-batch median would land on a cheap batch and understate cost by ~8x, and
a per-batch mean would hide the bimodality. Report **samples per second over
the whole run** (total rows / total wall time). That is invariant to how
fetches align with batches, which is exactly why it is the headline metric.

Read count at this regime (measured geometry: inner chunk 91,549 elements,
~1450 nnz/row, so a 1-row run spans 1 + 1450/91549 = 1.016 inner chunks)::

    per fetch:  8192 rows * 2 arrays * 1.016 = ~16,600 reads
    per batch:  ~2,080 reads (amortised)

Pure random is the most read-expensive regime possible: ~1 read per row per
array, and no coalescing is available because 1024 random rows out of 100.6M
essentially never share an inner chunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Fixed sampling regime. The runner asserts these; it does not sweep them.
# --------------------------------------------------------------------------

CHUNK_SIZE = 1
PRELOAD_NCHUNKS = 8192
BATCH_SIZE = 1024

ROWS_PER_FETCH = CHUNK_SIZE * PRELOAD_NCHUNKS  # 8192
BATCHES_PER_FETCH = ROWS_PER_FETCH // BATCH_SIZE  # 8

# Measured tahoe100_converted geometry.
NNZ_PER_ROW = 1450.0
INNER_CHUNK_ELEMENTS = 91_549
ARRAYS_PER_FETCH = 2  # indices + data


def reads_per_fetch() -> int:
    span = 1.0 + (CHUNK_SIZE * NNZ_PER_ROW) / INNER_CHUNK_ELEMENTS
    return int(ARRAYS_PER_FETCH * (ROWS_PER_FETCH // CHUNK_SIZE) * span)


def assert_sampling(actual: dict) -> None:
    """Refuse to run against a loader configured differently."""
    want = {
        "chunk_size": CHUNK_SIZE,
        "preload_nchunks": PRELOAD_NCHUNKS,
        "batch_size": BATCH_SIZE,
    }
    if {k: actual.get(k) for k in want} != want:
        raise RuntimeError(f"sampling config mismatch: want {want}, got {actual}")


# --------------------------------------------------------------------------
# Required versions. The runner MUST assert these and refuse to run otherwise.
# A silently-wrong version makes an arm measure something other than what its
# label claims -- which already happened here once and invalidated a sweep.
# --------------------------------------------------------------------------

REQUIRED_VERSIONS: dict[str, str] = {
    # Must contain zarr-python#4172 (CoordinateIndexer fast path). Note this
    # is INERT in the BASELINE, which never builds a CoordinateIndexer; it
    # only becomes a lever once the `vindex` ablation is on.
    "zarr": "latest",
    "anndata": "pin-and-record",
    "annbatch": "pin-and-record",
    "numpy": "pin-and-record",
    # Swappable input: a path or version, never hardcoded. Record its sha256
    # and the dirty-file list of the tree it was built from.
    "zarrs": "record-sha256-and-provenance",
}


# --------------------------------------------------------------------------
# Baseline: latest zarr-python + annbatch's NATIVE path (MultiBasicIndexer
# wrapping one BasicIndexer(slice) per row-run, utils.py:86, loader.py:855)
# + a pure-Python FD-cached store. At chunk_size=1 that is 1024 one-row
# slices per array, and it never constructs a CoordinateIndexer.
#
# NO zarrs wheel in the baseline. zarrs is an ablation on top of it.
# --------------------------------------------------------------------------

BASELINE: dict = {
    "codec_pipeline": "zarrs.ZarrsCodecPipeline",
    # Just a config. zarrs#422 added FilesystemStoreOptions::file_handle_cache_size
    # (default 0), merged 2026-07-20 and released in zarrs_filesystem 0.3.11.
    # It exists because the filesystem store otherwise opens, stats and closes
    # on every partial read; with sharded arrays issuing one call per inner
    # chunk that is 10-500x more open/stat/close syscalls on Lustre or NFS.
    # Do NOT reimplement this in Python -- it is a released upstream feature.
    #
    # Build note: this needs only RELEASED zarrs crates (zarrs_filesystem
    # >= 0.3.11). zarrs-python does not yet expose the option upstream; our
    # branch plumbs it through in 18 lines (src/store/filesystem.rs +
    # python/zarrs/pipeline.py) and that plumbing is independent of the local
    # zarrs path dependency, which exists only for the vindex multi-subset
    # API. So the baseline does not need the sibling zarrs checkout.
    "file_handle_cache_size": 512,
}
BASELINE_ANNBATCH_INTEGER_PATH = False


@dataclass(frozen=True)
class Ablation:
    """One matrix row: what changed, and which codebase it lives in."""

    name: str
    where: str  # zarrs-python | zarrs | zarr-python | annbatch | anndata | layout
    change: str
    config: dict = field(default_factory=dict)  # overrides on BASELINE
    integer_path: bool = False  # annbatch emits a coordinate selection
    requires: tuple[str, ...] = ()  # other ablations that must also be on
    landed: bool = True

    def resolved(self) -> dict:
        cfg = dict(BASELINE)
        for dep in self.requires:
            cfg.update(BY_NAME[dep].config)
        cfg.update(self.config)
        return cfg

    def resolved_integer_path(self) -> bool:
        return self.integer_path or any(
            BY_NAME[d].integer_path for d in self.requires
        )


# The vindex config is ONE TICK, not two knobs. It is an annbatch change and a
# zarrs-python change that only do anything together: the annbatch integer
# path produces a CoordinateIndexer, and the zarrs-python config is what acts
# on it. Measured separately, the zarrs-python half alone was 7946 ms against
# a 7931 ms baseline -- i.e. nothing. Splitting them in a matrix reports the
# vindex work as worthless, which is a measurement artefact, not a result.
VINDEX = Ablation(
    "vindex",
    where="annbatch + zarrs-python",
    change=(
        "TWO CHANGES AS ONE TICK. "
        "(annbatch) at chunk_size=1 emit an integer row array instead of "
        "1-row slices, so zarr builds a CoordinateIndexer -- this also makes "
        "zarr-python#4172 live. "
        "(zarrs-python) vindex_shard_index_cache_size=4096 (3,292 shards "
        "exist; the 256 default holds 8% and evicts in arbitrary hash order), "
        "vindex_io_concurrent_target=96 (= outstanding-read depth), "
        "vindex_decode_concurrent_target=48 (= codec thread budget). "
        "Neither half does anything without the other."
    ),
    config={
        "vindex_shard_index_cache_size": 4096,
        "vindex_io_concurrent_target": 96,
        "vindex_decode_concurrent_target": 48,
    },
    integer_path=True,
    requires=("zarrs",),
)


ABLATIONS: list[Ablation] = [
    Ablation(
        "baseline",
        where="-",
        change="latest zarr-python + annbatch native MultiBasicIndexer "
        "(slices) + zarrs pipeline with file_handle_cache_size=512. No vindex "
        "config. #4172 is inert here (no CoordinateIndexer is ever built).",
    ),
    VINDEX,
    # ---- depth sweep, all on top of the vindex tick -----------------------
    Ablation(
        "vindex_io32",
        where="zarrs-python",
        change="vindex_io_concurrent_target 96 -> 32.",
        config={"vindex_io_concurrent_target": 32},
        requires=("vindex",),
    ),
    Ablation(
        "vindex_io192",
        where="zarrs-python",
        change="vindex_io_concurrent_target 96 -> 192. Measured client "
        "ceiling was ~2455 preads/s at 64 threads and REGRESSED to 2090 at "
        "96, so expect a turnover. More relevant at preload 8192, which makes "
        "~16,600 reads available at once.",
        config={"vindex_io_concurrent_target": 192},
        requires=("vindex",),
    ),
    Ablation(
        "vindex_io384",
        where="zarrs-python",
        change="vindex_io_concurrent_target 96 -> 384. Lustre allows 16 rpcs "
        "x 50 OSTs; finds whether the client or the server binds.",
        config={"vindex_io_concurrent_target": 384},
        requires=("vindex",),
    ),
    Ablation(
        "vindex_no_index_cache",
        where="zarrs-python",
        change="vindex_shard_index_cache_size 4096 -> 0. Isolates what 48.4 "
        "MiB of cached shard indexes is worth.",
        config={"vindex_shard_index_cache_size": 0},
        requires=("vindex",),
    ),
    # ---- applies to baseline and vindex alike -----------------------------
    Ablation(
        "no_fd_cache",
        where="zarrs-python",
        change="file_handle_cache_size 512 -> 0, i.e. back to open/stat/close "
        "per partial read. Isolates what the metadata syscalls cost on Lustre.",
        config={"file_handle_cache_size": 0},
    ),
    Ablation(
        "direct_io",
        where="zarrs-python",
        change="O_DIRECT on. Measured NEGATIVE at 96-way (1024 vs 953 ms) but "
        "4.7x at low concurrency, so it may return if depth drops.",
        config={"direct_io": True},
    ),
    # ---- not yet implemented; declared so the matrix shows the gap --------
    Ablation(
        "ab_sorted_runs",
        where="annbatch",
        change="Sort row-runs before issuing. The stable argsort at "
        "loader.py:572 preserves shuffled order, so reads go out in random "
        "order and sequential locality is discarded.",
        landed=False,
    ),
    Ablation(
        "ab_csr_dataset",
        where="annbatch -> anndata",
        change="Route through anndata CSRDataset instead of the hand-rolled "
        "MultiBasicIndexer. Gets coordinate selection and indptr caching for "
        "free. Risk: may not support the direct out= buffer write.",
        landed=False,
    ),
    Ablation(
        "ad_concat_ranges",
        where="anndata",
        change="Vectorised _concat_ranges in get_compressed_vectors, "
        "replacing a per-row Python loop of slice+arange+concatenate. ~1% at "
        "1024 rows.",
        landed=False,
    ),
    Ablation(
        "layout_interleaved",
        where="layout",
        change="Interleave indices+data into one array: halves read count in "
        "every regime. Requires rewriting the dataset.",
        landed=False,
    ),
]

BY_NAME: dict[str, Ablation] = {a.name: a for a in ABLATIONS}


def runnable() -> list[Ablation]:
    return [a for a in ABLATIONS if a.landed]


def pending() -> list[Ablation]:
    return [a for a in ABLATIONS if not a.landed]


if __name__ == "__main__":
    print(
        f"sampling FIXED: chunk_size={CHUNK_SIZE} "
        f"preload_nchunks={PRELOAD_NCHUNKS} batch_size={BATCH_SIZE}"
    )
    print(
        f"  -> {ROWS_PER_FETCH} rows/fetch, {BATCHES_PER_FETCH} batches/fetch, "
        f"~{reads_per_fetch():,} reads/fetch"
    )
    print("  -> report SAMPLES/SECOND, not a per-batch median\n")
    print(f"{'ablation':<24} {'where':<24} {'int-path':<9} landed")
    for a in ABLATIONS:
        print(
            f"{a.name:<24} {a.where:<24} "
            f"{'yes' if a.resolved_integer_path() else '-':<9} "
            f"{'yes' if a.landed else 'NO'}"
        )
    print(f"\nrunnable: {len(runnable())}   pending: {len(pending())}")
