"""Ablation matrix for the sparse-read benchmark.

Drop into ``~/projects/sparse-read-bench/scripts/`` alongside ``bench_sparse.py``.

Everything here is DATA. Adding or removing an ablation must never require
editing the runner.

Fixed sampling regime
---------------------
The sampling config is FIXED, not swept::

    chunk_size      = 1       pure random, every row drawn independently
    preload_nchunks = 1024    rows materialised per fetch
    batch_size      = 1024    rows handed to the consumer

``_in_memory_size = chunk_size * preload_nchunks = 1024`` rows per fetch, so
**one fetch is exactly one batch**. Every batch performs I/O, which is what
makes per-batch numbers interpretable.

*** THE RUNNER MUST ASSERT THIS. ***
A previous regime used preload_nchunks=8192, where one fetch fed 8 batches
and only 1 batch in 8 did any I/O. Per-batch figures from that regime are
not comparable to these. samples/second IS comparable across both, because
it is invariant to how fetches align with batches -- which is why it stays
the headline metric even now that per-batch numbers mean something.

Read count at this regime (measured geometry: inner chunk 91,549 elements,
~1450 nnz/row, so a 1-row run spans 1 + 1450/91549 = 1.016 inner chunks)::

    per fetch = per batch:  1024 rows * 2 arrays * 1.016 = ~2,080 reads

Depth note: 2,080 reads are available at once, against a baseline measured
to be concurrency-limited at ~16 and a Lustre ceiling near 2,455 preads/s.
So this regime still offers far more outstanding work than any arm can
currently consume, and the reduction from ~16,600 should not itself bind.

Pure random is the most read-expensive regime possible: ~1 read per row per
array, and no coalescing is available because 1024 random rows out of 100.6M
essentially never share an inner chunk.

What the vindex arms actually vary
----------------------------------
``vindex`` is one tick spanning two codebases, because neither half does
anything alone:

* **annbatch** emits an integer row array at ``chunk_size=1`` instead of 1024
  one-row ``slice`` objects, so zarr constructs a ``CoordinateIndexer``.
* **zarrs-python** recognises that and takes its native 1-D sparse path:
  sort and group-run the coordinates, group every sparse subset by
  ``StoreKey`` (i.e. by physical shard), then one task per shard. Each task
  issues ONE logical multi-range read for its shard, maps the subsets to
  inner chunks, groups byte ranges per inner chunk, and decodes and scatters
  straight into a single contiguous output arena. The output permutation is
  retained so results land back in the caller's order.

Without the annbatch half there is no ``CoordinateIndexer`` and the whole
path is dead code -- measured at 7946 ms against a 7931 ms baseline.

Depth, and its ceiling
----------------------
Shard **indices** are cached (48.4 MiB, all resident). Shard **payload** is
312 GiB and never can be: every batch still pulls ~205 MiB of compressed
inner chunks off Lustre. ``vindex_io_concurrent_target`` governs PAYLOAD
reads in flight; caching indices does not touch that.

Ranges *within* one shard are issued as a single ``get_partial_many`` and
looped sequentially by the store. So outstanding reads equal concurrent
SHARD tasks, not concurrent ranges::

    1024 rows x 2 arrays = 2048 runs -> ~234 shard tasks x ~9 ranges each
    outstanding reads ~= 234 (max), NOT ~2,080

That is the ceiling: past ~234 there is nothing left to schedule, so
``io384`` cannot help by construction.

Why group by shard at all? Only so the store could reuse one open file and
loop preads over it. The FD cache (file_handle_cache_size) already provides
handle reuse, so that rationale is now largely vestigial -- and the grouping
is what imposes the ceiling.

The flat design removes it. Because shard indices are cached in memory, the
complete read plan is known BEFORE any payload I/O: a flat list of ~2,080
``(file, offset, length)``, one per needed inner chunk, each of which
decompresses independently. So::

    issue all ~2,080 reads concurrently
        -> as each completes, decompress that chunk
        -> scatter its elements into the output

Shard grouping is not needed for correctness, and this lifts outstanding
reads from ~234 to ~2,080.

The obstacle is zarrs' abstraction, not the idea: ``partial_decode_subsets``
runs on a PER-SHARD partial decoder that owns its shard index and does the
mapping internally. Flattening across shards requires zarrs to expose the
seam -- "which byte ranges do I need" separately from "here are the bytes,
now decode". That is the zarrs-side change, and this is the argument for it.

Decompression threading is NOT the lever. Blosc can decompress a single
buffer with multiple threads, but each inner chunk is ~360 KB raw,
``blosc_getitem`` touches roughly one block, and total codec CPU measured
~4 ms per batch against ~2,000 ms of wall. Parallelism belongs across the
~2,080 chunks, not within one.

Getting to ~2,080 outstanding means parallelising ranges within a shard.
Thread-per-range WAS tried and lost badly -- ~340 threads, 50k+ voluntary
context switches, slower than the batched control at every target up to 96.
But that was measured under the cross-pool rayon scheduler since shown to be
pathological (nesting 31 deep, no target bounding anything), so the verdict
is not safe and deserves a re-test. The principled version is io_uring:
2,080 outstanding reads without 2,080 threads.

``vindex_io32`` / ``io192`` / ``io384`` vary ``vindex_io_concurrent_target``.
Since the scheduler rework this sizes the SINGLE shared pool, and each shard
task performs its own read inline on its own thread, so this value **is the
outstanding-read depth** -- how many shard reads are in flight at once. The
sweep exists to find where the client stops rewarding depth: the baseline
measured concurrency-limited at ~16 (~790 reads/s, matching 16 / 21 ms), the
Lustre client ceiling was ~2455 preads/s at 64 threads but REGRESSED to 2090
at 96, and Lustre itself would allow 16 rpcs x 50 OSTs. So a turnover is
expected somewhere in this range; locating it is the point.

Shard-index caching. The option defaults to **0, i.e. off**, because the
cache cannot detect external mutation. Each index is 15,620 B; there are
3,292 shards in the collection but the cache is **per zarr array**, and with
1,639 data + 1,639 indices shards across 14 plates that is ~117 shards per
array. So ``4096`` is ~35x the working set: every index resident, and 48.4
MiB would hold all 3,292 even if one cache held the lot.

Because the BASELINE leaves it at 0, it rebuilds partial decoders and
re-reads shard indexes every batch (``build_ms=727`` on a miss versus
``0.029`` on a hit). ``vindex`` therefore bundles two wins -- the sparse path
and the cache -- which is why ``baseline_index_cache`` exists: it turns the
cache on WITHOUT the integer path, so the two can be separated instead of
reported as one lump.

``vindex_index_preload`` (not implemented) would go further and read every
shard index at open rather than lazily on first touch, so no batch pays a
miss at all. It targets the cold batch specifically: 2,108-5,259 ms cold
against 1,986-2,440 ms steady.

How annbatch should ideally drive this
--------------------------------------
1. At ``chunk_size=1``, emit an integer row array, not one-row slices.
2. **Sort** those indices and keep the inverse permutation, scattering
   results back into shuffled order afterwards. Sorting is what lets zarr's
   #4172 fast path fire at all, and it gives zarrs-python longer runs and
   tighter per-shard grouping. zarrs-python already retains an output
   permutation, so the two compose.
3. Keep ``indptr`` resident (already the default in both annbatch and
   anndata) -- it is the one dense access in this workload.
4. Ideally hand the whole preload window over at once, so shard grouping and
   shard-index reuse amortise across more rows than a single batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Fixed sampling regime. The runner asserts these; it does not sweep them.
# --------------------------------------------------------------------------

CHUNK_SIZE = 1
PRELOAD_NCHUNKS = 1024
BATCH_SIZE = 1024

ROWS_PER_FETCH = CHUNK_SIZE * PRELOAD_NCHUNKS  # 1024
BATCHES_PER_FETCH = ROWS_PER_FETCH // BATCH_SIZE  # 1

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
# The zarrs pipeline IS in the baseline -- it is how file_handle_cache_size
# is reached. What the baseline does not have is any vindex configuration.
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
VINDEX_NAME = "vindex (sorted+shard-grouped, shard-indices cached, fd cached)"
VINDEX = Ablation(
    VINDEX_NAME,
    where="annbatch + zarrs-python",
    change=(
        "EVERY TRICK ON. One tick spanning two codebases; neither half does "
        "anything alone. "
        "(annbatch) emit an integer row array at chunk_size=1 instead of 1024 "
        "one-row slices, so zarr builds a CoordinateIndexer -- this is also "
        "what makes zarr-python#4172 live. "
        "(zarrs-python) sort and group-run the coordinates; group every "
        "sparse subset by StoreKey, i.e. by physical shard; one task per "
        "shard; ONE logical multi-range read per shard via get_partial_many; "
        "map subsets to inner chunks and group byte ranges per inner chunk; "
        "decode and scatter straight into a single contiguous output arena, "
        "avoiding one Vec per subset plus a final flattening copy; retain the "
        "output permutation so results land in caller order. "
        "(caches) shard-index/partial-decoder cache 4096 = every index "
        "resident (~117 shards per array, 48.4 MiB for all 3,292); file "
        "handle cache 512 inherited from the baseline. "
        "(scheduling) reads run inline on one shared pool sized by "
        "vindex_io_concurrent_target, after the cross-pool rayon install was "
        "removed. "
        "(codec) Blosc partial decode via blosc_getitem. "
        "NOTE the depth ceiling: ranges WITHIN a shard are issued as one "
        "get_partial_many and looped sequentially by the store, so "
        "outstanding reads equal concurrent SHARD tasks (~234 here), not "
        "concurrent ranges (~2,080)."
    ),
    config={
        "vindex_shard_index_cache_size": 4096,
        "vindex_io_concurrent_target": 96,
        "vindex_decode_concurrent_target": 48,
    },
    integer_path=True,
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
        requires=(VINDEX_NAME,),
    ),
    Ablation(
        "vindex_io192",
        where="zarrs-python",
        change="vindex_io_concurrent_target 96 -> 192. Measured client "
        "ceiling was ~2455 preads/s at 64 threads and REGRESSED to 2090 at "
        "96, so expect a turnover. More relevant at preload 8192, which makes "
        "~16,600 reads available at once.",
        config={"vindex_io_concurrent_target": 192},
        requires=(VINDEX_NAME,),
    ),
    Ablation(
        "vindex_io384",
        where="zarrs-python",
        change="vindex_io_concurrent_target 96 -> 384. Lustre allows 16 rpcs "
        "x 50 OSTs; finds whether the client or the server binds.",
        config={"vindex_io_concurrent_target": 384},
        requires=(VINDEX_NAME,),
    ),
    # ---- not yet implemented; declared so the matrix shows the gap --------
    Ablation(
        "vindex_index_preload",
        where="zarrs-python",
        change="Eagerly read every shard index at array open instead of "
        "populating the cache lazily on first touch, so no batch ever pays an "
        "index miss. 48.4 MiB for the entire collection. Targets the cold "
        "batch specifically: measured 2,108-5,259 ms cold against 1,986-2,440 "
        "ms steady. NOT IMPLEMENTED -- needs a new zarrs-python option.",
        config={"vindex_shard_index_cache_size": 4096},
        requires=(VINDEX_NAME,),
        landed=False,
    ),
    Ablation(
        "ab_sorted_runs",
        where="annbatch",
        change="Sort the selected row indices before handing them to zarr, "
        "keeping the inverse permutation to scatter results back into the "
        "shuffled order the consumer expects. Unlocks TWO things: (a) zarr's "
        "#4172 CoordinateIndexer fast path requires a SORTED 1-D integer "
        "selection and falls back otherwise, and (b) zarrs-python's planner "
        "group-runs the coordinates anyway, so sorted input yields longer "
        "runs and tighter per-shard grouping. Today the stable argsort at "
        "loader.py:572 preserves shuffled chunk order, so what reaches zarr "
        "is unsorted and neither fires.",
        requires=(VINDEX_NAME,),
        landed=False,
    ),
    Ablation(
        "ab_csr_dataset",
        where="annbatch -> anndata",
        change="Route through anndata CSRDataset instead of the hand-rolled "
        "MultiBasicIndexer. Gets coordinate selection and indptr caching for "
        "free. BLOCKED ON VINDEX: CSRDataset issues coordinate selections, "
        "which only pay off once the vindex path is working and measured, so "
        "this is downstream of having vindex_* numbers -- not a parallel "
        "experiment. Risk: may not support the direct out= buffer write.",
        requires=(VINDEX_NAME,),
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
    print("  -> 1 fetch = 1 batch; per-batch numbers are meaningful\n")
    print(f"{'ablation':<24} {'where':<24} {'int-path':<9} landed")
    for a in ABLATIONS:
        print(
            f"{a.name:<24} {a.where:<24} "
            f"{'yes' if a.resolved_integer_path() else '-':<9} "
            f"{'yes' if a.landed else 'NO'}"
        )
    print(f"\nrunnable: {len(runnable())}   pending: {len(pending())}")
