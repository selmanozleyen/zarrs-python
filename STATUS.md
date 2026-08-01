# Status: sparse 1-D vindex read path

Supersedes the measurement claims in `HANDOFF.md`. Where the two disagree, this
file wins — several of the handoff's numbers were produced by harnesses or
metrics that have since been shown to be invalid.

## Repo state

Nothing committed, nothing pushed. Preserve all dirty files in both trees.

| repo | branch | HEAD | state |
|---|---|---|---|
| `~/projects/zarrs-python` | `feat/native-1d-vindex-sparse` | `02cef17` | dirty; new `src/vindex_stats.rs`, `src/store/concurrent_partial.rs` |
| `~/projects/zarrs` | `codex/partial-decode-subsets` | `49401927` | dirty; multi-subset + Blosc capability changes |
| `~/projects/anndata` | — | `a0c42837` | clean |
| `~/projects/annbatch` | — | — | clean |
| cluster `~/projects/sparse-read-bench` | — | `3d3720c` | benchmark harness + skill, committed locally only |

## Headline result — all 14 plates, one exclusive `cpu_96` node per arm

| arm | what | cold ms | later median | avg cores | peak RSS | max FDs |
|---|---|---:|---:|---:|---:|---:|
| A | annbatch main + stock zarr pipeline | 7021 | 6161 | 1.65 | 16.3 GiB | 24 |
| B | + zarrs pipeline, FD cache 512 | 10850 | 7931 | 0.84 | 16.3 GiB | 3248 |
| C0 | + full vindex config, annbatch main | 9547 | 7946 | 1.18 | 16.4 GiB | 3252 |
| **C** | **+ annbatch integer-fetch** | **1396** | **953** | 5.61 | **3.2 GiB** | 3252 |
| D | C + `ZARRS_VINDEX_STATS=1` | 1237 | 930 | 3.83 | 3.2 GiB | 3240 |
| E | C + `direct_io=True` | 1462 | 1024 | 5.79 | 3.2 GiB | 104 |

**C is 6.5x faster than baseline** on the later median and uses 5x less RSS.

Caveat: 8 batches per arm, entropy seeds, visible spread. Treat B vs C0 as a
tie and C/D/E as within ~10%. The A→C gap is far outside that noise.

## The B/C0 regression — root-caused, fixed

B and C0 were **slower than stock zarr-python**. This is a real bug in this
branch, not a harness artifact.

`src/lib.rs` wrapped the store in the concurrent-I/O adapter *unconditionally in
the constructor*, so that store backed every read path. annbatch main emits
`MultiBasicIndexer(BasicIndexer(slice))` and never reaches the vindex fast path,
so it paid the rayon cross-pool `install` handoff on every partial read while
getting none of the batching that is supposed to amortise it.

Corroborated by CPU: B used **0.84 average cores vs A's 1.65** — less CPU-busy,
i.e. blocking more. A measurement error would not systematically halve CPU.

**Fix applied:** the pipeline now holds both `store` (raw) and `vindex_store`
(wrapped). The path is decided before partial decoders are constructed, and only
the scattered path builds decoders from the wrapped store. Tests green
(5586 passed, 5 skipped, 36 xfailed).

**Not yet re-measured on the cluster.** Re-run B and C0 to confirm they return to
parity with A.

Known remaining subtlety: `partial_decoder_cache` is keyed by `StoreKey` only, so
an array read via both paths could reuse a decoder built from the other store.
Correct either way, but perf-relevant. Unusual in practice; not addressed.

## Instrumentation

New `src/vindex_stats.rs`. Enabled by `ZARRS_VINDEX_STATS`; behaviour-neutral when
unset (one cached bool; the submit-side `Instant::now()` is gated too).

Splits the old single summed `partial_decode_task_ms` into plan / decoder build /
`io[index]` / `io[payload]` (queue, call, blocked, wait, log2-µs histogram, ranges,
bytes) / decode / codec estimate / scatter. I/O is attributed to the blocking
thread via a thread-local phase scope, so index and payload I/O are separated and
concurrently-reading arrays do not pool counters.

### The nesting discovery

`rayon::ThreadPool::install` called from a worker of *another* pool does **not**
park the caller — rayon runs other pending jobs from the caller's own pool on that
thread. The outer task's timer then swallows the entire nested task it stole.

Measured directly with a nesting counter:

| config | `nest_depth_max` | `inflight_max` | `execute_wall_ms` | `partial_decode_task_ms` |
|---|---:|---:|---:|---:|
| io=96, decode=**1** | **31** | 31 | 21.6 | 164.2 |
| io=96, decode=**48** | **32** | 100 | 7.2 | 345.0 |

Consequences:

1. **`vindex_decode_concurrent_target` does not bound concurrency.** Set to 1, it
   still ran 31 concurrent reads. The handoff's decode-target sweep
   ("48 is best") did not measure what it appears to; do not carry that number
   forward as established.
2. **Summed task metrics are uninterpretable** under nesting. The report emits a
   WARNING naming which fields to trust (`execute_wall_ms`, `queue_ms`, `call_ms`,
   `scatter_ms`, counts) and which double-count.
3. **Unbounded recursion depth** is a latent stack-overflow risk.

The warning fired in production on the cluster (arm D): `nested 7 deep`,
`max_active_partial_reads=96` against a decode target of 48.

## Actual dataset geometry (measured, not assumed)

Path: `/ictstr01/groups/ml01/datasets/selman.ozleyen/tahoe100_converted`.
(`/projects/cf-train` does not exist; `cf-train` is `~/projects/cf-train`.)

- **14 plates, non-uniform**: 4.7M–10.5M rows each, not 7,418,000 uniformly
- **100,648,790 rows**, 62,710 cols, density 2.28%
- **145,997,980,995 nnz**, ~1450 nnz/row
- Zarr v3, `sharding_indexed -> [bytes, blosc]`
- dtypes: `data float32`, **`indices uint16`**, `indptr int64`
- **inner chunk 91,549 elements; shard 89,351,824 elements; 976 inner chunks/shard**
- **shard ~97 MiB, NOT 1 GB**
- **3,292 shards** (1,639 data + 1,639 indices + 14 indptr)
- **shard index = 15,620 B/shard; 48.4 MiB for the entire collection**
- **311.8 GiB / 334.8 GB compressed**

**`indices` (179 GiB, 1.51x) is larger than `data` (133 GiB, 4.14x).** The
uint16 index array compresses badly and dominates I/O. Start re-chunking there.

## Lustre

`read_ahead_stats` is debugfs and permission-denied unprivileged, so it was
measured in user space instead.

- `read_ahead_range_kb=1024`, `max_read_ahead_per_file_mb=128`
- **Readahead amplification confirmed: ~10x**, ~900 KiB wasted per ~100 KiB read
- **But the store is latency-bound, not bandwidth-bound.** Single-request latency
  ~21 ms either way. Concurrency is worth ~50x (46 → 2455 preads/s, 1 → 64
  threads); O_DIRECT is worth 4.7x at low concurrency, ~1.4x at 96.
- Arm E confirms: `direct_io` did not pay off. Keep buffered, keep concurrency high.
- **PFL striping**: `0–1 GiB: stripe_count 1`, `1–4 GiB: 4`, `4 GiB–EOF: 10`,
  `stripe_size 1 MiB`. A ~1 GB shard lives on **one OST of 50**, capped by that
  OST's `max_rpcs_in_flight=16`. Parallelism must come from touching many shards.
  Do not grow shards past 1 GiB or striping changes underneath you.
- 50 OSTs, 18.5 PiB, 82% full, pool `ddn_hdd` (spinning).

This **contradicts** the roofline model's bandwidth-bound conclusion, which
assumed 2 GB/s client bandwidth and said outright it should be measured first.

## Corrections to earlier analysis

| claim | status |
|---|---|
| Pin indptr in memory — it is re-read every batch | **Wrong for this stack.** annbatch reads indptr in full once on first `__iter__` (`loader.py:761-784`) and caches it — zero indptr reads per fetch. anndata caches it too, `should_cache_indptr=True` by default (`sparse_dataset.py:539-548`). Already done. |
| anndata needs a vindex path added for CSR zarr | **Already exists.** The zarr branch issues `get_coordinate_selection` and reaches `CoordinateIndexer` today (`sparse_dataset.py:169-186`); only the h5py branch is per-row slices. No fork needed. |
| Shard index ~140 KB each, ~300 shards | **Wrong factors, right total.** Actually 15,620 B × 3,292 = 48.4 MiB. Conclusion survives. |
| Shards are ~1 GB | ~97 MiB. |
| Cross-pool handoff is ~7x the storage call | **Wrong.** Mostly nested stolen work, not handoff. See nesting section. |
| Bandwidth-bound | **Latency-bound.** Measured. |
| Decode target 48 is tuned | The knob does not bound anything. Unexplained artifact. |

## Config that matters

- **`vindex_shard_index_cache_size`: default 256 holds only 8% of 3,292 shards.
  Set 4096** (costs ~48 MiB). Arm D steady state: `cache_hits=63 cache_misses=0
  build_ms=0.029` vs `build_ms=727` on a miss.
- **`file_handle_cache_size` is per zarr array, not per process.** 14 plates × 3
  arrays × 512 = up to 21,504 FDs; blew `RLIMIT_NOFILE=1024` and killed arms
  B/C0/C/D on first submission. Library should bound globally or document.

## annbatch fetching (measured, commit `60c1de9`)

annbatch **bypasses anndata entirely** (`_subset` appears only in a comment,
`loader.py:847`).

- Defaults (`loader.py:170`): `chunk_size=512, preload_nchunks=32, batch_size=1`,
  `shuffle=None` → `SequentialSampler`
- `chunk_size` = row extent of each contiguous slice; `preload_nchunks` = slices
  per round trip; their product (default 16384 rows) is the buffer materialised
- **`batch_size` has zero effect on I/O** — it only re-slices the in-memory buffer
- **`shuffle` shuffles the order of chunk slices, never producing per-row random
  disk requests**
- Per fetch: **2 × k zarr reads** (`data` + `indices`), k = datasets hit.
  **Zero indptr reads.**
- Concurrency: asyncio on zarr's single `zarr_io` thread, `asyncio.gather` across
  datasets → across data/indices → zarr `concurrent_map` at
  `async.concurrency=10`. **No prefetch**: fetch n+1 does not start until batch n
  is consumed.
- Call site: `data._get_selection(indexer, ...)` at `loader.py:870/875` with a
  hand-rolled `MultiBasicIndexer` (`utils.py:86-101`) — **a list of basic slices,
  never fancy indexing, never a coordinate selection**, and not sorted.

This is exactly why C0 shows no benefit and why the integer-fetch branch
(`_use_integer_fetch_rows`, `loader.py:55/900`) is what unlocks the fast path.

## Tricks the benchmark agent applied

Methodology:

1. **Added control arm C0** (not in the spec) — zarrs pipeline + full vindex
   config but annbatch *main*. Without it, the C-vs-B gap would have been
   misattributed to the zarrs build rather than to annbatch's integer fetch.
   This is the single most valuable thing it did.
2. **Timed from before the first `next(it)`** — a previous harness called
   `next(it)` before starting the timer, warming the shard data and invalidating
   its own "cold" numbers.
3. **Cold first batch reported separately** from the median of later batches.
4. **Entropy seeds logged but never fixed** — replayable for debugging, never the
   only measurement.
5. **Substituted a user-space measurement** when `lctl read_ahead_stats` turned
   out to be permission-denied: buffered vs O_DIRECT preads/s across request
   sizes (4 KiB–1 MiB) and thread counts (1–96). Answered the same question and
   additionally produced the latency-vs-bandwidth verdict.
6. **Tested its own prediction as an arm** — the probe suggested O_DIRECT, so it
   added arm E; the prediction failed and it reported that.
7. **Probed geometry from actual store metadata** rather than trusting the stated
   14 × 7.418M / 1 GB shards, and found both wrong.

Environment:

8. **Node-local `UV_CACHE_DIR`** under `/localscratch`, per user instruction.
9. **Committed `uv.lock`, verified with `uv sync --frozen`** in a fresh venv on a
   compute node, cold and warm.
10. **`uv pip install --no-deps` to overlay only the code under test**, so the
    resolved dependency set is byte-identical across all arms.
11. **`vendor/PROVENANCE.txt`** recording branch, HEAD and the dirty-file list of
    both dev trees the wheel was built from, plus the wheel's sha256.
12. **`ulimit -n` + `resource.setrlimit`** to survive the per-array FD cache
    blowout, after diagnosing it from the first failed submission.
13. **Per-job RSS, FD count, CPU cores, context switches** recorded alongside
    timings — which is what made the B regression diagnosable.

## Open items

1. **Re-measure B and C0** after the store-split fix — confirm parity with A.
2. **Replace the rayon cross-pool `install`** with a mechanism that genuinely
   parks (dedicated I/O workers + crossbeam channel). Prerequisite for
   trustworthy measurement, not just an optimization.
3. **Sampler**: move to `RandomSampler` with `seed=None`; add a `chunk_size=1`
   path that emits integer row lists on the annbatch side.
4. `vindex_shard_index_cache_size` → 4096.
5. Bound `file_handle_cache_size` globally.
6. Re-chunk `indices` (uint16, 1.51x, 179 GiB) — largest layout lever.
7. Evaluate the V1 `stream_rows` streaming API.
