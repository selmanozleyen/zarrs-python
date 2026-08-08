# Shard index caching

Two caches, both about not re-reading a shard index that never changes. One is
built and unmeasured; the other is unbuilt and has the larger measured case for
it. They are independent — either can ship alone.

---

## Where the fast path stands

Measured against **zarr-python**, the baseline a user actually has, at three
batch sizes. Two runs per cell, arms shuffled within each round, sorted rows,
`fd=512`.

| arm | 1024 | 4096 | 9192 |
|---|---|---|---|
| zarr-python | 150.8 · 185.8 | 401.2 · 430.1 | 483.7 · 477.3 |
| ours, no pool | 300.8 · 726.8 | 1914.3 · 2383.9 | 2997.2 · 3170.4 |
| ours + pool 128 | 1409.6 · 1283.8 | 2345.1 · 2570.1 | 3423.6 · 2597.7 |

Speedup over zarr-python, and what the pool itself adds:

| | 1024 | 4096 | 9192 |
|---|---|---|---|
| ours, no pool | 3.05× | 5.17× | **6.42×** |
| ours + pool | **8.00×** | 5.91× | 6.27× |
| pool's own contribution | 2.62× | 1.14× | **0.98×** |

**The pool contributes nothing at realistic batch sizes.** At 9192 the pooled
arm (2597.7–3423.6) brackets the unpooled one (2997.2–3170.4) and its mean is
lower. Same decay as the decoder cache.

**What grows is `split_1d_runs`** — 3.05× → 5.17× → 6.42×, rising with batch
size, because zarr-python barely scales (168 → 416 → 480 rows/s) while a path
that reaches Rust does.

Every "6.9×" quoted earlier in this work was pool-vs-our-own-unpooled-path,
which attributes the whole gain to the pool. Against the real baseline at the
batch sizes annbatch uses, the pool is worth nothing and the selection fix is
worth 6.4×.

That matters for the zarrs side specifically: `read_plan`,
`partial_decode_from_bytes` and `ArrayPartialDecoderPlanned` exist to enable
the pool. If the pool does not pay here, that API has no justification from
*this* workload — the measured win is ~180 lines of Python. It may still be
right for higher-latency stores or small batches, but that is unmeasured.

Caveat: two replicates on a shared node. The 9192 zarr baseline is tight
(483.7 / 477.3) which is what makes that column trustworthy; the 1024
no-pool cell spans 2.4× between identical configs and is unusable.

**zarrs — `perf/sharding-read-plan`** (in `../zarrs`)

Only one commit is required, touching two files:

```
zarrs_codec/src/codec_traits/array_partial_sync.rs   +54    as_planned, ArrayPartialDecoderPlanned
.../sharding/sharding_partial_decoder_sync.rs       +224    read_plan, partial_decode_from_bytes
```

The file handle cache needs **no zarrs change** — `FilesystemStoreOptions::file_handle_cache_size`
is already upstream (`edb5f735`, zarrs#422), defaulted to `0`. Only the
zarrs-python side that exposes it is outstanding.


| commit | needed for the speedup |
|---|---|
| `1db2c3ef` cover nested sharding | no — test coverage |
| `414c40dc` expose read plan | **yes** |
| `3773412d` cache inner shard indexes | no — nested only, never runs on flat data |
| `aec7be4d`/`5f514ce9`/`df833c7d` eager prefetch | no — added, measured null, removed |
| `a54f86ff` name it decode from bytes | rename |
| `59560425` gate planning behind a capability | API shape |

**zarrs-python — `perf/vindex-fetch-pool`**

| commit | needed for the speedup |
|---|---|
| `7a7b2e1` build against zarrs 0.24 | infrastructure |
| `0897ad9` pool planned chunk fetches | **yes — this is the 6.9×** |
| ~~`325f198` expose file handle cache~~ | superseded by upstream `85aa038` (#181) |
| `158a005` split 1-D fancy runs | **yes — without it 2 of 3 reads never reach Rust** |
| `7ca63c2` decline to shatter short runs | **yes — the guard, and the sorted/slice case** |
| `0eb90aa`/`ae94efe`/`6f8c904` bench | no |
| `29ba23a`/`12e0a19` eager plumbing | no — cancels out |
| `3e9a25d`/`fb4c7db` rename, capability | follow zarrs |

Measured, 32 cores, sorted rows, `fd=512`, 3 replicates per depth, shuffled
order, random seed per run:

| fetch threads | rows/s | vs no pool |
|---|---:|---:|
| 0 | 98.7 | — |
| 32 | 464.2 | 4.7× |
| **128** | **682.7** | **6.9×** |
| 256 | 599.5 | 6.1× |

4× the cores buys ~14%, so the workload is I/O-latency-bound. That is the
premise both caches below rest on.

---

## Task 1 — outer shard index cache (BUILT — helps only at small batches)

### The problem

`partial_decoder_cache` is a **local** in `retrieve_chunks_and_apply_index`
([src/lib.rs](src/lib.rs)):

```rust
let mut partial_decoder_cache: HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>> =
    HashMap::new();
```

It dies with the call. Every batch rebuilds a decoder per touched shard, and
`ShardingPartialDecoder::new` reads that shard's index from storage each time.

### Measured, tahoe, 1024 sorted rows over 4 datasets

| batch | shards touched | already seen | index re-read |
|---|---:|---:|---:|
| 0 | 814 | 0 | 0.0 MiB |
| 1 | 836 | 706 | 9.9 MiB |
| 2 | 832 | 814 | 11.4 MiB |
| 3 | 818 | **818** | **11.5 MiB** |
| 4 | 836 | **836** | **11.7 MiB** |

**In steady state every index read is a re-read.** The reason is the
denominator: only **962 distinct shards exist in the whole collection**, and
each index is 920 inner chunks × 16 B = **14.4 KiB**. The entire working set is
**13.5 MiB**.

Reproduce with `bench/probe_index_reads.py`.

### The change

Hoist the cache to a field on `CodecPipelineImpl`, bounded by the
`size_held()` that already exists on `ArrayPartialDecoderTraits` and is
currently unused — it reports exactly this:

```rust
fn size_held(&self) -> usize {
    self.input_handle.size_held()
        + self.shard_index.as_ref().map_or(0, Vec::len) * size_of::<u64>()
}
```

### The actual work is invalidation, not caching

A cached decoder holds a shard index that a write invalidates. Whichever of
these is chosen has to be stated in the PR:

- evict on `store_chunks_with_indices` for the keys it touches — simplest, and
  wrong only if something writes behind zarrs-python's back
- scope the cache to read-only arrays
- version it against the store

Also needs: a size bound and eviction policy (LRU over `size_held()`), and
thread safety, since `retrieve_chunks_and_apply_index` builds decoders under
`iter_concurrent_limit!`.

### Status: built, opt-in via `ZARRS_PYTHON_DECODER_CACHE`

`decoder_cache` is a field on `CodecPipelineImpl`; `store_chunks_with_indices`
evicts every key it touches before writing, and
`test_decoder_cache_survives_a_write` pins that.

Measured on tahoe, sorted 1024 rows, `fetch=128`, 4 shuffled rounds on one
allocation:

| | rows/s |
|---|---|
| off | 839.0 · 894.1 · 915.1 · 964.7 → median **904.6** |
| on | 938.8 · 1021.3 · 1040.6 · 1067.5 → median **1030.9** |

**+14% at 1024 rows**, winning every round including the one where it ran
first. But it does not survive larger batches:

| batch | cache effect | throughput |
|---:|---|---:|
| 1024 | **+14%**, won 4/4 rounds | ~950 rows/s |
| 4096 | ~0%, median +2% and mean -2% | ~1900 rows/s |
| 9192 | ~0%, off won 3/4 rounds | ~3300 rows/s |

Index reads are a fixed cost per *shard touched*, and by ~1024 rows a batch
already hits most of the 962 shards in the collection. Quadrupling the batch
reads the same indexes while doing 4x the data work, so the share the cache
removes shrinks to nothing. Throughput rising 3.5x from 1024 to 9192 is the
same effect seen from the other side.

**Recommendation: do not ship as-is.** A loader using batches past ~1024 gets
nothing, while carrying unbounded memory growth and a write-invalidation
hazard. Either scope it to small batches with that stated, or drop it.

Caveat: four replicates per arm in a noisy shared environment, with a visible
warming trend within each sweep. Enough to rule out a large effect at 4096+,
not enough to resolve a few percent.

Still outstanding if it does ship:

- no size bound or eviction policy; the map grows to every shard ever touched
  (13.5 MiB for this collection, unbounded in general). `size_held()` is there
  for exactly this
- eviction assumes every write goes through this pipeline
- env var -> `zarr.config` key

---

## Task 2 — inner shard index cache (built, needs its own PR)

Already on `perf/sharding-read-plan` as `3773412d`. **Nested sharding only** —
it never executes on flat-sharded data, so it contributed nothing to the 6.9×
and must not be justified by it.

Its own case, measured in `zarrs/tests/codec_sharding_nested.rs`:

- second access to the same inner shard: **1 read, was 2**
- eight visits to four inner shards: **12 reads, was 16**
- `origin/main` and read-plan-without-the-cache both do 2 — so dropping it is
  neutral versus main, not a regression

Cost to carry: an unbounded `HashMap`, a `Mutex` per inner-chunk access, and a
`local_subchunk_grids()` probe in `new` on every decoder including flat ones.
That probe sits next to a storage read so it is noise, but it is the only place
the two features touch and its job is to keep them apart.

Ship as: *repeat access to an inner shard re-reads its index; main does 2 reads
where 1 suffices.* Do not mention the fetch pool.

---

## Task 3 — benchmark, separate branch

Branch `bench/shard-index-cache` off `perf/vindex-fetch-pool`. Do not measure on
the branch being changed.

Before anything: **prove the code under test runs.** Four sweeps in this project
measured zarr-python's fallback rather than our pipeline, and a flat response
across every knob was the signal — it read as "the optimisation does not help"
for hours. `bench/probe_path.py` and `bench/probe_sorted.py` count which
pipeline carries the reads.

Setup:

- one dedicated node, not the shared `ai.sh` workspace — a neighbour's build
  moves timings more than the change being measured
- everything volatile created directly on `/localscratch/$USER`: venv,
  `UV_CACHE_DIR`, `CARGO_HOME`, `CARGO_TARGET_DIR`, `TMPDIR`. Build there, do
  not copy from Lustre
- `codec_pipeline.file_handle_cache_size = 512` fixed; do not sweep it
- fresh random seed per run, shuffled config order within each round, three
  replicates

Report per batch: index reads, wall time, and the checksum. The checksum must
not move when the cache is toggled — that is the correctness claim, and it makes
the first two runs double as the smoke test.

Expected: batch 0 unchanged, batches 1+ lose ~11.5 MiB of index reads. Whether
that shows up in wall time is the open question — those reads are concurrent
already, so the win may be smaller than the byte count suggests. Report the null
if it is null.

---

## Before any of this ships

- `ZARRS_PYTHON_FETCH_THREADS` → a `zarr.config` key
  (`codec_pipeline.fetch_threads`), following how `85aa038` exposed
  `codec_pipeline.file_handle_cache_size`; the env var is benchmark scaffolding
- ~~`MIN_COORDS_PER_RUN = 32` is tuned, not principled~~ — replaced by the
  output-selection type. `CoordinateIndexer` returns a plain slice exactly when
  `sel_sort is None`, i.e. it did not have to reorder the coordinates, which is
  the same condition zarr-python#4172 keys off. No constant, and it fails
  closed: if that optimisation changed, this would stop splitting rather than
  start fragmenting
- the pool ships opt-in until a second workload agrees
- `bench/` and `probe_*.py` do not ship; `tests/test_vindex_1d.py` already
  guards the behaviour permanently
