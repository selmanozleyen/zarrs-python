# Shard index caching

Two caches, both about not re-reading a shard index that never changes. One is
built and unmeasured; the other is unbuilt and has the larger measured case for
it. They are independent — either can ship alone.

---

## Where the fast path stands

The 6.9× on sorted 1024-row CSR gathers comes from four pieces, in a chain
where each gates the next.

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
| `325f198` expose file handle cache | **yes — zarrs-python side only** |
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

## Task 1 — outer shard index cache (unbuilt, the larger win)

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

### Definition of done

- index reads per batch drop to ~0 after the first, measured with the same probe
- correctness after write: a cached shard that is written then read returns the
  new bytes — this needs a test, it is the whole risk
- memory bounded and reported

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
- `ZARRS_PYTHON_FILE_HANDLE_CACHE=512` fixed; do not sweep it
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

- env vars → `zarr.config` keys (`codec_pipeline.fetch_threads`,
  `codec_pipeline.file_handle_cache_size`); the env vars are benchmark
  scaffolding
- `file_handle_cache_size` should default on — upstream's `0` is absence of an
  opinion, not a decision
- `MIN_COORDS_PER_RUN = 32` is tuned, not principled. It guards a 30×
  regression (unsorted rows give 1.9 coords/run against 1550 sorted). Either
  own it in the docstring with that measurement or weaken it to `> 1`
- the pool ships opt-in until a second workload agrees
- `bench/` and `probe_*.py` do not ship; `tests/test_vindex_1d.py` already
  guards the behaviour permanently
