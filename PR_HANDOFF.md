# Handoff: turning this into a clean chain of upstream PRs

The work exists and is measured. What it is *not* yet is a set of changes an
upstream maintainer would accept one at a time. This describes how to get
there, and — more importantly — what each maintainer needs to hear, which is
usually not what motivated us.

**The framing that matters:** none of these should be pitched as "this made our
minibatch loader faster." Every one has a reason that stands on its own for the
project receiving it. Where a change only makes sense because of our workload,
say so plainly and make it opt-in.

---

## The workload, once, so it can be cited briefly

Backed CSR matrices in Zarr v3, sharded, Blosc-compressed, on a Lustre parallel
filesystem. Training reads minibatches of ~1024 **randomly sampled rows** out of
100.6M, across 14 stores. Each row is a contiguous run of ~1450 nnz in `data`
and `indices`, so a batch is ~2,080 scattered reads of ~100 KiB, touching
essentially every shard.

Three properties drive everything below:

- **Sparse and scattered.** Two random rows almost never share an inner chunk.
- **Latency-bound storage.** ~21 ms per request; concurrency is worth ~50x
  (46 → 2,455 preads/s from 1 → 64 threads). Bandwidth is not the constraint.
- **Codec is negligible.** ~4 ms of decompression per batch against ~2,000 ms
  of wall. Anything that trades I/O concurrency for CPU tidiness loses.

Measured end state, 14 plates: **236 → 5,172 samples/s**.

---

## PR chain, in dependency order

Each is independently reviewable. Where one depends on another, it is stated.
Sizes are the real diffs, not estimates.

### 1. zarrs-python: expose `file_handle_cache_size`
`selmanozleyen/zarrs-python` branch `feat/file-handle-cache` @ `2e3e6a7`
**6 files, +22 −3. No dependencies. Builds against released zarrs.**

*Motivation for the maintainer:* `zarrs_filesystem` has had
`FilesystemStoreOptions::file_handle_cache_size` since 0.3.11 (zarrs#422), and
Python users cannot reach it. The Rust side already decided this option is
worth having and why; this is plumbing, defaulting to `0` so nothing changes
unless asked.

*Evidence:* the option exists to avoid open/stat/close per partial read. On a
sharded array issuing one call per inner chunk that is 10–500x more metadata
syscalls, which is exactly our access pattern.

*Note:* this is the cleanest PR in the set and a good first contact. It needs a
`zarrs` version bump (0.23.6 → 0.23.13) to pick up the crate that carries it.

### 2. anndata: overlap the `data` and `indices` reads
`selmanozleyen/anndata` branch `feat/overlap-csr-reads` @ `418a000c`
**+36 −2. Independent of everything else — can go first or last.**

*Motivation for the maintainer:* `get_compressed_vectors` issues
`self.data[coords]` and then `self.indices[coords]`. Each is internally
concurrent, but the two never overlap *each other*, so a backed-CSR read costs
their sum rather than their max. They are independent — same coordinates,
different arrays, neither feeds the other.

*Evidence:* **1.28x** end to end (2015.8 → 2572.4 samples/s), average cores
5.19 → 6.52. Verified byte-identical with the flag on and off, and against a
scipy reference, across scattered / contiguous / single-row / duplicate /
empty-row selections.

*Design:* uses zarr's existing async array and sync bridge. No API change, no
new dependency, falls back to sequential reads if the async plumbing is
unavailable so correctness never depends on it. Behind
`ANNDATA_OVERLAP_READS=1`; **propose promoting to default only after the
maintainer weighs in on the sync-bridge usage.**

### 3. zarrs: multi-subset partial decode
`selmanozleyen/zarrs` branch `perf/per-chunk-fetch`, commit `1b907784`

*Motivation:* `partial_decode` takes one indexer. A caller with many small
subsets of the same chunk must either call it repeatedly or expand the
selection into element coordinates. For a sharded array with a scattered
selection, the expansion is millions of coordinates for data that is already
grouped. `partial_decode_subsets` preserves the structure the caller already
has; the default implementation keeps every existing decoder working.

*Note:* this is the foundation for 4–7. Land it alone first.

### 4. zarrs: per-chunk fetch instead of one batched read per shard
`perf/per-chunk-fetch`, commit `5234b158`

*Motivation:* the sharding decoder fetched every inner chunk of a shard in one
`partial_decode_many`, which a store loops **sequentially**. Outstanding reads
then equal concurrent *shards*, not concurrent *inner chunks* — for us ~234
against ~2,080, capping depth ~9x below what the storage rewards.

*The honest caveat, state it up front:* the batching was a deliberate
mitigation for the open/stat/close-per-call metadata storm, and the code said
so. It is only safe to remove **with the file-handle cache enabled**. With the
cache on, both forms issue the same number of preads and roughly zero metadata
syscalls; the only difference is whether they go out sequentially or
concurrently. Object stores may still prefer batching, so this likely needs to
be **selectable by store**, not a flip. That negotiation has not happened.

### 5. zarrs: decode admission gate
`perf/per-chunk-fetch`, commit `c38b1f10`

*Motivation:* fetch and decode want opposite concurrency. A thread parked in
`pread` costs no CPU, so every chunk should have its read outstanding.
Decompression is CPU-bound, so admitting every completed read at once
oversubscribes the machine. A completed read waits on a condvar until a CPU
slot frees: a chunk decodes only when its bytes have arrived **and** there is
capacity.

### 6. zarrs: dedicated I/O threads, off rayon
`perf/per-chunk-fetch`, commit `753dc260`

*Motivation, and this one is a correctness argument rather than a performance
one:* rayon is a CPU work-stealing scheduler and blocking in it is against its
contract. `ThreadPool::install` called from another pool's worker does **not**
park the caller — the docs say it "will try to keep busy ... it may potentially
schedule other tasks to run on the current thread." Measured consequence:
nesting **31 deep on a single thread**, 31 concurrent reads even with the
target set to 1, and every summed timing double-counting nested work.

*Evidence:* a nesting counter reporting `nest_depth_max` 31 → 1 after the
change. This is worth showing the maintainer directly — it is the most
convincing artifact in the set.

### 7. zarrs: expose the read plan
`perf/per-chunk-fetch`, commits `5889c245` + `241b1988`

*Motivation:* `partial_decode_subsets` fuses "work out which bytes I need"
with "fetch them" and "decode them". The first part is pure computation against
an already-resident shard index. Fusing it means the only way to overlap
several decoders is a thread per decoder parked on its own I/O — threads that
exist to work around the call shape, not because anything is unknown.
`subsets_read_plan` + `partial_decode_subsets_prefetched` let a caller build one
plan across many shards and schedule it as a whole.

*Evidence:* a test asserting both properties by counting reads on the input
handle — planning does zero reads, prefetched decode does zero reads, and the
result matches the fetch-it-yourself path.

### 8. zarrs-python: the vindex path
`feat/native-1d-vindex-sparse` @ `3fdf372`. **Depends on 3–7.**

Large and least ready to upstream. Contains the scattered planner, the
`vindex_stats` instrumentation, and the scheduler rework. Split before
proposing; the instrumentation is separable and useful alone.

*Also here, and worth its own tiny PR:* `vindex_shard_index_cache_size`
defaults to `0`. Measured across 14 plates there are only **~78 shards per
zarr array** (largest ~183) and the whole collection's indexes are **48.4 MiB**
— so the existing `256` default size is already sufficient; the problem is only
that it ships disabled.

### 9. annbatch: call anndata instead of reimplementing it
`selmanozleyen/annbatch` branch `feat/anndata-csr-fetch` @ `363cdcb`
**Depends on 2 to be competitive.**

*Motivation for the maintainer:* annbatch unwraps each `CSRDataset` into raw
`(indptr, indices, data)` arrays and hand-builds the indexers. That duplication
has **already diverged from anndata in ways that mattered** — the runs were
left unsorted and no `CoordinateIndexer` was ever built, so zarr's own fast
path could not fire. Registering a `_fetch_data` overload that keeps the
`CSRDataset` intact costs ~11% and removes a second implementation.

*Evidence:* 2572 vs 2892 samples/s with the anndata read overlap on.

---

## What is measured and what is not

**Measured, 14 plates, held allocation, wheel provenance recorded per run:**
baseline 236 · vindex 3241 (`chunk_size=1`) · **5172** (`chunk_size=8`) ·
anndata reader 2572. Locality `chunk_size` 1→8 is **1.6x**; preload 4096→16384
a further **1.19x**, turning over at 32768.

**Not measured:** anything on an object store; anything non-sharded; write
paths; whether per-chunk fetch helps or hurts a store that coalesces ranges.
PR 4 in particular should not claim generality it has not earned.

**Reproduce:** `selmanozleyen/sparse-read-bench` (private) @ `7d641c9`.
`scripts/hold.sh start` then `scripts/hold.sh run <arm>`. Arms are data in
`scripts/scenarios.py`; the runner asserts the sampling regime off the
constructed sampler, asserts clean clones, and records the wheel sha256, so a
run cannot silently differ from the one it is compared against.

---

## Things this project got wrong, so they are not repeated

Recorded because each cost real time and each was only caught by measuring.

- **The Blosc `partial_decode` capability flip** looked obviously right and was
  reverted. Upstream had already considered it and left it off with a stated
  reason (needs coalescing to be efficient); our own coalescing attempt found
  no reliable win, corroborating them. It also broke an upstream test by
  changing what a codec chain holds.
- **Vectorising anndata's coordinate build** was argued for repeatedly and is a
  **pessimisation** at this shape — 0.53x, nearly 2x slower. It only wins with
  many short runs; ours are ~1450 long. The coordinate build is ~1% of a batch.
- **"The default shard-index cache holds only 8%"** was wrong: it compared 256
  against the 3,292 global total, but the cache is per array and sees ~78.
- **"anndata is single-core because it's serial"** was measured on a *bench
  script's* serial loop, not on anndata or annbatch. With annbatch's own gather
  it runs at 3.80 cores. Check which harness produced a number before
  attributing it to a library.
- **A "decode target" sweep** that appeared to show 48 was optimal was
  measuring a knob that bounded nothing, because of the rayon nesting above.

The pattern: every one was an argument that sounded right and was wrong. The
guards in the bench harness exist because of these.

---

## Suggested order

1 and 2 first — small, independent, immediately useful, and good first contact
with two different maintainer groups. Then 3 alone to establish the API. Then
4–7 as a chain, with 4 carrying the store-selectability discussion. 8 and 9
last, after splitting.
