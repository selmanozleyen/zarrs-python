# The three pull requests, explained

A twenty-minute read. Assumes you have read `architecture-explainer.md`, or already know what a
shard and an inner chunk are. Covers what each pull request does, why it is shaped that way,
what was tried and abandoned, and what is deliberately still wrong.

---

## The workload, stated once

Single-cell genomics. Fourteen files ("plates"), 100,648,790 rows by ~2,000 columns,
146 billion non-zeros, about 340 gigabytes. Stored as Zarr version 3, sharded, blosc-compressed.
Training reads batches of ~1,024 rows drawn at random.

The geometry that decides everything:

```
   inner chunk  =  91,549 elements  =  366 KiB     <- the decompression unit
   one row      =  ~1,452 elements                 <- what you actually wanted
                   ────────────────
                   1,452 / 91,549   =  1.6 % used

   measured end to end: 38x read amplification at fully scattered rows
```

Every design decision below is an attempt to move that number, or to stop paying it more than
once.

---

## Pull request #12 — `feat/index-normalisation` (+181 / −29, 3 files)

**Entirely Python.** No Rust. It changes *which* selections reach the fast path and fixes a
latent correctness bug on the way.

Three things:

**Indices are normalised to `int64` up front.** Numpy integer arrays arriving from a user can be
any width — `int32`, `uint32`, `uint16`. Downstream code computes differences between
consecutive indices to detect runs. On an *unsigned* dtype a decrease wraps: `5 - 7` becomes
4,294,967,294, not −2. A descending selection then looks like an enormous ascending step and is
silently accepted. `tests/test_index_dtype_overflow.py` (+101 lines, the bulk of this PR) pins
that.

**Reads and writes are routed apart.** They had shared a code path; they do not share
requirements — a write must not take a read-only fast path, and the read path is about to grow
capabilities the write path cannot use.

**The selection is recognised, not merely accepted.** The gate learns to say what *kind* of
selection it is looking at, which is what lets #13 dispatch on it.

**Why it carries no throughput number:** nothing reads faster because of it. It recognises and
normalises; the speed arrives in #13. This is worth stating in the pull request itself, because
a reviewer will look for a benchmark and should be told there isn't one and why.

---

## Pull request #13 — `feat/chunk-unit-read-path` (+5,912 / −209, 21 files)

The substantial one. It replaces the read path with a different decomposition.

### The idea in one diagram

```
   BEFORE                              AFTER
   ┌────────────────────────┐          ┌────────────────────────┐
   │ for each chunk:        │          │ one JOB per innermost  │
   │   make a partial       │          │ chunk                  │
   │   decoder, apply an    │          │   ↓                    │
   │   index through it     │          │ READ pool → DECODE pool│
   │ (on rayon)             │          │   ↓                    │
   └────────────────────────┘          │ gather every wanted    │
                                       │ row out of that chunk  │
                                       └────────────────────────┘
```

Three decisions, each of which was measured.

### Decision 1: the innermost chunk is the read unit

A shard is 179 megabytes; the inner chunk inside it is 366 kilobytes. Reading the shard to
reach one row is the difference between 40x amplification and 500x. So the path descends the
shard index and fetches exactly one inner chunk's byte range.

The cost is that it must *find* that byte range, which means reading and decoding the shard
index. `src/shard_index.rs` does the descent, and the index is cached per array — measured at
99.9% reuse within a run. Without the cache every read re-parses an index it already had.

### Decision 2: reads and decodes run on two separate persistent pools

```
   a read blocks on storage            a decode occupies a core
   ──────────────────────────          ─────────────────────────
   costs a thread, not a core          costs a core, not a thread
   wants MANY in flight                wants about as many as you have cores
```

One pool for both means a thread parked on input/output is unavailable to decompress, and a
core busy decompressing is not issuing reads. Two pools with work-stealing let each side run at
its own rate.

**What was tried and lost:** per-call worker threads drawing an equal share of a global budget.
Three reasons it failed, all in the doc comments: the share is a snapshot taken when every
caller has just arrived and is never recomputed; integer division strands capacity outright; and
a partition blocked on input/output cannot lend to one that is starved.

**The consequence a reviewer should push on:** the pools are process-wide and sized by the
*first* read. That is why the two sizes are read when an array is opened rather than per call —
offering a per-call choice that cannot be honoured is worse than not offering it. A size that
cannot be honoured now warns, or raises under `codec_pipeline.strict`. The knobs are
`codec_pipeline.read_pool_size` and `codec_pipeline.decode_pool_size`; `0` on either means "as
many threads as the machine has", which is the opposite of what `0` means on the two knobs
beside them, and the README now says so.

### Decision 3: describe a selection as runs, not as elements

`_chunk_unit_args` used to expand a contiguous slice with `np.arange`, on the reasoning that a
slice *is* a sorted integer axis spelled differently. True, and it discards the run. Measured:
one preload described ~130 runs as **11.9 million numbers, ~95 megabytes**, taking 209.7
milliseconds of a ~317 millisecond read. Described as runs — one coordinate and a length per
inner chunk — the same read is 0.69 milliseconds.

### What else is in it

Point selections, grid selections, arrays whose shard divides a trailing axis (a "banded" item,
which is why the output side needs strides at all), nested sharding, unsharded arrays, and the
shard-index cache. Roughly 900 of the 5,912 lines are tests.

### Measured

29 of 32 configurations above parity against the Python pipeline; scattered draws 0.85x–1.96x,
strided 1.38x–27.55x. Three cells at or below parity, named rather than averaged away — dense
row-major at chunk size 64 is a real 0.85x loss on the shape where a chunk already holds most of
what the draw wants. The same job measures its own noise floor at 0.88x–1.08x from sixteen cells
where two arms run identical code.

It spends more of the machine: 20.6 cores against 6.4 on the most fragmented dense cell.

---

## Pull request #17 — `feat/raw-read-unit` (+575 / −31, 8 files)

A special case, gated.

When a sharding codec's inner chain is *exactly* the `bytes` codec — no filter, no compressor —
the stored bytes are the elements in row-major order. The byte offset of any row inside a chunk
is then arithmetic, so you can fetch just that row's bytes and skip decoding entirely.

```
   ordinary:  read whole inner chunk (366 KiB) → decode → copy out one row (5.8 KiB)
   raw:       read exactly one row's bytes (5.8 KiB) ──────→ copy it
```

**What it trades:** bytes for requests. A row costs nearly what the chunk holding it costs to
*fetch* — the request count is the same either way, only the volume differs. So it is gated per
item: take the path only where a chunk's wanted rows collapse to a handful of reads.
`max_row_reads_per_chunk` is that threshold, default 2, and zero disables the path.

The counter that matters is **reads, not rows**: 64 consecutive rows are one read, 64 scattered
ones are 64. Consecutive rows are merged into one range — without that, a selection of 8-row
blocks issues eight 8-kilobyte requests where one 64-kilobyte request says the same thing.

**Measured:** 6.12x on dense row-major at stride 128, 4.76x at random chunk size 1, 3.21x at
stride 32. Nothing at all on column-major stores, by design — the gate declines there, and a
flat row in that half of the grid is the *control*, not a failed change.

**The correctness catch, found in review:** the `bytes` codec reverses multi-byte elements when
the array's byte order is not the platform's. The raw path copies stored bytes verbatim, so on a
big-endian array the chunk path would swap and the raw path would not — same array, two answers,
no error. Big-endian is legal Zarr version 3. The gate now requires native byte order, and
matches the inner chain as *one* codec rather than by collecting the names that parse, so a
codec carrying no `name` cannot be skipped past.

---

## How the three fit together

```
   #12  recognises and normalises a selection        no speed of its own
     │
     ▼
   #13  serves it as inner-chunk jobs on two pools   the throughput
     │
     ▼
   #17  skips the decode when the bytes are raw      dense row-major only
```

Each is one squashed commit. The suite counts chain: 5,740 → 5,912 → 5,917 Python tests,
1 → 13 → 16 Rust tests.

---

## What is deliberately still wrong

Worth reading before reviewing, because these are known and chosen.

**`X[:, cols]` declines.** Selecting columns from every row is not this path's shape; it falls
back to zarr-python rather than being served badly.

**Dense row-major at chunk size 64 is 0.85x.** A real loss, on the shape where the whole-chunk
unit is already the right unit. It is named in the pull request rather than averaged into a mean.

**`get_partial_many` is not used.** It is a CPU trade, not a throughput one: measured at +1.3%
throughput for −23% cores on one cell and −3.3% for −17% on another. Worth taking where cores
are scarce; not obviously worth carrying otherwise.

**The output extents are bounded by Python.** `chunk_item.rs` checks the *chunk* extents but
nothing checks that an output box fits its output row; `_step1_span` in Python is the whole
enforcement. A band spilling past its row would be silent. Cheap to close, and should be closed
when the band split grows.

**A `transpose` outside the sharding codec declines.** Legal Zarr v3, and the elements inside a
shard are then not in the array's own order. It reads through zarr-python instead. (A transpose
*inside* the shard — the ordinary `filters=` spelling — is served normally.)

**Formatting has never been run on this branch** — `cargo fmt` was applied only to the stack's
own new code, and the repository's own continuous integration rewrites files and then fails on
the diff. That job does not currently run on the fork, so nothing is red today.

---

## What review has already found and closed

Kept short, because a reviewer's time is better spent on what is still open. Every one of these
came from a review pass with no context, and every one is fixed with a test that fails without
the fix:

- **A `fork()`ed child hung forever.** The pools are process-wide, and `fork` copies memory but
  only the calling thread — so a child inherited worker threads that do not exist and parked on
  a latch nothing would signal. `DataLoader(num_workers>0)` forks by default. Keyed on the
  process id now, and the stale pair is forgotten rather than dropped (dropping joins those same
  absent threads).
- **Big-endian arrays read wrong on the row path.** The `bytes` codec reverses multi-byte
  elements when the array's order is not the platform's; the row path copies verbatim. The gate
  now requires native order.
- **The row path checked its output side and not its source.** A chunk's bytes sit inside a
  shard, so an offset past this chunk's end lands on the next chunk's bytes at exactly the length
  asked for.
- **`X[[]]` raised under `strict`** where zarr-python returns an empty array — three
  all-or-nothing passes each also required a non-empty list, and `all()` over an empty list is
  already true.
- **Fifteen assertions could not fail**, and the fixture they used would have gone silently
  vacuous on a rename — inside the fixture written to prevent exactly that.

## What a reviewer should attack first

1. **The `unsafe` block.** Several threads write into one Python-owned buffer. `DisjointBytes`
   vends every output range from a forward-only cursor, so two pieces cannot both claim a byte,
   and `covered() != output_len` catches a batch that would leave part uninitialised. Decide
   whether that argument holds.
2. **Process-wide pools.** They are sized by the first read and never resized. Ask what happens
   under a PyTorch `DataLoader` with worker processes, or when two arrays with different
   pool sizes are open at once.
3. **Scope.** Roughly how much of 5,912 lines helps anyone who is not reading scattered rows
   from sharded two-dimensional arrays? That is a fair question and the answer is honestly "less
   than half".
