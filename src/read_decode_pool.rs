//! A reader pool and a decode pool, both plain OS threads that live for the process.
//!
//! One job per innermost chunk. A reader does the blocking byte-range read and hands the
//! bytes to a decode worker; the worker decodes the chunk once and copies out the elements
//! the selection wants. A read costs ~16x a decode here (~2.9 ms against ~0.185 ms for a
//! 358 KB blosc chunk), so what matters is how many reads are outstanding, and that is what
//! `read_concurrency` sets independently of the decode width.
//!
//! # Why not rayon
//!
//! It deadlocked. Rayon assumes a task runs to completion so its worker can return to the
//! scheduler; a task that parks on a channel breaks that. The thread running a
//! `ThreadPool::scope` closure blocks inside `ScopeBase::complete` waiting for the tasks it
//! spawned, and while blocked rayon has it *steal* work from its own pool -- so it picked up
//! one of the decode workers and parked in `recv`, waiting for the sender that only it could
//! ever drop. gdb caught it exactly:
//!
//! ```text
//! #8 StackJob::execute            the scope closure body
//! #6 ScopeBase::complete          waiting for its spawned jobs
//! #4 WorkerThread::wait_until_cold  looking for work to steal while it waits
//! #3 HeapJob::execute             stole a decode worker
//! #1 std::thread::functions::park blocked on done_rx.recv() -- on itself
//! ```
//!
//! Blocking belongs on threads nothing schedules. These are `std::thread`s, spawned once,
//! parked on `recv` between calls, and nothing can steal them.
//!
//! Tokio would not help either: the store here is `FilesystemStore` and `get_partial` is a
//! blocking `pread`, so tokio would hand it to `spawn_blocking` -- the same threads with an
//! async layer in front. The tokio runtime this crate already has is an async->sync bridge
//! for the *remote* stores, which is the opposite direction. Tokio earns its keep on the
//! object-store path, where one thread can hold thousands of requests outstanding.
//!
//! # Data movement
//!
//! Exactly one copy, and it is the gather:
//!
//! 1. `get_partial` -> `Bytes`: allocated by the store, **moved** through the channel.
//! 2. decode -> scratch: a write, into a buffer each worker keeps and reuses forever.
//! 3. gather scratch -> output: the one copy, scattered to contiguous, straight into the
//!    numpy buffer. No intermediate `Vec`, no assembly pass.
//!
//! # Safety
//!
//! Writing into the output from a worker needs a `'static` message, so a job carries a raw
//! region rather than a borrowed slice. The disjointness that makes this sound is
//! **verified** before any job is sent: regions are sorted by the offset each item's own
//! subset names and checked for overlap, and the caller blocks until every job it sent has
//! replied, so the buffer outlives the jobs. The fused path makes the same aliasing argument
//! in a comment and never checks it.
//!
//! ponytail: the pool's scope is ONE call, so a chunk touched by two selections is read and
//! decoded twice, and read concurrency is capped by the items in one call (~70 here).
//! Widening to a whole batch fixes both -- a job would carry several (region, coords)
//! targets and dedup would be exact -- but it needs the Python side to hand over more than
//! one selection at a time.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use pyo3::PyResult;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{
    ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset, ArrayToBytesCodecTraits,
    CodecOptions, DataType, FillValue,
};
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{MaybeBytes, ReadableWritableListableStorage, StoreKey};

use crate::CodecPipelineImpl;
use crate::chunk_item::ChunkItem;
use crate::shard_index::ShardInfo;
use crate::utils::PyErrExt as _;

/// A disjoint region of the output, owned exclusively by one job.
///
/// A raw pointer rather than `&mut [u8]` so a job is `'static` and can be handed to a thread
/// that outlives the call. Sound because `verify_disjoint` proves no two regions in a call
/// overlap, and because the caller does not return until every job has replied.
struct OutRegion {
    ptr: *mut u8,
    len: usize,
    /// Where this region starts in the output, in bytes. Kept for the overlap check and for
    /// error messages; not used to address memory.
    offset: usize,
}

// SAFETY: each region is exclusive to its job (checked by `verify_disjoint`) and the buffer
// outlives the job (the caller blocks for every reply before returning).
unsafe impl Send for OutRegion {}

impl OutRegion {
    /// # Safety
    /// Only one job may hold this region, and the output buffer must still be alive.
    unsafe fn as_mut(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// The coordinates wanted from a chunk, borrowed from the `ChunkItem` the caller holds.
///
/// Same argument as `OutRegion`: `'static` for the channel, alive because the caller blocks.
struct CoordsRef {
    ptr: *const u64,
    len: usize,
}

// SAFETY: read-only, and the `ChunkItem` it points into outlives the job.
unsafe impl Send for CoordsRef {}

impl CoordsRef {
    unsafe fn as_slice(&self) -> &[u64] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

/// What one worker reports back when it is done with a job.
enum Outcome {
    /// A chunk was read and decoded.
    Decoded,
    Failed(String),
}

/// One innermost chunk of work.
struct Job {
    key: StoreKey,
    range: ByteRange,
    out: OutRegion,
    coords: CoordsRef,
    /// Everything the decode needs, shared rather than copied per job.
    ctx: Arc<JobContext>,
    /// Where to report completion. Per call, so a reply finds the right caller.
    done: Sender<Outcome>,
}

/// The per-array state a decode needs, shared by every job of a call.
struct JobContext {
    shard: Arc<ShardInfo>,
    store: ReadableWritableListableStorage,
    data_type: DataType,
    fill_value: FillValue,
    codec_options: CodecOptions,
    element_size: usize,
}

/// The process-wide pipeline: R reader threads and D decode threads, parked between calls.
struct Pipeline {
    jobs: Sender<Job>,
}

static PIPELINE: OnceLock<Pipeline> = OnceLock::new();

/// The pool widths, fixed by the first pipeline that asks for them.
///
/// The pool is a PROCESS resource -- R readers and D decoders for the process, not per
/// array -- so its size is a process setting, the way `numba.set_num_threads`,
/// `torch.set_num_threads` and `omp_set_num_threads` are process settings. `zarr.config`
/// is a context that each array snapshots when it is opened, so two arrays can legitimately
/// hold different widths while only one of them can size the pool. The first one wins and
/// `CodecPipelineImpl::new` warns the others, instead of a later READ failing far away from
/// the `zarr.config.set` that caused it.
static WIDTHS: OnceLock<(usize, usize)> = OnceLock::new();

/// The process-wide widths, taking these as the process's if nothing has set them yet.
pub(crate) fn resolve_widths(read_threads: usize, decode_threads: usize) -> (usize, usize) {
    *WIDTHS.get_or_init(|| (read_threads.max(1), decode_threads.max(1)))
}
/// Serialises construction. `OnceLock::get().is_none()` then `set()` is check-then-act: two
/// first callers both spawn R + D threads and the loser's are thrown away -- and when they
/// asked for different widths, which one gets the "fixed for the process" error is a coin
/// flip. `pipeline.py` dispatches reads through `asyncio.to_thread` and the GIL is released,
/// so concurrent first calls are ordinary, not hypothetical.
static PIPELINE_INIT: Mutex<()> = Mutex::new(());

/// Start the threads on the first read request, at the process's widths.
///
/// Sweeping a width therefore takes one process per point, which is what a ratio wants
/// anyway: each job runs its own pool arm beside its own fallback arm, and the ratios are
/// compared across points rather than the absolutes.
fn pipeline(read_threads: usize, decode_threads: usize) -> PyResult<&'static Pipeline> {
    let (read_threads, decode_threads) = resolve_widths(read_threads, decode_threads);
    if PIPELINE.get().is_none() {
        let _init = PIPELINE_INIT.lock().expect("pipeline init poisoned");
        // Re-check under the lock: whoever held it may have built it already.
        if let Some(existing) = PIPELINE.get() {
            return Ok(existing);
        }
        let (job_tx, job_rx) = unbounded::<Job>();
        // Unbounded on purpose: a reader never waits for a decoder. Peak resident is
        // items-per-call x encoded chunk size, small at ~70 items per call.
        let (decode_tx, decode_rx) = unbounded::<(Job, MaybeBytes)>();

        for i in 0..read_threads {
            let (job_rx, decode_tx) = (job_rx.clone(), decode_tx.clone());
            std::thread::Builder::new()
                .name(format!("zarrs-read-{i}"))
                .spawn(move || read_loop(&job_rx, &decode_tx))
                .map_py_err::<PyRuntimeError>()?;
        }
        for i in 0..decode_threads {
            let decode_rx = decode_rx.clone();
            std::thread::Builder::new()
                .name(format!("zarrs-decode-{i}"))
                .spawn(move || decode_loop(&decode_rx))
                .map_py_err::<PyRuntimeError>()?;
        }
        let _ = PIPELINE.set(Pipeline { jobs: job_tx });
    }
    Ok(PIPELINE.get().expect("set above"))
}

/// A reader: block on the filesystem, hand the bytes on, take the next job.
///
/// Never decodes. Parks on `recv` when there is nothing to read, for the life of the process.
fn read_loop(jobs: &Receiver<Job>, decodes: &Sender<(Job, MaybeBytes)>) {
    while let Ok(job) = jobs.recv() {
        match job.ctx.store.get_partial(&job.key, job.range) {
            // The bytes are MOVED to a decoder, not copied.
            Ok(bytes) => {
                if decodes.send((job, bytes)).is_err() {
                    return; // no decoders left: the process is going away
                }
            }
            Err(e) => {
                let key = job.key.clone();
                let _ = job.done.send(Outcome::Failed(format!("read {key} failed: {e}")));
            }
        }
    }
}

/// A decode worker: decode the whole chunk into a scratch buffer it keeps forever, then
/// copy out the wanted elements.
fn decode_loop(decodes: &Receiver<(Job, MaybeBytes)>) {
    // Allocated once per thread, reused for every chunk it ever handles, rather than once
    // per chunk (which would be ~2,081 allocations of 386 KB per batch).
    let mut scratch: Vec<u8> = Vec::new();
    while let Ok((job, bytes)) = decodes.recv() {
        let outcome = match decode_one(&job, bytes, &mut scratch) {
            Ok(outcome) => outcome,
            Err(e) => Outcome::Failed(e),
        };
        let _ = job.done.send(outcome);
    }
}

fn decode_one(job: &Job, bytes: MaybeBytes, scratch: &mut Vec<u8>) -> Result<Outcome, String> {
    let ctx = &job.ctx;
    let size = ctx.element_size;
    // SAFETY: this job owns this region exclusively (checked by `verify_disjoint`) and the
    // output buffer is alive because the caller has not returned.
    let out = unsafe { job.out.as_mut() };
    // SAFETY: read-only, and the ChunkItem it points into outlives this job.
    let coords = unsafe { job.coords.as_slice() };

    let Some(bytes) = bytes else {
        // The index named an extent the store does not have.
        return Err(format!("{} vanished between index and read", job.key));
    };

    let shape = &ctx.shard.inner_shape;
    let elements: u64 = shape.iter().map(|s| s.get()).product();
    let needed = usize::try_from(elements).map_err(|e| e.to_string())? * size;
    scratch.clear();
    scratch.resize(needed, 0);

    let shape_u64: Vec<u64> = shape.iter().map(|s| s.get()).collect();
    {
        let slice = UnsafeCellSlice::new(scratch.as_mut_slice());
        let mut view = unsafe {
            // SAFETY: this view is the only writer to `scratch`, which this thread owns.
            ArrayBytesFixedDisjointView::new(
                slice,
                size,
                &shape_u64,
                ArraySubset::new_with_shape(shape_u64.clone()),
            )
            .map_err(|e| e.to_string())?
        };
        // The whole chunk, deliberately: blosc partial decode is quantised to blocks, which
        // prices one getitem per element at 23,233x a full decode plus a gather.
        ctx.shard
            .inner_chain
            .decode_into(
                Cow::Owned(bytes.into()),
                shape,
                &ctx.data_type,
                &ctx.fill_value,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                &ctx.codec_options,
            )
            .map_err(|e| e.to_string())?;
    }

    gather(scratch, coords, out, size).map_err(|e| format!("{}: {e}", job.key))?;
    Ok(Outcome::Decoded)
}

/// The gather zarr-python does with one numpy fancy index, over an already-decoded buffer.
///
/// `out` is exactly `coords.len()` elements and contiguous, because the indices reached us
/// non-decreasing -- so this writes straight into the output. This is the one copy.
fn gather(scratch: &[u8], coords: &[u64], out: &mut [u8], size: usize) -> Result<(), String> {
    if out.len() != coords.len() * size {
        return Err("output region does not match the coordinate count".to_string());
    }
    for (n, &c) in coords.iter().enumerate() {
        let src = usize::try_from(c).map_err(|e| e.to_string())? * size;
        let Some(element) = scratch.get(src..src + size) else {
            return Err(format!(
                "coordinate {c} is outside the {} elements decoded",
                scratch.len() / size
            ));
        };
        out[n * size..(n + 1) * size].copy_from_slice(element);
    }
    Ok(())
}

/// An absent chunk contributes only fill value, repeated.
fn fill(out: &mut [u8], fill_value: &FillValue, size: usize) -> Result<(), String> {
    let bytes = fill_value.as_ne_bytes();
    if bytes.len() != size {
        return Err("the fill value is not one element wide".to_string());
    }
    for slot in out.chunks_exact_mut(size) {
        slot.copy_from_slice(bytes);
    }
    Ok(())
}

/// What the pool did, so a run can prove this path executed rather than assuming it.
///
/// The fused path produces correct output too, so only a count separates "the pool ran" from
/// "the pool was configured and something else ran".
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PoolCounts {
    pub chunks: usize,
    pub decoded: usize,
    pub absent: usize,
    pub shard_indexes: usize,
    pub declined: usize,
}

impl CodecPipelineImpl {
    /// Read and decode `items` through the process-wide reader and decode pools.
    ///
    /// `items` must be chunk-unit items: one whole innermost chunk each, carrying the
    /// coordinates wanted from it. Returns the items this path could not take, for the caller
    /// to run down the fused path.
    pub(crate) fn retrieve_read_decode_pool<'a>(
        &self,
        shard: &Arc<ShardInfo>,
        items: &'a [ChunkItem],
        output: &mut [u8],
        codec_options: &CodecOptions,
    ) -> PyResult<(Vec<&'a ChunkItem>, PoolCounts)> {
        let element_size = self
            .data_type
            .fixed_size()
            .ok_or("variable length data type not supported")
            .map_py_err::<PyTypeError>()?;

        // Nothing nests: the codec chain gets a target of one, so a decode does not spawn
        // work underneath itself. Inheriting the fused path's target would renest silently --
        // and that target is the 4 of the (4, 4) split `calc_concurrency_outer_inner`
        // produces here, which is what leaves the fused path using a quarter of its threads.
        let ctx = Arc::new(JobContext {
            shard: shard.clone(),
            store: self.store.clone(),
            data_type: self.data_type.clone(),
            fill_value: self.fill_value.clone(),
            codec_options: codec_options.clone().with_concurrent_target(1),
            element_size,
        });

        let pipeline = pipeline(self.read_concurrency, self.decode_concurrency)?;

        // ---------------------------------------------- locate each chunk in its shard
        let (located, declined, shard_indexes) = self.locate_chunks(shard, items, &ctx)?;
        let declined_n = declined.len();
        if located.is_empty() {
            return Ok((
                declined,
                PoolCounts {
                    declined: declined_n,
                    shard_indexes,
                    ..PoolCounts::default()
                },
            ));
        }

        // ------------------------------------------------------ carve the output, checked
        let mut regions = Vec::with_capacity(located.len());
        for (item, _) in &located {
            let offset = output_offset(item) * element_size;
            let len = coords_of(item)?.len() * element_size;
            if offset + len > output.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "{} names output bytes {offset}..{} beyond the {} available",
                    item.key,
                    offset + len,
                    output.len()
                )));
            }
            regions.push(OutRegion {
                // SAFETY of the pointer arithmetic: offset + len is within the slice, checked
                // just above.
                ptr: unsafe { output.as_mut_ptr().add(offset) },
                len,
                offset,
            });
        }
        verify_disjoint(&regions, &located)?;

        // --------------------------------------------- absent chunks: no read, no decode
        let (done_tx, done_rx) = bounded::<Outcome>(located.len());
        let mut absent = 0usize;
        let mut sent = 0usize;
        for ((item, range), region) in located.iter().zip(regions) {
            match range {
                Some(range) => {
                    pipeline
                        .jobs
                        .send(Job {
                            key: item.key.clone(),
                            range: *range,
                            out: region,
                            coords: CoordsRef {
                                ptr: coords_of(item)?.as_ptr(),
                                len: coords_of(item)?.len(),
                            },
                            ctx: ctx.clone(),
                            done: done_tx.clone(),
                        })
                        .map_py_err::<PyRuntimeError>()?;
                    sent += 1;
                }
                None => {
                    absent += 1;
                    // SAFETY: this region belongs to no job; nothing else can touch it.
                    let out = unsafe { region.as_mut() };
                    fill(out, &self.fill_value, element_size).map_py_err::<PyRuntimeError>()?;
                }
            }
        }
        drop(done_tx);

        // ------------------------------------------------------------------ wait for all
        // Every job must reply before this returns: the regions and coordinate slices the
        // workers hold point into memory owned by the caller.
        let mut decoded = 0usize;
        let mut first_error: Option<String> = None;
        for _ in 0..sent {
            match done_rx.recv() {
                Ok(Outcome::Decoded) => decoded += 1,
                Ok(Outcome::Failed(e)) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
                // A worker died without replying. Returning now would free memory the
                // survivors still hold, so this is not recoverable.
                Err(e) => {
                    return Err(PyRuntimeError::new_err(format!(
                        "a pool worker stopped without replying after {decoded} of {sent} \
                         chunks ({e}); the pool cannot be trusted for the rest of this run"
                    )));
                }
            }
        }
        if let Some(e) = first_error {
            return Err(PyRuntimeError::new_err(e));
        }

        let counts = PoolCounts {
            chunks: sent,
            decoded,
            absent,
            shard_indexes,
            declined: declined_n,
        };
        // Per-call stats, behind an env var so a measurement run stays clean. The shard
        // index count is the one that matters: those reads happen on the CALLING thread,
        // one after another, before any job reaches the reader pool -- so they are
        // full-latency and entirely serial, and if there are many of them per call they
        // dominate no matter how many readers are waiting.
        if std::env::var_os("ZARRS_POOL_STATS").is_some() {
            eprintln!(
                "POOL call: {} chunks, {} decoded, {} absent, {} shard indexes read \
                 serially, {} declined",
                counts.chunks, counts.decoded, counts.absent, counts.shard_indexes,
                counts.declined
            );
        }
        Ok((declined, counts))
    }

    /// Where each item's innermost chunk lives, from its shard's own offset/size table.
    ///
    /// One index read per distinct shard, cached for the call, so N items sharing a shard
    /// cost one index read rather than N.
    #[allow(clippy::type_complexity)]
    fn locate_chunks<'a>(
        &self,
        shard: &ShardInfo,
        items: &'a [ChunkItem],
        ctx: &JobContext,
    ) -> PyResult<(Vec<(&'a ChunkItem, Option<ByteRange>)>, Vec<&'a ChunkItem>, usize)> {
        let mut indexes: HashMap<StoreKey, Option<Vec<u64>>> = HashMap::new();
        let mut located = Vec::with_capacity(items.len());
        let mut declined = Vec::new();

        for item in items {
            if item.coords.is_none() {
                declined.push(item);
                continue;
            }
            if !indexes.contains_key(&item.key) {
                let chunks_per_shard = shard.chunks_per_shard(&item.shape)?;
                let index = shard.read_index(
                    &self.store,
                    &item.key,
                    &chunks_per_shard,
                    &ctx.codec_options,
                )?;
                indexes.insert(item.key.clone(), index);
            }
            let Some(index) = indexes.get(&item.key).expect("just inserted") else {
                located.push((item, None)); // no shard at all: fill value
                continue;
            };
            let linear =
                shard.linear_index_1d(item.chunk_subset.start().first().copied().unwrap_or(0));
            located.push((item, ShardInfo::chunk_range(index, linear)?));
        }
        let shard_indexes = indexes.len();
        Ok((located, declined, shard_indexes))
    }
}

/// Prove no two regions overlap before any of them is handed to a thread.
///
/// The chunk-unit grouping emits non-decreasing index groups, so the regions should partition
/// the part of the output this call owns. Checking it is the difference between a sound
/// `unsafe` and an assertion in a comment.
fn verify_disjoint(regions: &[OutRegion], located: &[(&ChunkItem, Option<ByteRange>)]) -> PyResult<()> {
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by_key(|&i| regions[i].offset);
    let mut prev_end = 0usize;
    let mut prev: Option<usize> = None;
    for &i in &order {
        let r = &regions[i];
        if let Some(p) = prev {
            if r.offset < prev_end {
                return Err(PyRuntimeError::new_err(format!(
                    "{} claims output bytes {}..{} which overlap {}'s {}..{}",
                    located[i].0.key,
                    r.offset,
                    r.offset + r.len,
                    located[p].0.key,
                    regions[p].offset,
                    prev_end
                )));
            }
        }
        prev_end = r.offset + r.len;
        prev = Some(i);
    }
    Ok(())
}

fn coords_of(item: &ChunkItem) -> PyResult<&Vec<u64>> {
    item.coords
        .as_ref()
        .ok_or("this path requires chunk-unit items, which carry coordinates")
        .map_py_err::<PyRuntimeError>()
}

/// Where an item's elements land in the output, in elements.
///
/// 1-D only, which is what this path accepts: `_chunk_unit_items` declines any selection whose
/// chunk subset, chunk selection or output selection is not one-dimensional.
fn output_offset(item: &ChunkItem) -> usize {
    usize::try_from(item.subset.start().first().copied().unwrap_or(0)).unwrap_or(usize::MAX)
}
