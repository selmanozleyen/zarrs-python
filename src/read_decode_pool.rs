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
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

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

/// One innermost chunk to decode, once its bytes have arrived.
struct Job {
    key: StoreKey,
    out: OutRegion,
    coords: CoordsRef,
    /// Everything the decode needs, shared rather than copied per job.
    ctx: Arc<JobContext>,
    /// Where to report completion. Per call, so a reply finds the right caller.
    done: Sender<Outcome>,
}

/// One `pread`: a contiguous extent covering one or more inner chunks.
///
/// Chunks that sit back to back in the shard are read together. The saving is a syscall
/// and a seek, never a byte: only ranges that actually touch are merged, so a merged read
/// fetches exactly what the separate reads would have.
///
/// Whether this helps is a property of the STORE, not of the code. A shard written in
/// chunk-id order makes a run of consecutive ids one extent; the plates measured here were
/// written concurrently, so id order and byte order barely correlate (r = 0.18) and only
/// ~16 of 1,952 consecutive-id pairs touch. Sorting by offset is what finds the pairs that
/// do, and it costs an O(n log n) sort of a few hundred items against ~2.9 ms per read.
struct ReadJob {
    key: StoreKey,
    range: ByteRange,
    /// `(offset within this extent, encoded length, the chunk to decode)`, in byte order.
    members: Vec<(usize, usize, Job)>,
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
    jobs: Sender<ReadJob>,
    read_threads: usize,
    decode_threads: usize,
}

/// Worker-side time, in nanoseconds, accumulated across every pool thread.
///
/// The caller's own phases are timed inline; these three are not, because they happen on
/// threads the caller is blocked waiting for. Summed rather than maxed: the question is
/// where the CPU-seconds go, and this run spends ~18.4 of them against zarr-python's
/// ~11.2 for the same data.
static NS_READ: AtomicU64 = AtomicU64::new(0);
static NS_DECODE: AtomicU64 = AtomicU64::new(0);
static NS_GATHER: AtomicU64 = AtomicU64::new(0);

/// Caller-side time: the phases the Python thread is inside, holding nothing else up.
static NS_LOCATE: AtomicU64 = AtomicU64::new(0);
static NS_DISPATCH: AtomicU64 = AtomicU64::new(0);
static NS_WAIT: AtomicU64 = AtomicU64::new(0);
static NS_CALL: AtomicU64 = AtomicU64::new(0);
static N_CALLS: AtomicU64 = AtomicU64::new(0);

/// Print the budget once, when the process exits, so a run reports totals rather than a
/// line per call. Only when ZARRS_PHASE_STATS is set.
pub(crate) fn report_phases() {
    let calls = N_CALLS.load(Ordering::Relaxed);
    if calls == 0 {
        return;
    }
    let ms = |x: &AtomicU64| x.load(Ordering::Relaxed) as f64 / 1e6;
    let (call, locate, dispatch, wait) = (
        ms(&NS_CALL), ms(&NS_LOCATE), ms(&NS_DISPATCH), ms(&NS_WAIT),
    );
    let (read, decode, gather) = (ms(&NS_READ), ms(&NS_DECODE), ms(&NS_GATHER));
    eprintln!(
        "PHASES over {calls} calls, milliseconds\n\
         \x20 caller  total {call:9.1}  = locate {locate:9.1} + dispatch {dispatch:9.1} \
         + wait {wait:9.1}\n\
         \x20 workers          {:9.1}  = read   {read:9.1} + decode   {decode:9.1} \
         + gather {gather:9.1}\n\
         \x20 caller time NOT in those three: {:.1} ms",
        read + decode + gather,
        call - locate - dispatch - wait,
    );
}

static PIPELINE: OnceLock<Pipeline> = OnceLock::new();
/// Serialises construction. `OnceLock::get().is_none()` then `set()` is check-then-act: two
/// first callers both spawn R + D threads and the loser's are thrown away -- and when they
/// asked for different widths, which one gets the "fixed for the process" error is a coin
/// flip. `pipeline.py` dispatches reads through `asyncio.to_thread` and the GIL is released,
/// so concurrent first calls are ordinary, not hypothetical.
static PIPELINE_INIT: Mutex<()> = Mutex::new(());

/// Start the threads on the first read request, at the configured widths.
///
/// Fixed for the process: a later request for a different width is an error rather than a
/// silent run at somebody else's width, which would put a number in the report under the
/// wrong config. Sweeping `read_concurrency` therefore takes one job per point, which is
/// what the ratios want anyway -- each job runs its own pool arm beside its own fallback arm
/// and the ratios are compared across points, never the absolutes.
fn pipeline(read_threads: usize, decode_threads: usize) -> PyResult<&'static Pipeline> {
    let (read_threads, decode_threads) = (read_threads.max(1), decode_threads.max(1));
    if PIPELINE.get().is_none() {
        let _init = PIPELINE_INIT.lock().expect("pipeline init poisoned");
        // Re-check under the lock: whoever held it may have built it already.
        if PIPELINE.get().is_some() {
            return check_widths(PIPELINE.get().expect("just checked"), read_threads, decode_threads);
        }
        let (job_tx, job_rx) = unbounded::<ReadJob>();
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
        let _ = PIPELINE.set(Pipeline {
            jobs: job_tx,
            read_threads,
            decode_threads,
        });
    }
    check_widths(PIPELINE.get().expect("set above"), read_threads, decode_threads)
}

/// The widths are fixed for the process, so a call asking for different ones is refused
/// rather than run at somebody else's width -- which would put a number in the report under
/// the wrong config.
fn check_widths(
    pipeline: &'static Pipeline,
    read_threads: usize,
    decode_threads: usize,
) -> PyResult<&'static Pipeline> {
    if pipeline.read_threads != read_threads || pipeline.decode_threads != decode_threads {
        return Err(PyRuntimeError::new_err(format!(
            "the pipeline is running {} readers and {} decoders and this call asked for {} \
             and {}; the threads are fixed for the process, so sweep one job per point",
            pipeline.read_threads, pipeline.decode_threads, read_threads, decode_threads
        )));
    }
    Ok(pipeline)
}

/// A reader: block on the filesystem, hand the bytes on, take the next job.
///
/// Never decodes. Parks on `recv` when there is nothing to read, for the life of the process.
fn read_loop(reads: &Receiver<ReadJob>, decodes: &Sender<(Job, MaybeBytes)>) {
    while let Ok(read) = reads.recv() {
        let t0 = Instant::now();
        let fetched = read.ctx_store().get_partial(&read.key, read.range);
        NS_READ.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        match fetched {
            Ok(bytes) => {
                for (at, len, job) in read.members {
                    // `Bytes` is refcounted and `slice` is O(1), so splitting one extent
                    // across its chunks copies nothing -- each decoder gets a view.
                    let part = match &bytes {
                        Some(all) if at + len <= all.len() => Some(all.slice(at..at + len)),
                        Some(all) => {
                            let _ = job.done.send(Outcome::Failed(format!(
                                "{} wanted bytes {at}..{} of an extent {} long",
                                job.key,
                                at + len,
                                all.len()
                            )));
                            continue;
                        }
                        None => None,
                    };
                    if decodes.send((job, part)).is_err() {
                        return; // no decoders left: the process is going away
                    }
                }
            }
            Err(e) => {
                for (_, _, job) in read.members {
                    let key = job.key.clone();
                    let _ = job
                        .done
                        .send(Outcome::Failed(format!("read {key} failed: {e}")));
                }
            }
        }
    }
}

impl ReadJob {
    /// Every member of one extent shares a context, so any of them names the store.
    fn ctx_store(&self) -> &ReadableWritableListableStorage {
        &self.members[0].2.ctx.store
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
    let t_decode = Instant::now();
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

    NS_DECODE.fetch_add(t_decode.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let t_gather = Instant::now();
    gather(scratch, coords, out, size).map_err(|e| format!("{}: {e}", job.key))?;
    NS_GATHER.fetch_add(t_gather.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
    /// `pread` calls issued. Below `chunks` when chunks were byte-adjacent and merged.
    pub reads: usize,
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
        let t_call = Instant::now();

        // ---------------------------------------------- locate each chunk in its shard
        let t_locate = Instant::now();
        let (located, declined, shard_indexes) = self.locate_chunks(shard, items, &ctx)?;
        NS_LOCATE.fetch_add(t_locate.elapsed().as_nanos() as u64, Ordering::Relaxed);
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
        let t_dispatch = Instant::now();
        let (done_tx, done_rx) = bounded::<Outcome>(located.len());
        let mut absent = 0usize;

        // Absent chunks first: they are output, not I/O, and keeping them out of the
        // coalescing keeps that a question purely about byte ranges.
        let mut present: Vec<(StoreKey, u64, u64, Job)> = Vec::with_capacity(located.len());
        for ((item, range), region) in located.iter().zip(regions) {
            let Some(ByteRange::FromStart(offset, Some(size))) = range else {
                match range {
                    None => {
                        absent += 1;
                        // SAFETY: this region belongs to no job; nothing else can touch it.
                        let out = unsafe { region.as_mut() };
                        fill(out, &self.fill_value, element_size)
                            .map_py_err::<PyRuntimeError>()?;
                        continue;
                    }
                    // The shard index gives every present chunk an offset and a size, so
                    // any other shape means the index was read wrong rather than that this
                    // chunk needs a different path.
                    Some(other) => {
                        return Err(PyRuntimeError::new_err(format!(
                            "{} has byte range {other:?}, which the shard index cannot \
                             produce for a present chunk",
                            item.key
                        )));
                    }
                }
            };
            present.push((
                item.key.clone(),
                *offset,
                *size,
                Job {
                    key: item.key.clone(),
                    out: region,
                    coords: CoordsRef {
                        ptr: coords_of(item)?.as_ptr(),
                        len: coords_of(item)?.len(),
                    },
                    ctx: ctx.clone(),
                    done: done_tx.clone(),
                },
            ));
        }
        let sent = present.len();

        // Sort by (shard, offset), then merge runs. With a gap budget of 0 a merged read
        // fetches exactly what the separate reads would have. Above 0 it deliberately
        // fetches bytes nobody wants, to buy one seek instead of several -- which is what
        // zarr-python's sharding decoder does at 1 MiB gap / 16 MiB per read, and is why
        // a shard written out of chunk-id order costs it nothing and costs us everything.
        let gap = self.read_coalesce_max_gap_bytes;
        let cap = self.read_coalesce_max_bytes.max(1);
        present.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        let mut reads = 0usize;
        while !present.is_empty() {
            let (key, start, first_size, _) = &present[0];
            let (key, start, first_size) = (key.clone(), *start, *first_size);
            let mut end = start + first_size;
            let mut j = 1usize;
            while j < present.len()
                && present[j].0 == key
                // Sorted by offset, so this only underflows if two items name the same
                // chunk; saturating_sub makes that a zero gap rather than a huge one.
                && present[j].1.saturating_sub(end) <= gap
                && (present[j].1 + present[j].2).saturating_sub(start) <= cap
            {
                end = end.max(present[j].1 + present[j].2);
                j += 1;
            }
            let members = present
                .drain(..j)
                .map(|(_, offset, size, job)| {
                    (
                        usize::try_from(offset - start).unwrap_or(usize::MAX),
                        usize::try_from(size).unwrap_or(usize::MAX),
                        job,
                    )
                })
                .collect();
            pipeline
                .jobs
                .send(ReadJob {
                    key,
                    range: ByteRange::FromStart(start, Some(end - start)),
                    members,
                })
                .map_py_err::<PyRuntimeError>()?;
            reads += 1;
        }
        drop(done_tx);
        NS_DISPATCH.fetch_add(t_dispatch.elapsed().as_nanos() as u64, Ordering::Relaxed);

        // ------------------------------------------------------------------ wait for all
        let t_wait = Instant::now();
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
        NS_WAIT.fetch_add(t_wait.elapsed().as_nanos() as u64, Ordering::Relaxed);
        NS_CALL.fetch_add(t_call.elapsed().as_nanos() as u64, Ordering::Relaxed);
        N_CALLS.fetch_add(1, Ordering::Relaxed);
        if let Some(e) = first_error {
            return Err(PyRuntimeError::new_err(e));
        }

        let counts = PoolCounts {
            chunks: sent,
            decoded,
            absent,
            shard_indexes,
            declined: declined_n,
            reads,
        };
        // Experiment branch only: the merge rate is the whole question here, and a
        // timing cannot answer it -- flat could mean "merged nothing" or "merged plenty
        // and reads were never the cost". Behind an env var so a measured arm stays clean.
        if std::env::var_os("ZARRS_MERGE_STATS").is_some() {
            eprintln!(
                "MERGE call: {} chunks -> {} reads ({:.1}% fewer), {} shard indexes",
                counts.chunks,
                counts.reads,
                if counts.chunks > 0 {
                    100.0 * (counts.chunks - counts.reads) as f64 / counts.chunks as f64
                } else {
                    0.0
                },
                counts.shard_indexes
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
