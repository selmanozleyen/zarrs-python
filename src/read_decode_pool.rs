//! A reader pool and a decode pool, both plain OS threads that live for the process.
//!
//! One job per innermost chunk. A reader does the blocking byte-range read and hands the
//! bytes to a decode worker; the worker decodes the chunk once and copies out the elements
//! the selection wants.
//!
//! The two are split because they are bounded by different things: a read waits on storage
//! and a decode occupies a core, so the useful number of readers is however many reads a
//! store will answer at once, which is not the core count. Hence `read_concurrency` and
//! `decode_concurrency` are separate knobs. On networked or high-latency storage a read can
//! cost an order of magnitude more than a decode, which is the case this arrangement exists
//! for; on a local `NVMe` drive it may not, and then the widths should be equal, which is what they
//! default to.
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

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use pyo3::PyResult;
use pyo3::exceptions::PyRuntimeError;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{
    ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset, ArrayToBytesCodecTraits,
    CodecOptions, FillValue,
};
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{MaybeBytes, ReadableWritableListableStorage, StoreKey};

use crate::CodecPipelineImpl;
use crate::chunk_item::ChunkItem;
use crate::shard_index::ShardInfo;
use zarrs::array::codec::api::ByteIntervalPartialDecoder;
use zarrs::array::codec::array_to_bytes::sharding::ShardingPartialDecoder;

use crate::utils::{PyCodecErrExt as _, PyErrExt as _, gather, key_partial_decoder};

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
    ///
    /// Taking `&self` is the point: the regions are handed out from a shared slice and are
    /// disjoint by construction (`verify_disjoint`), which is what makes the aliasing sound.
    #[allow(clippy::mut_from_ref)]
    unsafe fn as_mut(&self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// The coordinates wanted from a chunk, borrowed from the `ChunkItem` the caller holds.
///
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
    coords: Arc<[u64]>,
    /// Everything the decode needs, shared rather than copied per job.
    ctx: Arc<JobContext>,
    /// Where to report completion. Per call, so a reply finds the right caller.
    done: Sender<Outcome>,
}

/// The per-array state a decode needs, shared by every job of a call.
struct JobContext {
    shard: Arc<ShardInfo>,
    store: ReadableWritableListableStorage,
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

/// Bring the pool up now, rather than leaving it to whichever read reaches it first.
///
/// Spawning `4 * cpus` readers and `cpus` decoders measures ~1.3 ms once per process, so
/// paying it when an array is opened takes it off the latency of every read. Called for
/// every array, including ones the pool cannot serve today -- an unsharded array, or one
/// only ever read with slices -- because the intent is for all reads to come through here.
pub(crate) fn start(read_threads: usize, decode_threads: usize) -> PyResult<()> {
    pipeline(read_threads, decode_threads).map(|_| ())
}

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

/// The process's pool, starting its threads if they are not up yet.
///
/// A width can only be changed by restarting the process, which is what fixing them once
/// implies: threads cannot be respawned underneath work already in flight.
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
                let _ = job
                    .done
                    .send(Outcome::Failed(format!("read {key} failed: {e}")));
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
    let coords = &job.coords;

    let Some(bytes) = bytes else {
        // The index named an extent the store does not have.
        return Err(format!("{} vanished between index and read", job.key));
    };

    let shape = &ctx.shard.subchunk_shape;
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
        // The whole chunk, deliberately. Partial decode of a compressed chunk is quantised to
        // the compressor's blocks, so asking for individual elements re-does block work per
        // element; decoding the chunk once and gathering from the result is cheaper by orders
        // of magnitude for a selection of any density.
        ctx.shard
            .inner_chain
            .decode_into(
                Cow::Owned(bytes.into()),
                shape,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                &ctx.codec_options,
            )
            .map_err(|e| e.to_string())?;
    }

    gather(scratch, coords, out, size).map_err(|e| format!("{}: {e}", job.key))?;
    Ok(Outcome::Decoded)
}

/// Block until every job sent has replied, returning how many decoded.
///
/// Nothing may return before the last reply: each worker holds an `OutRegion`, a raw pointer
/// into the output buffer the CALLER owns and frees on the way out. The coordinates are an
/// `Arc` and would survive on their own; the output regions are what make this load-bearing.
fn await_replies(done_rx: &Receiver<Outcome>, sent: usize) -> PyResult<usize> {
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
            // A worker died without replying. Returning now would free memory the survivors
            // still hold, so this is not recoverable.
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
    Ok(decoded)
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
    ) -> PyResult<Vec<&'a ChunkItem>> {
        let element_size = self.element_size()?;

        // Nothing nests: the codec chain gets a target of one, so a decode does not spawn
        // work underneath itself. Inheriting the fused path's target would renest silently --
        // and that target is the 4 of the (4, 4) split `calc_concurrency_outer_inner`
        // produces here, which is what leaves the fused path using a quarter of its threads.
        let ctx = Arc::new(JobContext {
            shard: shard.clone(),
            store: self.store.clone(),
            codec_options: (*codec_options).with_concurrent_target(1),
            element_size,
        });

        let pipeline = pipeline(self.read_concurrency, self.decode_concurrency)?;

        // ---------------------------------------------- locate each chunk in its shard
        let (located, declined) = self.locate_chunks(shard, items, &ctx)?;
        if located.is_empty() {
            return Ok(declined);
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
        let mut sent = 0usize;
        for ((item, range), region) in located.iter().zip(regions) {
            if let Some(range) = range {
                let coords = coords_of(item)?;
                pipeline
                    .jobs
                    .send(Job {
                        key: item.key.clone(),
                        range: *range,
                        out: region,
                        coords: coords.clone(),
                        ctx: ctx.clone(),
                        done: done_tx.clone(),
                    })
                    .map_py_err::<PyRuntimeError>()?;
                sent += 1;
            } else {
                // SAFETY: this region belongs to no job; nothing else can touch it.
                let out = unsafe { region.as_mut() };
                fill(out, &self.fill_value, element_size).map_py_err::<PyRuntimeError>()?;
            }
        }
        drop(done_tx);

        // ------------------------------------------------------------------ wait for all
        let decoded = await_replies(&done_rx, sent)?;

        // Every job sent was one innermost chunk, read once and decoded once. `await_replies`
        // waits for exactly `sent` outcomes and fails on any that did not decode, so this
        // cannot differ -- asserted rather than checked, so a future change to that function
        // trips it in a debug build instead of silently returning short output.
        debug_assert_eq!(sent, decoded, "pool accounting does not close");
        Ok(declined)
    }

    /// What this array remembers about one shard's index.
    /// A decoder from `cache`, or built and remembered.
    ///
    /// Shared by both levels because they differ only in what keys them and how they are
    /// built: the outermost by store key, deeper ones by that key plus the path of subchunk
    /// indices reaching them. Takes the key by reference and clones it only on INSERT, so a
    /// cache hit on the ordinary path allocates nothing.
    ///
    /// The lock is NOT held across `build`, which reads an index. Two callers can therefore
    /// read one index at the same time and the second insert wins; that costs a duplicate
    /// read and cannot give a different answer. Holding it would serialise every caller
    /// behind one full-latency read.
    fn decoder_or_read<K, B>(
        &self,
        cache: &Mutex<HashMap<K, Arc<ShardingPartialDecoder>>>,
        key: &K,
        build: B,
    ) -> PyResult<Arc<ShardingPartialDecoder>>
    where
        K: Eq + std::hash::Hash + Clone,
        B: FnOnce() -> PyResult<ShardingPartialDecoder>,
    {
        if self.cache_shard_indexes {
            let found = cache
                .lock()
                .expect("shard index cache poisoned")
                .get(key)
                .cloned();
            if let Some(found) = found {
                return Ok(found);
            }
        }
        let decoder = Arc::new(build()?);
        if self.cache_shard_indexes {
            cache
                .lock()
                .expect("shard index cache poisoned")
                .insert(key.clone(), decoder.clone());
        }
        Ok(decoder)
    }

    /// The absolute byte range of the innermost chunk holding element `start`, or `None` if
    /// any level says it was never written.
    ///
    /// One index read per LEVEL, each remembered. A singly sharded array runs this loop once
    /// and touches only the depth-0 cache, so it does exactly what it did before nesting was
    /// supported: no path vector is allocated and no second map is consulted.
    ///
    /// Below depth 0 the handle is a byte interval of the SAME store key rather than a nested
    /// interval, because `subchunk_byte_range` returns an offset relative to the level's own
    /// extent and absolute offsets compose by addition. One interval, not a chain of them.
    fn locate(
        &self,
        shard: &ShardInfo,
        item: &ChunkItem,
        start: u64,
        ctx: &JobContext,
    ) -> PyResult<Option<ByteRange>> {
        let file = key_partial_decoder(&self.store, &item.key);
        let mut shard_shape = item.shape.clone();
        let mut offset = start;
        // (offset, length) of the level being descended INTO, absolute in the store value.
        let mut extent: Option<(u64, u64)> = None;
        // The subchunk indices taken so far. Only built below depth 0, so the single-level
        // path never allocates it.
        let mut path: Vec<u64> = Vec::new();

        for depth in 0..shard.depth() {
            let subchunk = shard.subchunk_shape_at(depth)[0].get();
            let index = offset / subchunk;
            offset %= subchunk;

            let decoder = if depth == 0 {
                // Keyed by the store key alone, so the ordinary path never builds a tuple.
                self.decoder_or_read(&self.shard_indexes, &item.key, || {
                    shard.level_decoder(
                        0,
                        key_partial_decoder(&self.store, &item.key),
                        item.shape.clone(),
                        &ctx.codec_options,
                    )
                })?
            } else {
                // A subshard's index is not its shard's, so the path taken to reach it is
                // part of the key. Only built below depth 0.
                let (base, len) = extent.expect("a level below 0 has a parent extent");
                let key = (item.key.clone(), path.clone());
                self.decoder_or_read(&self.subshard_indexes, &key, || {
                    let input = Arc::new(ByteIntervalPartialDecoder::new(file.clone(), base, len));
                    shard.level_decoder(depth, input, shard_shape.clone(), &ctx.codec_options)
                })?
            };

            let Some(range) = decoder.subchunk_byte_range(&[index]).map_codec_err()? else {
                // Absent at this level: the shard is not there, or the entry is the
                // never-written marker. Either way there is nothing below it.
                return Ok(None);
            };
            // Always `FromStart` with an explicit length, so the `size` argument is unused.
            let base = extent.map_or(0, |(base, _)| base);
            extent = Some((base + range.start(0), range.length(0)));

            shard_shape.clone_from(shard.subchunk_shape_at(depth));
            if depth + 1 < shard.depth() {
                path.push(index);
            }
        }
        Ok(extent.map(|(base, len)| ByteRange::FromStart(base, Some(len))))
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
    ) -> PyResult<(Vec<(&'a ChunkItem, Option<ByteRange>)>, Vec<&'a ChunkItem>)> {
        let mut located = Vec::with_capacity(items.len());
        let mut declined = Vec::new();

        for item in items {
            if item.coords.is_none() {
                declined.push(item);
                continue;
            }
            let start = item.chunk_subset.start().first().copied().unwrap_or(0);
            located.push((item, self.locate(shard, item, start, ctx)?));
        }
        Ok((located, declined))
    }
}

/// Prove no two regions overlap before any of them is handed to a thread.
///
/// The chunk-unit grouping emits non-decreasing index groups, so the regions should partition
/// the part of the output this call owns. Checking it is the difference between a sound
/// `unsafe` and an assertion in a comment.
fn verify_disjoint(
    regions: &[OutRegion],
    located: &[(&ChunkItem, Option<ByteRange>)],
) -> PyResult<()> {
    let mut order: Vec<usize> = (0..regions.len()).collect();
    order.sort_by_key(|&i| regions[i].offset);
    let mut prev_end = 0usize;
    let mut prev: Option<usize> = None;
    for &i in &order {
        let r = &regions[i];
        if let Some(p) = prev
            && r.offset < prev_end
        {
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
        prev_end = r.offset + r.len;
        prev = Some(i);
    }
    Ok(())
}

fn coords_of(item: &ChunkItem) -> PyResult<&Arc<[u64]>> {
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
