//! Reading and decoding the innermost chunks of one call, concurrently.
//!
//! One job per innermost chunk: a reader does the blocking byte-range read, a decode worker
//! decodes the chunk and copies out the elements the selection wants. The two are separate
//! because a read waits on storage and a decode occupies a core, so the useful number of
//! each is different -- hence `read_concurrency` and `decode_concurrency`.
//!
//! Workers belong to the CALL. `std::thread::scope` cannot exit until they finish, so a job
//! can hold `&mut [u8]` into the caller's output rather than a raw pointer, and the join is
//! the barrier. The output is carved once in offset order by `split_at_mut`, so two jobs
//! cannot name the same bytes -- not checked, unrepresentable.
//!
//! There is exactly one copy of the data, the gather from the decoded chunk into the numpy
//! buffer; the bytes are moved, not copied, between reader and decoder.
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, unbounded};
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

/// The per-array state a decode needs, shared by every job of a call.
struct JobContext {
    shard: Arc<ShardInfo>,
    store: ReadableWritableListableStorage,
    codec_options: CodecOptions,
    element_size: usize,
}

impl CodecPipelineImpl {
    /// Read and decode `items`, one job per innermost chunk, on workers scoped to this call.
    ///
    /// `items` must be chunk-unit items: one whole innermost chunk each, carrying the
    /// coordinates wanted from it. Returns the items this path could not take, for the caller
    /// to run down the fused path.
    pub(crate) fn retrieve_chunk_units<'a>(
        &self,
        shard: &Arc<ShardInfo>,
        items: &'a [ChunkItem],
        output: &mut [u8],
        codec_options: &CodecOptions,
    ) -> PyResult<Vec<&'a ChunkItem>> {
        let element_size = self.element_size()?;
        let ctx = JobContext {
            shard: shard.clone(),
            store: self.store.clone(),
            codec_options: (*codec_options).with_concurrent_target(1),
            element_size,
        };

        let (located, declined) = self.locate_chunks(shard, items, &ctx)?;
        if located.is_empty() {
            return Ok(declined);
        }

        let (jobs, absent) = carve(output, &located, element_size, &ctx)?;

        // No read, no decode, no thread.
        for piece in absent {
            fill(piece, &self.fill_value, element_size).map_py_err::<PyRuntimeError>()?;
        }
        if jobs.is_empty() {
            return Ok(declined);
        }

        // What this call WANTS is its configured width, capped by the work it has: three
        // chunks give a fourth reader nothing to do. What it may HAVE at once is bounded by
        // the global ceiling, which is a different and much larger number.
        // One ceiling, two counters: each kind gets its own budget of this size.
        let ceiling = worker_ceiling();
        let want_readers = self.read_concurrency.min(jobs.len());
        let want_decoders = self.decode_concurrency.min(jobs.len());
        let _call = ActiveCall::enter();
        let failure: Mutex<Option<String>> = Mutex::new(None);
        let alive = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            let (job_tx, job_rx) = unbounded::<Job<'_>>();
            // Unbounded, but only this call's own jobs are ever sent, so peak resident is
            // the batch the caller asked for. A tighter bound cost 12%: readers running
            // ahead is the prefetch on high-latency storage.
            let (dec_tx, dec_rx) = unbounded::<(Job<'_>, MaybeBytes)>();
            let spawn_reader = |permit: Permit| {
                debug_assert_eq!(permit.0, Kind::Read);
                let (jobs, decodes, failure) = (job_rx.clone(), dec_tx.clone(), &failure);
                let alive = Alive::enter(&alive);
                scope.spawn(move || {
                    // Both held for the life of the thread and released as it returns: the
                    // permit is free the moment this worker stops using it, and the count
                    // falls the moment it stops existing.
                    let _permit = permit;
                    let _alive = alive;
                    read_loop(&jobs, &decodes, failure);
                });
            };
            let spawn_decoder = |permit: Permit| {
                debug_assert_eq!(permit.0, Kind::Decode);
                let (decodes, failure) = (dec_rx.clone(), &failure);
                let alive = Alive::enter(&alive);
                scope.spawn(move || {
                    let _permit = permit;
                    let _alive = alive;
                    decode_loop(&decodes, failure);
                });
            };

            let mut readers = initial_permits(Kind::Read, want_readers, ceiling);
            let mut decoders = initial_permits(Kind::Decode, want_decoders, ceiling);
            let (mut live_readers, mut live_decoders) = (readers.len(), decoders.len());
            for permit in readers.drain(..) {
                spawn_reader(permit);
            }
            for permit in decoders.drain(..) {
                spawn_decoder(permit);
            }

            for job in jobs {
                if job_tx.send(job).is_err() {
                    record(&failure, "no readers left to take the job".to_string());
                    break;
                }
            }
            drop(job_tx);

            // Widen while there is still queued work, so a call that started at the floor
            // does not keep one worker for its whole duration. Run from the calling thread,
            // which is idle until the join. Polls; a Condvar in `Permit::drop` is the upgrade
            // if this ever shows up in a profile.
            //
            // Outstanding work in EITHER queue keeps this alive, not just the job queue:
            // reads can drain while decode is still the bottleneck, and decoders that started
            // narrow would then have no way to widen.
            //
            // Stops for one reason only: nothing is left alive to drain the queues. Not a
            // busy ceiling, which is normal with several calls in flight, and not elapsed
            // time, which is what slow storage looks like.
            while (live_readers < want_readers || live_decoders < want_decoders)
                && (!job_rx.is_empty() || !dec_rx.is_empty())
                && alive.load(Ordering::Relaxed) > 0
            {
                let mut took = false;
                if live_readers < want_readers && !job_rx.is_empty() {
                    if let Some(permit) = Permit::take(Kind::Read, ceiling) {
                        spawn_reader(permit);
                        live_readers += 1;
                        took = true;
                    }
                }
                if live_decoders < want_decoders && !dec_rx.is_empty() {
                    if let Some(permit) = Permit::take(Kind::Decode, ceiling) {
                        spawn_decoder(permit);
                        live_decoders += 1;
                        took = true;
                    }
                }
                if !took {
                    std::thread::sleep(WIDEN_POLL);
                }
            }

            // The clones held here would keep every worker waiting on a channel that will
            // never deliver again, and the scope cannot exit until they return.
            drop(job_rx);
            drop(dec_tx);
            drop(dec_rx);
        });

        if let Some(e) = failure.lock().expect("failure slot poisoned").take() {
            return Err(PyRuntimeError::new_err(e));
        }
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
            // Same silent 1-D assumption `output_offset` makes about `subset`, and checked
            // for the same reason: a multi-dimensional `chunk_subset` would locate the wrong
            // inner chunk here and report success.
            if item.chunk_subset.dimensionality() != 1 {
                return Err(PyRuntimeError::new_err(format!(
                    "{} has a {}-dimensional chunk subset; this path reads 1-D chunks",
                    item.key,
                    item.chunk_subset.dimensionality()
                )));
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
/// Split the output into the disjoint piece each located chunk writes, in offset order.
///
/// This is where disjointness comes from, and why nothing downstream has to check it: each
/// piece is split off the tail of the last, so two overlapping items would need the same
/// bytes twice and `split_at_mut` has already moved them out of reach. A backwards offset is
/// the one thing to reject -- it means the sort did not produce a partition.
///
/// Returns the jobs to read, and the pieces of chunks that were never written, which need
/// only the fill value.
fn carve<'a>(
    output: &'a mut [u8],
    located: &[(&'a ChunkItem, Option<ByteRange>)],
    element_size: usize,
    ctx: &'a JobContext,
) -> PyResult<(Vec<Job<'a>>, Vec<&'a mut [u8]>)> {
    let mut order: Vec<usize> = (0..located.len()).collect();
    order.sort_by_key(|&i| output_offset(located[i].0));

    let mut jobs: Vec<Job<'a>> = Vec::with_capacity(order.len());
    let mut absent: Vec<&mut [u8]> = Vec::new();
    let mut rest: &mut [u8] = output;
    let mut cursor = 0usize;
    for &i in &order {
        let (item, range) = &located[i];
        let coords = coords_of(item)?;
        // WHERE a piece starts comes from `subset`, and HOW LONG it is comes from `coords`.
        // Nothing ties the two together: `ChunkItem` is constructible from Python and skips
        // the element-count check when coords are present. If they disagree, every later
        // piece is carved at the wrong offset and the read silently returns the right number
        // of wrong elements. The shipped pipeline cannot produce that -- only
        // `build_chunk_unit_items` sets coords, and it keeps the two the same length -- so
        // this is the check that keeps "cannot" from meaning "has not been tried".
        if item.subset.dimensionality() != 1 {
            return Err(PyRuntimeError::new_err(format!(
                "{} has a {}-dimensional subset; this path carves a 1-D output",
                item.key,
                item.subset.dimensionality()
            )));
        }
        if coords.len() as u64 != item.subset.num_elements() {
            return Err(PyRuntimeError::new_err(format!(
                "{} wants {} coordinates but its output subset holds {} elements",
                item.key,
                coords.len(),
                item.subset.num_elements()
            )));
        }
        // Checked, because `output_offset` saturates to `usize::MAX` when a start does not
        // fit. Unchecked, that multiplication wraps in release and the wrapped value then
        // slips past the bounds test below, so the clean error turns into a panic inside
        // `split_at_mut`.
        let (Some(start), Some(len)) = (
            output_offset(item).checked_mul(element_size),
            coords.len().checked_mul(element_size),
        ) else {
            return Err(PyRuntimeError::new_err(format!(
                "{} names an output offset or length too large to address",
                item.key
            )));
        };
        if start < cursor {
            return Err(PyRuntimeError::new_err(format!(
                "{} claims output bytes from {start}, behind the {cursor} already carved",
                item.key
            )));
        }
        let skip = start - cursor;
        if skip.checked_add(len).is_none_or(|end| end > rest.len()) {
            return Err(PyRuntimeError::new_err(format!(
                "{} names output bytes from {start} for {len}, beyond the buffer",
                item.key
            )));
        }
        let (_gap, tail) = std::mem::take(&mut rest).split_at_mut(skip);
        let (piece, tail) = tail.split_at_mut(len);
        rest = tail;
        cursor = start + len;

        match range {
            Some(range) => jobs.push(Job {
                key: item.key.clone(),
                range: *range,
                out: piece,
                coords,
                ctx,
            }),
            None => absent.push(piece),
        }
    }
    Ok((jobs, absent))
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

/// Live workers across every in-flight call, counted separately per kind.
///
/// One counter for both would let a wide decode width spend the readers' budget, starving
/// every later call's readers. A call takes what is free rather than waiting for a full set,
/// with a floor of one so it never waits forever.
static LIVE_READERS: AtomicUsize = AtomicUsize::new(0);
static LIVE_DECODERS: AtomicUsize = AtomicUsize::new(0);

/// Which budget a worker is drawn from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Read,
    Decode,
}

impl Kind {
    fn live(self) -> &'static AtomicUsize {
        match self {
            Self::Read => &LIVE_READERS,
            Self::Decode => &LIVE_DECODERS,
        }
    }
}

/// Workers of one kind per core, at the process-wide ceiling.
///
/// Over-providing is cheap here: a thread parked on a storage round trip costs a stack, not
/// a core. Measured on 16 cores over a sharded store, 8x beat 1x and 32x bought nothing.
const CEILING_WORKERS_PER_CORE: usize = 8;

/// How long the widening loop waits between attempts when no permit is free.
const WIDEN_POLL: Duration = Duration::from_micros(200);

/// Workers alive in ONE call, raised before a thread starts and lowered when it returns.
///
/// The widening loop below needs to tell "everything is busy" from "nothing is left to drain
/// this queue". Do not infer that from elapsed time: on slow storage, which is what this
/// module is for, no progress for a while is normal.
struct Alive<'a>(&'a AtomicUsize);

impl<'a> Alive<'a> {
    /// Taken on the spawning thread, so the loop cannot read zero between spawn and run.
    fn enter(count: &'a AtomicUsize) -> Self {
        count.fetch_add(1, Ordering::AcqRel);
        Self(count)
    }
}

impl Drop for Alive<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// How many workers of one kind may exist at once across every in-flight call.
///
/// Taken from the machine, not from any array's width: two arrays may be opened with
/// different `read_concurrency`, and deriving a process-wide ceiling from one of them would
/// leave the other permanently locked out.
fn worker_ceiling() -> usize {
    static CEILING: OnceLock<usize> = OnceLock::new();
    *CEILING.get_or_init(|| {
        std::thread::available_parallelism()
            .map_or(8, std::num::NonZeroUsize::get)
            .saturating_mul(CEILING_WORKERS_PER_CORE)
    })
}

/// Calls in flight, so a call takes a SHARE of the ceiling rather than racing for it.
///
/// First-come-first-served leaves a late call with nothing: eight calls at a width of 16
/// spend a ceiling of 128, and the ninth runs single-threaded for its whole duration.
static ACTIVE_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Counts one call for as long as it is in flight. A guard, so an early return or a panic
/// cannot leave the count high for the life of the process.
struct ActiveCall;

impl ActiveCall {
    fn enter() -> Self {
        ACTIVE_CALLS.fetch_add(1, Ordering::AcqRel);
        Self
    }

    /// An equal share of the ceiling. Never zero, or a call's queue would never be read.
    fn share(ceiling: usize) -> usize {
        (ceiling / ACTIVE_CALLS.load(Ordering::Relaxed).max(1)).max(1)
    }
}

impl Drop for ActiveCall {
    fn drop(&mut self) {
        ACTIVE_CALLS.fetch_sub(1, Ordering::AcqRel);
    }
}

/// One worker's permit, released when THAT worker exits.
///
/// The previous shape released a call's whole allocation in one subtraction after the join,
/// so a call held every thread it was given until it finished, long after the ones that had
/// drained were idle. Per-worker release is what makes those threads available to the calls
/// still working.
struct Permit(Kind);

impl Permit {
    /// `None` when the ceiling is full, so the caller decides whether to insist or wait.
    fn take(kind: Kind, ceiling: usize) -> Option<Self> {
        let counter = kind.live();
        let mut live = counter.load(Ordering::Relaxed);
        loop {
            if live >= ceiling {
                return None;
            }
            match counter.compare_exchange_weak(live, live + 1, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => return Some(Self(kind)),
                Err(actual) => live = actual,
            }
        }
    }

    /// The floor of one, taken over the ceiling. Without it a call that finds nothing free
    /// waits for a call that is itself waiting, and ten of them deadlock on each other.
    fn insist(kind: Kind) -> Self {
        kind.live().fetch_add(1, Ordering::AcqRel);
        Self(kind)
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        self.0.live().fetch_sub(1, Ordering::AcqRel);
    }
}

/// What a call STARTS with: its equal share of the ceiling, capped by the work it has.
///
/// Capped by the work because three chunks give a fourth reader nothing to do, and shared
/// rather than grabbed because a call that arrives second should not find the ceiling already
/// spent. It grows from here -- see the widening loop in `retrieve_chunk_units` -- so a share that
/// is small because ten calls were in flight is a starting point, not a sentence.
fn initial_permits(kind: Kind, want: usize, ceiling: usize) -> Vec<Permit> {
    let target = want.min(ActiveCall::share(ceiling));
    let mut held = Vec::with_capacity(target);
    while held.len() < target {
        match Permit::take(kind, ceiling) {
            Some(permit) => held.push(permit),
            None => break,
        }
    }
    if held.is_empty() {
        held.push(Permit::insist(kind));
    }
    held
}

/// One innermost chunk, and the slice of the output its elements belong in.
///
/// Every field borrows from the call. That is the whole point: the threads cannot outlive the
/// scope, so the compiler checks what a `'static` message could only assert.
struct Job<'a> {
    key: StoreKey,
    range: ByteRange,
    out: &'a mut [u8],
    coords: &'a [u64],
    ctx: &'a JobContext,
}

/// Keep the FIRST failure; later ones are usually consequences of it.
fn record(failure: &Mutex<Option<String>>, message: String) {
    let mut slot = failure.lock().expect("failure slot poisoned");
    if slot.is_none() {
        *slot = Some(message);
    }
}

fn read_loop<'a>(
    jobs: &Receiver<Job<'a>>,
    decodes: &Sender<(Job<'a>, MaybeBytes)>,
    failure: &Mutex<Option<String>>,
) {
    while let Ok(job) = jobs.recv() {
        match job.ctx.store.get_partial(&job.key, job.range) {
            // The bytes are MOVED to a decoder, not copied.
            Ok(bytes) => {
                if let Err(returned) = decodes.send((job, bytes)) {
                    // Every decoder is gone. Today that only happens when they panicked, and
                    // the scope re-raises that -- but returning in silence would leave this
                    // job's bytes of the output buffer at whatever `np.empty` left, and the
                    // call would report success. Say so instead of trusting the panic.
                    // The channel hands the job back rather than dropping it, so the key
                    // is still available to name in the error.
                    let (job, _) = returned.into_inner();
                    record(
                        failure,
                        format!("{}: no decoder left to take the chunk", job.key),
                    );
                    return;
                }
            }
            Err(e) => record(failure, format!("read {} failed: {e}", job.key)),
        }
    }
}

fn decode_loop(decodes: &Receiver<(Job<'_>, MaybeBytes)>, failure: &Mutex<Option<String>>) {
    // Allocated once per thread and reused, not once per chunk.
    let mut scratch: Vec<u8> = Vec::new();
    while let Ok((mut job, bytes)) = decodes.recv() {
        if let Err(e) = decode_one(&mut job, bytes, &mut scratch) {
            record(failure, e);
        }
    }
}

/// Decode one innermost chunk into scratch, then gather the wanted elements into `out`.
fn decode_one(job: &mut Job<'_>, bytes: MaybeBytes, scratch: &mut Vec<u8>) -> Result<(), String> {
    let ctx = job.ctx;
    let size = ctx.element_size;
    let Some(bytes) = bytes else {
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
    gather(scratch, job.coords, job.out, size).map_err(|e| format!("{}: {e}", job.key))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test rather than several: `LIVE_READERS`, `LIVE_DECODERS` and `ACTIVE_CALLS` are
    /// process-wide, and separate `#[test]` functions run concurrently in one process, so
    /// they would race each other.
    #[test]
    fn a_call_takes_a_share_and_releases_per_worker() {
        let ceiling = 16;
        let first = ActiveCall::enter();

        // Alone, a call may have the whole ceiling -- capped by the work it has.
        assert_eq!(initial_permits(Kind::Read, 4, ceiling).len(), 4);
        assert_eq!(
            LIVE_READERS.load(Ordering::Relaxed),
            0,
            "a permit is released when its worker drops it, not when the call ends"
        );

        // A second call in flight halves the share rather than finding it already spent.
        let second = ActiveCall::enter();
        assert_eq!(ActiveCall::share(ceiling), 8);
        drop(second);
        assert_eq!(ActiveCall::share(ceiling), ceiling);

        // With every permit held, a call still gets a worker: the floor of one is what stops
        // calls from waiting on each other.
        let held = initial_permits(Kind::Read, ceiling, ceiling);
        assert_eq!(held.len(), ceiling);
        let starved = initial_permits(Kind::Read, 4, ceiling);
        assert_eq!(starved.len(), 1);
        assert_eq!(LIVE_READERS.load(Ordering::Relaxed), ceiling + 1);

        // THE TWO BUDGETS ARE SEPARATE. One counter for both meant a wide decode width spent
        // the readers' ceiling, and every later call fell to the floor of one reader for its
        // whole life. Readers are exhausted here; decoders must not notice.
        assert_eq!(LIVE_DECODERS.load(Ordering::Relaxed), 0);
        let decoders = initial_permits(Kind::Decode, 4, ceiling);
        assert_eq!(
            decoders.len(),
            4,
            "decoders were charged the readers' ceiling"
        );

        drop(held);
        drop(starved);
        drop(decoders);
        drop(first);
        assert_eq!(LIVE_READERS.load(Ordering::Relaxed), 0);
        assert_eq!(LIVE_DECODERS.load(Ordering::Relaxed), 0);
        assert_eq!(ACTIVE_CALLS.load(Ordering::Relaxed), 0);
    }
}
