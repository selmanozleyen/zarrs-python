//! Reading and decoding the innermost chunks of one call, concurrently.
//!
//! One job per innermost chunk: a reader does the blocking byte-range read, a decode worker
//! decodes the chunk and copies out the elements the selection wants. The two are separate
//! because a read waits on storage and a decode occupies a core, so the useful number of
//! each is different -- hence `read_concurrency` and `decode_concurrency`.
//!
//! Workers belong to the CALL. `std::thread::scope` cannot exit until they finish, so a job
//! can hold `&mut [u8]` into the caller's output rather than a raw pointer, and the join is
//! the barrier. `DisjointBytes` vends each range of the output once, in increasing offset
//! order, and what it returns is a `&mut [u8]`, so two jobs cannot name the same bytes.
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
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

use crate::utils::{PyCodecErrExt as _, PyErrExt as _, gather, gather_columns, key_partial_decoder};

/// The per-array state a decode needs, shared by every job of a call.
struct JobContext {
    shard: Arc<ShardInfo>,
    store: ReadableWritableListableStorage,
    codec_options: CodecOptions,
    element_size: usize,
    /// What an absent chunk contributes. Needed in the workers, not just at carve time,
    /// because an UNSHARDED chunk's absence is only discovered by the read.
    fill_value: FillValue,
}

/// Shard and subshard decoders built during ONE call.
///
/// Building one reads and decodes that shard's index -- a full-latency round trip, on the
/// CALLING thread, before any worker starts. A shard holds many inner chunks, so without
/// this a call pays that round trip once per ITEM rather than once per shard.
///
/// Separate from the array-lifetime cache, which only applies to a read-only store. This one
/// always applies: within a single call nothing can have moved the bytes a decoder addresses.
#[derive(Default)]
struct CallDecoders {
    shards: HashMap<StoreKey, Arc<ShardingPartialDecoder>>,
    subshards: HashMap<(StoreKey, Vec<u64>), Arc<ShardingPartialDecoder>>,
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
        output: UnsafeCellSlice<'_, u8>,
        output_len: usize,
        widths: CallWidths,
        codec_options: &CodecOptions,
    ) -> PyResult<Vec<&'a ChunkItem>> {
        let element_size = self.element_size()?;
        let ctx = JobContext {
            shard: shard.clone(),
            store: self.store.clone(),
            codec_options: (*codec_options).with_concurrent_target(1),
            element_size,
            fill_value: self.fill_value.clone(),
        };

        let (located, declined) = self.locate_chunks(shard, items, &ctx)?;
        if located.is_empty() {
            return Ok(declined);
        }

        let output = DisjointBytes::new(output, output_len);
        let (jobs, absent) = carve(&output, &located, element_size, &ctx)?;
        // Disjointness is proven above; COVERAGE is not. zarr hands us a buffer from
        // `np.empty`, so a byte no job owns is returned as whatever was in that memory.
        //
        // Only when nothing was DECLINED: a declined item's bytes are written by the fused
        // path afterwards, so a partial batch legitimately covers part of the output here.
        if declined.is_empty() && output.covered() != output_len {
            return Err(PyRuntimeError::new_err(format!(
                "the batch covers {} of {output_len} output bytes; the rest would be returned \
                 uninitialised",
                output.covered()
            )));
        }

        // No read, no decode, no thread.
        for piece in absent {
            fill(piece, &self.fill_value, element_size).map_py_err::<PyRuntimeError>()?;
        }
        if jobs.is_empty() {
            return Ok(declined);
        }

        let want_readers = widths.read.min(jobs.len());
        let want_decoders = widths.decode.min(jobs.len());
        let _call = ActiveCall::enter();
        let failure: Mutex<Option<String>> = Mutex::new(None);
        let alive = AtomicUsize::new(0);

        std::thread::scope(|scope| {
            let (job_tx, job_rx) = unbounded::<Job<'_>>();
            // Unbounded, but only this call's own jobs are ever sent, so peak resident is
            // the batch the caller asked for. Do not bound it: readers running ahead of
            // decoders is the prefetch on high-latency storage.
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

            let mut readers = initial_permits(Kind::Read, want_readers, widths.read_ceiling);
            let mut decoders = initial_permits(Kind::Decode, want_decoders, widths.decode_ceiling);
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
            // which is idle until the join.
            //
            // Outstanding work in EITHER queue keeps this alive, not just the job queue:
            // reads can drain while decode is still the bottleneck, and decoders that started
            // narrow would then have no way to widen.
            //
            // Stops only when nothing is left alive to drain the queues.
            while (live_readers < want_readers || live_decoders < want_decoders)
                && (!job_rx.is_empty() || !dec_rx.is_empty())
                && alive.load(Ordering::Relaxed) > 0
            {
                let mut took = false;
                if live_readers < want_readers && !job_rx.is_empty() {
                    if let Some(permit) = Permit::take(Kind::Read, widths.read_ceiling) {
                        spawn_reader(permit);
                        live_readers += 1;
                        took = true;
                    }
                }
                if live_decoders < want_decoders && !dec_rx.is_empty() {
                    if let Some(permit) = Permit::take(Kind::Decode, widths.decode_ceiling) {
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

    /// A decoder from `call_cache`, then `array_cache`, or built and inserted into both.
    ///
    /// Shared by both sharding levels: they differ only in the key type -- store key at
    /// depth 0, store key plus subchunk path below it.
    ///
    /// The lock is NOT held across `build`, which reads an index. Two callers can therefore
    /// read one index at the same time and the second insert wins; that costs a duplicate
    /// read and cannot give a different answer.
    fn decoder_or_read<K, B>(
        &self,
        array_cache: &Mutex<HashMap<K, Arc<ShardingPartialDecoder>>>,
        call_cache: &mut HashMap<K, Arc<ShardingPartialDecoder>>,
        key: &K,
        build: B,
    ) -> PyResult<Arc<ShardingPartialDecoder>>
    where
        K: Eq + std::hash::Hash + Clone,
        B: FnOnce() -> PyResult<ShardingPartialDecoder>,
    {
        // The call's own cache first, and it is not conditional: within one call nothing can
        // have moved the bytes a decoder addresses, so it is always safe to reuse one.
        if let Some(found) = call_cache.get(key) {
            return Ok(found.clone());
        }
        if self.cache_shard_indexes {
            let found = array_cache
                .lock()
                .expect("shard index cache poisoned")
                .get(key)
                .cloned();
            if let Some(found) = found {
                call_cache.insert(key.clone(), found.clone());
                return Ok(found);
            }
        }
        let decoder = Arc::new(build()?);
        call_cache.insert(key.clone(), decoder.clone());
        if self.cache_shard_indexes {
            array_cache
                .lock()
                .expect("shard index cache poisoned")
                .insert(key.clone(), decoder.clone());
        }
        Ok(decoder)
    }

    /// The absolute byte range of the innermost chunk holding element `start`, or `None` if
    /// any level says it was never written.
    ///
    /// One index read per LEVEL, each remembered.
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
        decoders: &mut CallDecoders,
    ) -> PyResult<Option<ByteRange>> {
        // Not sharded: there is no index to read and nothing to descend -- the store value is
        // the chunk. Whether the key EXISTS is the read's business, and a missing one comes
        // back as absent bytes there, exactly as a never-written shard entry does here.
        if shard.depth() == 0 {
            return Ok(Some(ByteRange::FromStart(0, None)));
        }
        let file = key_partial_decoder(&self.store, &item.key);
        let mut shard_shape = item.shape.clone();
        let mut offset = start;
        // (offset, length) of the level being descended INTO, absolute in the store value.
        let mut extent: Option<(u64, u64)> = None;
        // The subchunk indices taken so far. Only built below depth 0.
        let mut path: Vec<u64> = Vec::new();

        for depth in 0..shard.depth() {
            let level_shape = shard.subchunk_shape_at(depth);
            // The descent walks axis 0 and asks for `&[index, 0, 0, ...]`, which addresses
            // the right subchunk only if every other axis holds exactly ONE at this level.
            // Rechecked per level rather than assumed from the item: the grid can divide at
            // one depth and not another, and getting it wrong returns a WRONG chunk's bytes
            // rather than an error.
            if level_shape.len() != shard_shape.len()
                || level_shape
                    .iter()
                    .zip(shard_shape.iter())
                    .skip(1)
                    .any(|(sub, whole)| sub.get() != whole.get())
            {
                return Err(PyRuntimeError::new_err(format!(
                    "{}: level {depth} divides an axis after the first ({:?} within {:?}); \
                     this path descends on axis 0 alone",
                    item.key,
                    level_shape.iter().map(|d| d.get()).collect::<Vec<_>>(),
                    shard_shape.iter().map(|d| d.get()).collect::<Vec<_>>()
                )));
            }
            let subchunk = level_shape[0].get();
            let index = offset / subchunk;
            offset %= subchunk;
            // One entry per axis; zero everywhere but the split, because those axes hold a
            // single subchunk -- just checked.
            let mut grid_index = vec![0u64; level_shape.len()];
            grid_index[0] = index;

            let decoder = if depth == 0 {
                self.decoder_or_read(&self.shard_indexes, &mut decoders.shards, &item.key, || {
                    shard.level_decoder(
                        0,
                        key_partial_decoder(&self.store, &item.key),
                        item.shape.clone(),
                        &ctx.codec_options,
                    )
                })?
            } else {
                // A subshard's index is not its shard's, so the path taken to reach it is
                // part of the key.
                let (base, len) = extent.expect("a level below 0 has a parent extent");
                let key = (item.key.clone(), path.clone());
                self.decoder_or_read(
                    &self.subshard_indexes,
                    &mut decoders.subshards,
                    &key,
                    || {
                        let input =
                            Arc::new(ByteIntervalPartialDecoder::new(file.clone(), base, len));
                        shard.level_decoder(depth, input, shard_shape.clone(), &ctx.codec_options)
                    },
                )?
            };

            let Some(range) = decoder.subchunk_byte_range(&grid_index).map_codec_err()? else {
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
    #[allow(clippy::type_complexity)]
    fn locate_chunks<'a>(
        &self,
        shard: &ShardInfo,
        items: &'a [ChunkItem],
        ctx: &JobContext,
    ) -> PyResult<(Vec<(&'a ChunkItem, Option<ByteRange>)>, Vec<&'a ChunkItem>)> {
        let mut located = Vec::with_capacity(items.len());
        let mut declined = Vec::new();
        let mut decoders = CallDecoders::default();

        for item in items {
            if item.coords.is_none() {
                declined.push(item);
                continue;
            }
            // `locate` descends on axis 0 alone, so every other axis must hold the chunk
            // whole -- otherwise it would locate a different subchunk and report success.
            // Checked here as well as in Python, because `push_entry` is `#[pymethods]`.
            if !trailing_axes_are_whole(&item.chunk_subset, &item.shape) {
                return Err(PyRuntimeError::new_err(format!(
                    "{}: this path splits axis 0 and takes the rest whole, but the chunk \
                     subset is {} against a chunk of {:?}",
                    item.key,
                    item.chunk_subset,
                    item.shape.iter().map(|d| d.get()).collect::<Vec<_>>()
                )));
            }
            let start = item.chunk_subset.start().first().copied().unwrap_or(0);
            located.push((item, self.locate(shard, item, start, ctx, &mut decoders)?));
        }
        Ok((located, declined))
    }
}

/// Split the output into the disjoint piece each located chunk writes, in offset order.
///
/// Each piece comes from `DisjointBytes::take`, whose cursor only moves forward, so a second
/// claim on the same bytes is refused rather than aliased.
///
/// Returns the jobs to read, and the pieces of chunks that were never written, which need
/// only the fill value.
fn carve<'a>(
    output: &'a DisjointBytes<'a>,
    located: &[(&'a ChunkItem, Option<ByteRange>)],
    element_size: usize,
    ctx: &'a JobContext,
) -> PyResult<(Vec<Job<'a>>, Vec<&'a mut [u8]>)> {
    let mut order: Vec<usize> = (0..located.len()).collect();
    order.sort_by_key(|&i| output_offset(located[i].0));

    let mut jobs: Vec<Job<'a>> = Vec::with_capacity(order.len());
    let mut absent: Vec<&mut [u8]> = Vec::new();
    for &i in &order {
        let (item, range) = &located[i];
        let coords = coords_of(item)?;
        // WHERE a piece starts comes from `subset`, and HOW LONG it is comes from `coords`.
        // Nothing ties the two together: `ChunkItem` is constructible from Python and skips
        // the element-count check when coords are present. If they disagree, every later
        // piece is carved at the wrong offset and the read silently returns the right number
        // of wrong elements.
        // The output piece is handed out as ONE contiguous byte range, so the item's rows
        // must be contiguous in the output -- which they are exactly when every axis after
        // the first is taken whole. Anything else would carve a range that is not the item's.
        if !trailing_axes_are_whole(&item.subset, &item.array_shape) {
            return Err(PyRuntimeError::new_err(format!(
                "{}: this path carves one contiguous output range, so axes after the first \
                 must be whole, but the subset is {} against an output of {:?}",
                item.key,
                item.subset,
                item.array_shape.iter().map(|d| d.get()).collect::<Vec<_>>()
            )));
        }
        // A coordinate stands for `run_len` elements now, so it is the PRODUCT that has to
        // match the output subset. Nothing ties `coords` to `subset` otherwise: `ChunkItem`
        // is constructible from Python and skips the element-count check when coords are
        // present, and a disagreement carves every later piece at the wrong offset.
        if (coords.len() as u64).checked_mul(item.run_len) != Some(item.subset.num_elements()) {
            return Err(PyRuntimeError::new_err(format!(
                "{} wants {} coordinates of {} elements but its output subset holds {}",
                item.key,
                coords.len(),
                item.run_len,
                item.subset.num_elements()
            )));
        }
        // `output_offset` saturates to `usize::MAX`, so an unchecked multiply wraps in
        // release and the wrapped value then slips past `take`'s bounds test.
        let (Some(start), Some(len)) = (
            output_offset(item).checked_mul(element_size),
            usize::try_from(item.run_len)
                .ok()
                .and_then(|run| coords.len().checked_mul(run))
                .and_then(|elements| elements.checked_mul(element_size)),
        ) else {
            return Err(PyRuntimeError::new_err(format!(
                "{} names an output offset or length too large to address",
                item.key
            )));
        };
        let Some(piece) = output.take(start, len) else {
            return Err(PyRuntimeError::new_err(format!(
                "{} claims output bytes {start}..{}, which run backwards into a piece already \
                 handed out or past the buffer",
                item.key,
                start.saturating_add(len)
            )));
        };

        match range {
            Some(range) => jobs.push(Job {
                key: item.key.clone(),
                range: *range,
                out: piece,
                coords,
                run_len: item.run_len,
                // Sharded: the shard's inner chunk. Not sharded: the item's own chunk, which
                // is the whole store value and the whole decode.
                decode_shape: ctx
                    .shard
                    .subchunk_shape
                    .as_deref()
                    .unwrap_or(item.shape.as_slice()),
                may_be_absent: ctx.shard.depth() == 0,
                col_offsets: item.col_offsets.as_deref(),
                ctx,
            }),
            None => absent.push(piece),
        }
    }
    Ok((jobs, absent))
}

/// Hands out each byte range of the output at most once.
///
/// The output buffer is Python's, reached through a shared handle, so there is no `&mut` to
/// take for the whole of it -- and taking one anyway would claim `noalias` over memory a
/// Python caller may still hold (an `out=` array is theirs, not ours). This vends the pieces
/// instead: each range is handed out once, and what comes back IS a `&mut [u8]`, so from
/// that point the compiler enforces the disjointness rather than anyone asserting it.
///
/// The one `unsafe` is inside `take`, and its argument is local: `cursor` only moves
/// forward, so no two ranges it returns can intersect.
struct DisjointBytes<'a> {
    slice: UnsafeCellSlice<'a, u8>,
    len: usize,
    /// A `Cell` so `take` can vend from a SHARED reference. It has to: each piece borrows
    /// from `&self`, and `&mut self` would allow only one to be alive at a time.
    cursor: Cell<usize>,
    /// Bytes actually vended. Separate from `cursor` because `cursor` jumps over a gap and
    /// would report it as covered.
    covered: Cell<usize>,
}

impl<'a> DisjointBytes<'a> {
    fn new(slice: UnsafeCellSlice<'a, u8>, len: usize) -> Self {
        Self {
            slice,
            len,
            cursor: Cell::new(0),
            covered: Cell::new(0),
        }
    }

    /// How many bytes have actually been handed out. NOT `cursor`: that is the end of the
    /// last range, so it counts a gap as covered and the completeness check would pass with
    /// a hole in the middle of the output.
    fn covered(&self) -> usize {
        self.covered.get()
    }

    /// `None` if the range runs backwards into one already handed out, or past the buffer.
    ///
    /// Callers must therefore ask in non-decreasing order of `start`, which `carve` does by
    /// sorting first.
    // Making a `&mut` from a `&` is the whole job, and the lint cannot see why it is sound:
    // the guarantee is `cursor`, not the type. `UnsafeCellSlice::get_mut` carries the same
    // allow for the same reason.
    #[allow(clippy::mut_from_ref)]
    fn take(&self, start: usize, len: usize) -> Option<&mut [u8]> {
        let end = start.checked_add(len)?;
        if start < self.cursor.get() || end > self.len {
            return None;
        }
        self.cursor.set(end);
        self.covered.set(self.covered.get() + len);
        // SAFETY: `start >= cursor` and `cursor` is the end of the last range handed out, so
        // this range overlaps none of them; `end <= len` keeps it inside the buffer.
        unsafe { self.slice.get_mut(start..end) }
    }
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

/// Where an item's elements land in the output, as a FLAT element offset.
///
/// The subset starts at 0 on every axis after the first and spans it whole -- `carve`
/// rechecks that -- so the output rows of one item are contiguous and the offset is just
/// the row index times the row length.
fn output_offset(item: &ChunkItem) -> usize {
    let row = usize::try_from(item.subset.start().first().copied().unwrap_or(0))
        .unwrap_or(usize::MAX);
    let run: usize = item.array_shape[1..]
        .iter()
        .try_fold(1usize, |acc, d| {
            usize::try_from(d.get()).ok().and_then(|d| acc.checked_mul(d))
        })
        .unwrap_or(usize::MAX);
    row.saturating_mul(run)
}

/// Is every axis after the first taken WHOLE, against `shape`?
///
/// The premise of the rank-N chunk-unit path: with the trailing axes complete, one selected
/// index is one contiguous run, the output piece of an item is one contiguous range, and
/// `locate`'s descent along axis 0 alone addresses the right subchunk.
fn trailing_axes_are_whole(subset: &zarrs::array::ArraySubset, shape: &[NonZeroU64]) -> bool {
    subset.dimensionality() == shape.len()
        && subset
            .start()
            .iter()
            .zip(subset.shape())
            .zip(shape)
            .skip(1)
            .all(|((&start, &extent), whole)| start == 0 && extent == whole.get())
}

/// Live workers across every in-flight call, counted separately per kind.
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

/// The default ceiling on live workers of ONE kind, across every in-flight call.
///
/// The available parallelism. Raise it per read; see `codec_pipeline.read_worker_ceiling`.
fn default_ceiling() -> usize {
    std::thread::available_parallelism().map_or(8, std::num::NonZeroUsize::get)
}

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

/// The concurrency ONE call may use, read from `zarr.config` when that call starts.
///
/// Per call, not per array: read at array open these would be frozen for the array's life,
/// and `with zarr.config.set(...)` around a read would silently do nothing.
#[derive(Clone, Copy)]
pub(crate) struct CallWidths {
    /// Readers one call may run at once.
    pub(crate) read: usize,
    /// Decoders one call may run at once.
    pub(crate) decode: usize,
    /// Live READERS across every in-flight call.
    pub(crate) read_ceiling: usize,
    /// Live DECODERS across every in-flight call.
    ///
    /// Separate from the reader ceiling: a read waits on storage, a decode occupies a core.
    pub(crate) decode_ceiling: usize,
}

impl CallWidths {
    /// `None` for any of them takes the default: the given parallelism for the per-call
    /// widths, and [`default_ceiling`] for either ceiling.
    pub(crate) fn new(
        read: Option<usize>,
        decode: Option<usize>,
        read_ceiling: Option<usize>,
        decode_ceiling: Option<usize>,
        parallelism: usize,
    ) -> Self {
        Self {
            read: read.unwrap_or(parallelism).max(1),
            decode: decode.unwrap_or(parallelism).max(1),
            read_ceiling: read_ceiling
                .filter(|c| *c > 0)
                .unwrap_or_else(default_ceiling),
            decode_ceiling: decode_ceiling
                .filter(|c| *c > 0)
                .unwrap_or_else(default_ceiling),
        }
    }
}

/// Calls in flight, so a call takes a SHARE of the ceiling rather than racing for it.
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

/// One worker's permit, released when THAT worker exits, not when the call does.
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
/// It grows from here -- see the widening loop in `retrieve_chunk_units`.
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
struct Job<'a> {
    key: StoreKey,
    range: ByteRange,
    out: &'a mut [u8],
    coords: &'a [u64],
    /// Elements per coordinate; 1 on the 1-D path. See `ChunkItem::run_len`.
    run_len: u64,
    /// The unit that gets decoded into scratch: the shard's inner chunk where the array is
    /// sharded, the item's own chunk where it is not.
    decode_shape: &'a [NonZeroU64],
    /// The columns each coordinate takes, when they are scattered rather than consecutive --
    /// `oindex[rows, cols]`. `None` is a contiguous run, which is every other case.
    col_offsets: Option<&'a [u64]>,
    /// Whether missing bytes are ordinary here. A shard index that named this chunk and then
    /// did not have it means someone rewrote the store mid-read, which is worth failing on.
    /// An unsharded chunk has no index to consult, so its key simply may not exist yet, and
    /// that is a fill, not a fault.
    may_be_absent: bool,
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
            Ok(bytes) => {
                if let Err(returned) = decodes.send((job, bytes)) {
                    // Every decoder is gone. Returning silently would leave this job's bytes
                    // of the output at whatever `np.empty` left, and the call would report
                    // success.
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
        if job.may_be_absent {
            return fill(job.out, &ctx.fill_value, size);
        }
        return Err(format!("{} vanished between index and read", job.key));
    };

    let shape = job.decode_shape;
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
    match job.col_offsets {
        Some(cols) => gather_columns(scratch, job.coords, cols, job.out, size),
        None => gather(scratch, job.coords, job.run_len, job.out, size),
    }
        .map_err(|e| format!("{}: {e}", job.key))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vendor is what the whole path's disjointness rests on, so its refusals are
    /// pinned here rather than left to the caller that happens to ask in order.
    #[test]
    fn bytes_are_vended_once_and_forwards() {
        let mut buffer = vec![0u8; 16];
        let slice = UnsafeCellSlice::new(buffer.as_mut_slice());
        let bytes = DisjointBytes::new(slice, 16);

        let first = bytes.take(0, 4).expect("in bounds");
        let second = bytes.take(4, 4).expect("adjacent, not overlapping");
        // Two live `&mut` into one buffer, which is the point: they cannot alias.
        first[0] = 1;
        second[0] = 2;

        assert!(bytes.take(4, 4).is_none(), "a range already handed out");
        assert!(
            bytes.take(0, 2).is_none(),
            "backwards into one already handed out"
        );
        assert!(bytes.take(8, 9).is_none(), "past the end of the buffer");
        assert!(bytes.take(usize::MAX, 1).is_none(), "start + len overflows");
        // Vending over a gap is allowed -- the caller may skip bytes it does not own -- but
        // it must not COUNT as covered, or the completeness check in `retrieve_chunk_units`
        // would pass with a hole and hand `np.empty` contents back as data.
        assert!(bytes.take(12, 4).is_some(), "forwards over a gap");
        assert_eq!(
            bytes.covered(),
            12,
            "4 + 4 + 4 vended; the 4-byte hole at 8..12 is not covered"
        );

        assert_eq!(buffer[0], 1);
        assert_eq!(buffer[4], 2);
    }

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

        // THE TWO BUDGETS ARE SEPARATE. Readers are exhausted here; decoders must not notice.
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
