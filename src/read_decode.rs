//! Reading and decoding the innermost chunks of one call, concurrently.
//!
//! One job per READ -- an innermost chunk, or one run of rows inside a chunk whose inner
//! chain is plain bytes: a reader does the blocking byte-range read, a decode worker
//! decodes the chunk and copies out the elements the selection wants. The two are separate
//! because a read waits on storage and a decode occupies a core, so the useful number of
//! each is different -- hence a separate size for each.
//!
//! Workers belong to the CALL. `std::thread::scope` cannot exit until they finish, so a job
//! can hold `&mut [u8]` into the caller's output rather than a raw pointer, and the join is
//! the barrier. `DisjointBytes` vends each range of the output once, in increasing offset
//! order, and what it returns is a `&mut [u8]`, so two jobs cannot name the same bytes.
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyAnyMethods;
use pyo3::{PyResult, Python};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{
    ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset, ArrayToBytesCodecTraits,
    CodecOptions, FillValue, ravel_indices,
};
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{MaybeBytes, ReadableWritableListableStorage, StoreKey};

use crate::CodecPipelineImpl;
use crate::chunk_item::ChunkItem;
use crate::shard_index::ShardInfo;
use zarrs::array::codec::api::ByteIntervalPartialDecoder;
use zarrs::array::codec::array_to_bytes::sharding::ShardingPartialDecoder;

use crate::utils::{
    PyCodecErrExt as _, PyErrExt as _, gather, gather_pieces, gather_runs, key_partial_decoder,
    offset_runs,
};

/// The per-array state a decode needs, shared by every job of a call.
struct JobContext {
    /// See `CodecPipelineImpl::inner_chunk_is_plain_bytes`. When true a row's bytes are addressable
    /// inside its chunk, so a job reads the ROW rather than the chunk holding it.
    by_row: bool,
    shard: Arc<ShardInfo>,
    store: ReadableWritableListableStorage,
    codec_options: CodecOptions,
    element_size: usize,
    /// What an absent chunk contributes. Needed in the workers, not just at carve time,
    /// because an UNSHARDED chunk's absence is only discovered by the read.
    fill_value: FillValue,
    /// Whether missing bytes are ordinary. A shard index that named a chunk which is then
    /// missing means the store changed under the read, and that is worth failing on; an
    /// unsharded chunk has no index to consult, so its key simply may not exist yet.
    ///
    /// Per CALL, not per job: an array is sharded or it is not. It lived on `Job` for one
    /// commit and cost measurable throughput on the loader, because `Job` crosses TWO channels
    /// per job -- dispatch to reader, reader to decoder -- so every byte added to it is copied
    /// twice per job, ~8,000 times a call.
    may_be_absent: bool,
    /// See [`MAX_ROW_READS`]. Per call, so a caller can disable the row path for one read.
    max_row_reads: usize,
    /// The unit decoded into scratch: the shard's inner chunk where the array is sharded, the
    /// CHUNK where it is not. Also per call -- an array's chunks are all one shape -- so it is
    /// resolved once here from the first item rather than carried on every `Job`.
    decode_shape: Vec<NonZeroU64>,
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
    /// coordinates wanted from it. Returns the items this path could not take -- there is no
    /// second path to hand them to, so the caller turns them into an error rather than leave
    /// their output bytes unwritten.
    pub(crate) fn retrieve_chunk_units<'a>(
        &self,
        shard: &Arc<ShardInfo>,
        items: &'a [ChunkItem],
        output: UnsafeCellSlice<'_, u8>,
        output_len: usize,
        config: ReadConfig,
        codec_options: &CodecOptions,
    ) -> PyResult<Vec<&'a ChunkItem>> {
        let element_size = self.element_size()?;
        let ctx = JobContext {
            by_row: self.inner_chunk_is_plain_bytes,
            max_row_reads: config.max_row_reads,
            shard: shard.clone(),
            store: self.store.clone(),
            codec_options: (*codec_options).with_concurrent_target(1),
            element_size,
            fill_value: self.fill_value.clone(),
            may_be_absent: shard.depth() == 0,
            // Sharded: the shard says. Not sharded: any item does, because chunk shapes are
            // uniform across an array -- and an empty batch never reaches a decode.
            decode_shape: shard.subchunk_shape.as_ref().map_or_else(
                || items.first().map(|i| i.shape.clone()).unwrap_or_default(),
                |shape| shape.to_vec(),
            ),
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

        // Two persistent work-stealing pools, and capacity is never divided between calls: a
        // free worker takes the next task, whoever queued it, so a call with more chunks
        // simply gets more workers.
        //
        // READS AND DECODES ARE SEPARATE POOLS. A read blocks on storage; a decode occupies a
        // core. A reader parked on Lustre must occupy a READER, never a decode worker, or one
        // slow shard starves every decode in the process.
        //
        // The scopes nest so a reader hands its chunk straight to the decode pool, and
        // `in_place_scope` keeps the caller ON this thread: only work it spawns runs in the
        // pool, and the caller blocks until the scope ends rather than stealing from it. That
        // is deliberate and load-bearing. An owner thread that DID steal is what deadlocked an
        // earlier design -- it picked up a decode task and parked on a channel only it could
        // have drained. Do not "fix" this to `scope`, and do not call this from inside a rayon
        // worker, which flips the same latch to the stealing form.
        // Both scopes block until their tasks finish, which is what keeps `&'a mut [u8]` into
        // the caller's numpy buffer valid without a raw pointer or a completion latch.
        let failure: Mutex<Option<String>> = Mutex::new(None);

        let (read_pool, decode_pool) = pools(config.read_pool_size, config.decode_pool_size)?;
        decode_pool.in_place_scope(|dec| {
            read_pool.in_place_scope(|rd| {
                for job in jobs {
                    let (failure, ctx) = (&failure, &ctx);
                    rd.spawn(move |_| read_one(job, dec, failure, ctx));
                }
            });
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
            INDEX_CALL_HITS.fetch_add(1, Ordering::Relaxed);
            return Ok(found.clone());
        }
        if self.cache_shard_indexes {
            let found = array_cache
                .lock()
                .expect("shard index cache poisoned")
                .get(key)
                .cloned();
            if let Some(found) = found {
                INDEX_ARRAY_HITS.fetch_add(1, Ordering::Relaxed);
                call_cache.insert(key.clone(), found.clone());
                return Ok(found);
            }
        }
        INDEX_BUILDS.fetch_add(1, Ordering::Relaxed);
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
        start: &[u64],
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
        let mut offset: Vec<u64> = start.to_vec();
        // (offset, length) of the level being descended INTO, absolute in the store value.
        let mut extent: Option<(u64, u64)> = None;
        // The subchunk indices taken so far. Only built below depth 0.
        let mut path: Vec<u64> = Vec::new();

        for depth in 0..shard.depth() {
            let level_shape = shard.subchunk_shape_at(depth);
            // The descent walks EVERY axis. `subchunk_byte_range` has always taken a full
            // grid index; this used to fill axis 0 and leave the rest zero, which addressed
            // the right subchunk only when every other axis held exactly one -- the guard
            // that made a shard dividing a trailing axis decline outright.
            //
            // Arity is still checked per level: the grid can divide at one depth and not
            // another, and a mismatched index returns a WRONG chunk's bytes rather than an
            // error.
            if level_shape.len() != shard_shape.len() || level_shape.len() != offset.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "{}: level {depth} has {} axes against a chunk of {} and a position of {}",
                    item.key,
                    level_shape.len(),
                    shard_shape.len(),
                    offset.len()
                )));
            }
            let mut grid_index = vec![0u64; level_shape.len()];
            for axis in 0..level_shape.len() {
                let subchunk = level_shape[axis].get();
                grid_index[axis] = offset[axis] / subchunk;
                offset[axis] %= subchunk;
            }

            let decoder = if depth == 0 {
                self.decoder_or_read(
                    &self.shard_decoders,
                    &mut decoders.shards,
                    &item.key,
                    || {
                        shard.level_decoder(
                            0,
                            key_partial_decoder(&self.store, &item.key),
                            item.shape.clone(),
                            &ctx.codec_options,
                        )
                    },
                )?
            } else {
                // A subshard's index is not its shard's, so the path taken to reach it is
                // part of the key.
                let (base, len) = extent.expect("a level below 0 has a parent extent");
                let key = (item.key.clone(), path.clone());
                self.decoder_or_read(
                    &self.subshard_decoders,
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
                // Every axis, not just the split. The path is the cache key for a
                // subshard's decoder, and two positions differing only on a trailing axis
                // would otherwise collide on it -- returning the wrong subshard's index.
                path.extend_from_slice(&grid_index);
            }
        }
        // The item must lie inside the ONE inner chunk just located. `offset` is now its
        // position within that chunk, and `shard_shape` the chunk's own extent.
        //
        // Checked here rather than left to the caller, and it is not a weaker check than
        // descending axis 0 with whole trailing axes would be. Without it an
        // item claiming rows 0..8 x cols 0..12 of a shard whose inner chunk is 8x6 locates
        // chunk (0,0), and its coordinates -- built for a 12-wide row -- address exactly the
        // 48 elements that chunk holds. In bounds, wrong data, no error. `push_indices` takes
        // arbitrary arguments from Python, so this is a trust boundary rather than an
        // invariant the caller can be assumed to keep.
        // AXIS 0 MUST START AT THE CHUNK. Every correct caller groups by the real inner
        // extent, so a group's first index is a multiple of it and this offset is zero. A
        // caller passing an extent that DIVIDES the real one -- inner 4 where the chunk is 8
        // -- gets a group at `lo = 4`, which fits inside the chunk and so passes the extent
        // check below, but whose element offsets were built for a 4-row unit and address the
        // FIRST four rows of an eight-row chunk. In bounds, wrong data, no error.
        if offset.first().is_some_and(|at| *at != 0) {
            return Err(PyRuntimeError::new_err(format!(
                "{}: the item starts {} into its inner chunk on the split axis, so the extent \
                 it was grouped by is not the one that gets decoded",
                item.key, offset[0],
            )));
        }
        let held = item.chunk_subset.shape();
        if held.len() != offset.len()
            || held
                .iter()
                .zip(offset.iter())
                .zip(shard_shape.iter())
                .any(|((want, at), extent)| at + want > extent.get())
        {
            return Err(PyRuntimeError::new_err(format!(
                "{}: the item spans {} from {:?} within an inner chunk of {:?}, so it is not \
                 one decode unit",
                item.key,
                item.chunk_subset,
                offset,
                shard_shape.iter().map(|d| d.get()).collect::<Vec<_>>()
            )));
        }
        Ok(extent.map(|(base, len)| ByteRange::FromStart(base, Some(len))))
    }

    /// Where each item's innermost chunk lives, from its shard's own offset/size table.
    ///
    /// The descent divides on every axis, so an item is not required to take the trailing axes
    /// whole -- but it IS required to lie within one inner chunk, which `locate` checks.
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
            if item.element_offsets.is_none() {
                declined.push(item);
                continue;
            }
            // The whole position, not just axis 0: the descent divides on every axis now,
            // so a shard that splits a trailing one is addressed rather than refused.
            let start = item.chunk_subset.start().to_vec();
            located.push((item, self.locate(shard, item, &start, ctx, &mut decoders)?));
        }
        Ok((located, declined))
    }
}

/// The output byte ranges an item fills, ascending.
///
/// ONE range while every axis after the first is taken whole -- every rank-1 read, so the CSR
/// path always gets one. When a shard divides a trailing axis the item fills a sub-box, which
/// in row-major order is one range per row at the array's row stride.
fn output_pieces(item: &ChunkItem, element_size: usize) -> PyResult<Vec<(usize, usize)>> {
    let full: Vec<u64> = item.array_shape.iter().map(|d| d.get()).collect();
    let start = item.subset.start();
    let shape = item.subset.shape();
    if start.len() != full.len() || shape.len() != full.len() {
        return Err(PyRuntimeError::new_err(format!(
            "{}: subset {} does not match an output of {full:?}",
            item.key, item.subset
        )));
    }
    // The arithmetic itself is zarrs': `contiguous_linearised_indices` walks the subset in C
    // order and merges whole trailing axes into one run exactly as this did by hand -- the
    // reverse stride fold, the `elem_offset`, and the whole-trailing special case are all
    // that walk. It also rechecks that `full` encapsulates the subset, which is the bounds
    // half of the guard above.
    //
    // zarrs does not REFUSE a strided sub-box, it emits more runs for one -- so the refusal
    // below is a COUNT, read off the walk zarrs already did rather than predicted by a second
    // copy of the contiguity rule. An item's output is one run per axis-0 index, or a single
    // run when the trailing axes are whole and the rows are therefore adjacent. Anything else
    // is strided within a row, and vending it as one run per index would claim bytes belonging
    // to the next item -- which `DisjointBytes` reports as a backwards claim, naming the
    // symptom rather than this.
    //
    // Checked HERE because this is the funnel: `push_indices`, `push_span`, `push_grid`,
    // `push_points` and a hand-built `ChunkItem` all reach bytes through it.
    let runs = item
        .subset
        .contiguous_linearised_indices(&full)
        .map_err(|e| PyRuntimeError::new_err(format!("{}: {e}", item.key)))?;
    let one_per_row = u64::try_from(runs.len()).is_ok_and(|n| n == item.subset.shape()[0]);
    if runs.len() != 1 && !one_per_row {
        return Err(PyRuntimeError::new_err(format!(
            "{}: output {:?} of {:?} is strided within one index, and an item's output is \
             vended as one run per index",
            item.key,
            &item.subset.shape()[1..],
            &full[1..]
        )));
    }
    // Fixed across the iteration by construction, so it is read once.
    let piece_bytes = usize::try_from(runs.contiguous_elements())
        .ok()
        .and_then(|r| r.checked_mul(element_size))
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!("{}: output run too large to address", item.key))
        })?;
    runs.iter()
        .map(|(index, _)| {
            usize::try_from(index)
                .ok()
                .and_then(|i| i.checked_mul(element_size))
                .map(|at| (at, piece_bytes))
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "{}: output offset too large to address",
                        item.key
                    ))
                })
        })
        .collect()
}

/// Split the output into the disjoint piece each located chunk writes, and turn each into a
/// job, in output-offset order.
///
/// Every piece comes from `DisjointBytes::take`, whose cursor only moves forward, so a second
/// claim on the same bytes is refused rather than aliased -- which is what the `unsafe` in
/// `take` rests on, and why the pieces are cut here rather than by each job for itself.
///
/// Returns the jobs to read, and the pieces of chunks that were never written, which need no
/// read at all -- only the fill value.
fn carve<'a>(
    output: &'a DisjointBytes<'a>,
    located: &[(&'a ChunkItem, Option<ByteRange>)],
    element_size: usize,
    ctx: &'a JobContext,
) -> PyResult<(Vec<Job<'a>>, Vec<&'a mut [u8]>)> {
    // Pass 1: what each item needs, and the element-count agreement that used to be checked
    // inline. Nothing is vended yet.
    let mut plan: Vec<(usize, Vec<(usize, usize)>)> = Vec::with_capacity(located.len());
    for (i, (item, _)) in located.iter().enumerate() {
        let element_offsets = element_offsets_of(item)?;
        // WHERE a piece starts comes from `subset`, and HOW LONG it is comes from `element_offsets`.
        // Nothing ties the two together: `ChunkItem` is constructible from Python and skips
        // the element-count check when element_offsets are present. If they disagree, a piece is
        // carved at the wrong offset and the read returns the right number of wrong elements.
        if (element_offsets.len() as u64).checked_mul(item.run_len)
            != Some(item.subset.num_elements())
        {
            return Err(PyRuntimeError::new_err(format!(
                "{} wants {} coordinates of {} elements but its output subset holds {}",
                item.key,
                element_offsets.len(),
                item.run_len,
                item.subset.num_elements()
            )));
        }
        plan.push((i, output_pieces(item, element_size)?));
    }

    // Pass 2: vend every piece of every item in ASCENDING output order.
    //
    // `DisjointBytes` moves a cursor forward only, which is what makes the disjointness a
    // fact rather than an assertion. Items alone cannot be put in that order once a shard
    // divides a trailing axis: the item for columns 0..6 and the item for columns 6..12
    // alternate row by row through the output. Sorting the PIECES restores it.
    let mut vend: Vec<(usize, usize, usize)> = plan
        .iter()
        .flat_map(|(i, pieces)| pieces.iter().map(move |&(at, len)| (at, len, *i)))
        .collect();
    vend.sort_unstable_by_key(|&(at, _, _)| at);
    let mut taken: Vec<Vec<&'a mut [u8]>> = (0..located.len()).map(|_| Vec::new()).collect();
    for (at, len, i) in vend {
        let Some(piece) = output.take(at, len) else {
            return Err(PyRuntimeError::new_err(format!(
                "{} claims output bytes {at}..{}, which run backwards into a piece already \
                 handed out or past the buffer",
                located[i].0.key,
                at.saturating_add(len)
            )));
        };
        taken[i].push(piece);
    }

    let mut jobs: Vec<Job<'a>> = Vec::with_capacity(located.len());
    let mut absent: Vec<&'a mut [u8]> = Vec::new();
    // Jobs stay in ascending output order: readers take them in turn, and on high-latency
    // storage that keeps the order they arrive in close to the order they are wanted.
    let mut order: Vec<usize> = (0..located.len()).collect();
    order.sort_by_key(|&i| output_offset(located[i].0));
    for i in order {
        let (item, range) = &located[i];
        let pieces = std::mem::take(&mut taken[i]);
        // A range is the chunk's place in its shard; its absence means the chunk was never
        // written, and the output it owns is filled rather than read.
        match range {
            // One job per ROW, each reading exactly its own bytes.
            //
            // Only when the chunk is a plain byte tiling, so a row's offset inside it is
            // arithmetic: `coord` is already the row's element offset within the chunk, and
            // `run_len` its length. Measured at scale, 8,192 rows this way take 628 ms
            // against 1121 for the chunks holding them -- the request COUNT is the same
            // either way, so all that changes is how many bytes each one moves.
            //
            // The pieces are taken in coordinate order, which is ascending, so
            // `DisjointBytes` still vends each byte once and coverage is still checked.
            // ONE OUTPUT PIECE and no grid, which is the whole admission rule. That covers
            // every rank-1 read and every read whose trailing axes are whole -- and also a
            // BANDED item of a single row, where `element_offsets[0]` is a column offset inside the
            // row rather than a row start and `run_len` is the band width rather than the row
            // width. The arithmetic is the same either way, which is why the band needs no
            // second implementation; a scattered banded selection reaches it often, since
            // rows in different chunks each give a one-row item.
            //
            // A grid item carries its own per-element offsets, and a banded item of SEVERAL
            // rows fills one output range per row. Neither is a single contiguous claim, so
            // both take the ordinary path.
            Some(range)
                if ctx.by_row
                    // Zero DISABLES, which the threshold alone does not say: an item with no
                    // coordinates is 0 reads, and `0 <= 0` would take the path the knob was
                    // set to refuse. Nothing builds such an item today -- `push_span` returns
                    // early on an empty count -- so this makes the documented behaviour true
                    // by construction rather than by the absence of a caller.
                    && ctx.max_row_reads > 0
                    && pieces.len() == 1
                    && item.grid.is_none()
                    && row_read_count(element_offsets_of(item)?, item.run_len) <= ctx.max_row_reads =>
            {
                let piece = pieces.into_iter().next().expect("length checked");
                row_jobs(
                    item,
                    *range,
                    piece,
                    element_offsets_of(item)?,
                    element_size,
                    ctx,
                    &mut jobs,
                )?;
            }
            Some(range) => {
                CHUNK_JOBS.fetch_add(1, Ordering::Relaxed);
                jobs.push(Job {
                    key: item.key.clone(),
                    range: *range,
                    by_row: false,
                    out: pieces,
                    element_offsets: element_offsets_of(item)?,
                    run_len: item.run_len,
                    grid: item.grid.as_ref().map(|(starts, run)| (&starts[..], *run)),
                    ctx,
                });
            }
            None => absent.extend(pieces),
        }
    }
    Ok((jobs, absent))
}

/// Jobs that took the ROW path, and jobs that read a whole chunk, since the run began.
///
/// The project rule -- a knob that was set is not a knob that arrived -- applied to a code
/// path. A gate that silently refuses everything is indistinguishable from a gate that is
/// working: values stay correct either way and only the throughput differs, which reads as
/// "the row path did not pay" rather than "the row path was never taken". Both failures have
/// already happened here once.
pub(crate) static ROW_JOBS: AtomicU64 = AtomicU64::new(0);
pub(crate) static CHUNK_JOBS: AtomicU64 = AtomicU64::new(0);

/// How many READS this chunk's rows become once consecutive ones are merged.
///
/// The count that matters is runs, not rows: 64 consecutive rows are ONE read, and 64
/// scattered ones are 64. See [`offset_runs`] for what counts as consecutive; this is the gate, and
/// `row_jobs` emits exactly the runs it counts here -- from the same walk, so the gate
/// cannot come to disagree with what it admits.
pub(crate) fn row_read_count(element_offsets: &[u64], run_len: u64) -> usize {
    offset_runs(element_offsets, run_len).count()
}

/// Default for `codec_pipeline.max_row_reads_per_chunk`.
///
/// The row path reads a row's exact bytes instead of the chunk around it, which trades BYTES
/// for REQUESTS -- and requests are the scarce resource, since a row costs nearly what the
/// chunk holding it costs to fetch. Hence a PER-ITEM gate: take it only where a chunk's wanted
/// rows collapse to a handful of reads. Two is measured; zero disables the path and costs ~75%
/// on an uncompressed scattered draw. See README.
const MAX_ROW_READS: usize = 2;

/// One job per RUN of consecutive rows, each reading exactly its own bytes, for a chunk that
/// is a plain byte tiling.
///
/// `coord` is already the row's element offset within the chunk and `run_len` its length, so
/// the row's byte range is arithmetic. Measured at scale, 8,192 rows read this way take
/// 628 ms against 1121 for the chunks holding them: the request COUNT is the same either way,
/// and only the bytes each one moves change.
///
/// `piece` is the item's single contiguous claim, split here rather than re-claimed, so the
/// vend-once cursor still sees exactly one take per item and coverage is still checked.
fn row_jobs<'a>(
    item: &'a ChunkItem,
    range: ByteRange,
    piece: &'a mut [u8],
    element_offsets: &'a [u64],
    element_size: usize,
    ctx: &'a JobContext,
    jobs: &mut Vec<Job<'a>>,
) -> PyResult<()> {
    let ByteRange::FromStart(base, chunk_bytes) = range else {
        return Err(PyRuntimeError::new_err(format!(
            "{}: the row path needs a FromStart range, got {range:?}",
            item.key
        )));
    };
    let row_bytes = usize::try_from(item.run_len)
        .ok()
        .and_then(|r| r.checked_mul(element_size))
        .ok_or_else(|| PyRuntimeError::new_err(format!("{}: row too large", item.key)))?;
    // CONSECUTIVE rows are one range, not one each. Without this a selection of 8-row blocks
    // issues 8 requests of 8 KiB where one of 64 KiB would do, and requests are the scarce
    // resource. The rows were always adjacent; the code just did not look.
    let mut rest = piece;
    for run in offset_runs(element_offsets, item.run_len) {
        let span = row_bytes
            .checked_mul(run.len())
            .ok_or_else(|| PyRuntimeError::new_err(format!("{}: run too large", item.key)))?;
        let (run_out, tail) = rest.split_at_mut(span.min(rest.len()));
        rest = tail;
        // `checked_mul`, like every other offset computation here. A wrap would produce a
        // small `within` that then PASSES the bound below, land inside a real chunk, and come
        // back at exactly the length asked for -- which is all `decode_one`'s row branch
        // compares. No selection builds offsets that large, but `push_indices` is reachable
        // from Python with arbitrary arguments.
        let within = element_offsets[run.start]
            .checked_mul(element_size as u64)
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("{}: element offset overflow", item.key))
            })?;
        // THE SOURCE SIDE, which nothing else checks. The output side is checked below (a
        // short piece leaves `rest` non-empty) and the chunk EXTENTS are checked in
        // `chunk_item.rs` as the description is built. But a chunk's bytes sit inside a shard,
        // so a coordinate past this chunk's own end still lands on real bytes -- the next
        // chunk's -- and comes back with exactly the length asked for. `decode_one`'s row
        // branch compares only lengths, so that read would be silently wrong data rather than
        // an error. One comparison here is what makes it an error instead.
        if let Some(len) = chunk_bytes
            && within.saturating_add(span as u64) > len
        {
            return Err(PyRuntimeError::new_err(format!(
                "{}: a run at byte {within} of {span} runs past the {len} bytes the chunk holds",
                item.key
            )));
        }
        let at = base
            .checked_add(within)
            .ok_or_else(|| PyRuntimeError::new_err(format!("{}: offset overflow", item.key)))?;
        jobs.push(Job {
            key: item.key.clone(),
            range: ByteRange::FromStart(at, Some(span as u64)),
            by_row: true,
            out: vec![run_out],
            element_offsets: &[],
            run_len: item.run_len,
            grid: None,
            ctx,
        });
        ROW_JOBS.fetch_add(1, Ordering::Relaxed);
    }
    if !rest.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "{}: {} output bytes left after {} rows",
            item.key,
            rest.len(),
            element_offsets.len()
        )));
    }
    Ok(())
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
    if size == 0 {
        // Both the modulo and the `chunks_exact_mut` below divide by this. No real data type
        // reaches here with a zero size, but this is the last divisor in the file that is not
        // a `NonZeroU64`, and the guard below would let a zero-length fill value through.
        return Err("an element size of zero has no fill".to_string());
    }
    if bytes.len() != size {
        return Err(format!(
            "the fill value is {} bytes wide, not the {size} one element holds",
            bytes.len()
        ));
    }
    // `chunks_exact_mut` would leave `out.len() % size` bytes exactly as `np.empty` gave
    // them and still return Ok -- uninitialised memory handed back as fill value. Every piece
    // reaching here is a whole number of elements today, because `output_pieces` builds them
    // as `contiguous_elements * element_size`; this is what keeps that true.
    if out.len() % size != 0 {
        return Err(format!(
            "a fill piece of {} bytes is not a whole number of {size}-byte elements",
            out.len()
        ));
    }
    for slot in out.chunks_exact_mut(size) {
        slot.copy_from_slice(bytes);
    }
    Ok(())
}

fn element_offsets_of(item: &ChunkItem) -> PyResult<&Arc<[u64]>> {
    item.element_offsets
        .as_ref()
        .ok_or_else(|| {
            format!(
                "{}: this path requires chunk-unit items, and this one carries no element \
                 offsets -- it was built by `ChunkItem::new`, which only the write path uses",
                item.key
            )
        })
        .map_py_err::<PyRuntimeError>()
}

/// Where an item's elements land in the output, as a FLAT element offset.
///
/// The C-order ravel of the subset's start, which is `ravel_indices`. This used to be the row
/// index times the row length, on the grounds that an item spans every axis after the first
/// whole -- but `output_pieces` admits a BANDED item, whose trailing start is not zero, and
/// two bands of one row then sorted equal. It orders jobs and never places bytes, so that was
/// harmless; the ravel is exact and does not need the invariant to hold.
///
/// `must_cast_slice` rather than a collect, because this is a sort key: `NonZeroU64` and `u64`
/// have the same layout, and `is_whole_chunk` already reads a shape this way.
fn output_offset(item: &ChunkItem) -> u64 {
    let shape = bytemuck::must_cast_slice::<_, u64>(&item.array_shape);
    ravel_indices(item.subset.start(), shape).unwrap_or(u64::MAX)
}

/// What the shard index cache did. Counted because nothing else can: a cache that is never
/// consulted passes every correctness test ever written.
///
/// Three outcomes, kept apart because they mean different things: the CALL cache is
/// unconditional and always safe, the ARRAY cache only engages on a read-only store, and a
/// BUILD is an index actually read and decoded.
pub(crate) static INDEX_CALL_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INDEX_ARRAY_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INDEX_BUILDS: AtomicU64 = AtomicU64::new(0);

/// The default size of either pool: the machine's parallelism.
///
/// A read waits on storage and a decode wants a core, so these are not the same quantity and
/// only the decode side is really bounded by cores. They share a default because the machine
/// is the only thing either can be guessed from; a caller that knows its storage sets the read
/// side higher.
fn default_pool_size() -> usize {
    std::thread::available_parallelism().map_or(8, std::num::NonZeroUsize::get)
}

/// The two pools, built once and shared by every call in the process.
///
/// Persistent and work-stealing, so capacity is never divided between calls: a free worker
/// takes the next task, whoever queued it. The alternative -- per-call threads drawing an
/// equal share of a global budget -- was measured and lost. The share is a snapshot taken
/// when every caller has just arrived and is never recomputed; integer division strands
/// capacity outright; and a partition blocked on I/O cannot lend to one that is starved.
///
/// SIZED ONCE, and this is the honest cost. `read_pool_size` and `decode_pool_size`
/// are read from `zarr.config` when the ARRAY is opened, but only the first read in the
/// process builds the pools; a
/// later `with zarr.config.set(...)` around a read resizes nothing. [`pool_sizes`] reports
/// what was actually built, so a caller can assert rather than assume.
/// REBUILT AFTER A FORK, which is the difference between a slow child and a hung one.
///
/// `fork()` copies the parent's memory but only the calling thread. A child that inherits a
/// built pool inherits worker threads that do not exist, and the first `in_place_scope` parks
/// the calling thread on a latch nothing will ever signal -- a permanent hang, not a
/// slowdown. This is not a corner case: the workload this path exists for is minibatch
/// loading, and `torch.utils.data.DataLoader(num_workers > 0)` forks by default on Linux. It
/// only hangs when the parent read BEFORE forking, so it is data-dependent and passes every
/// test that does not fork.
///
/// Keyed on the process id instead: a child sees a mismatch and builds its own. `OnceLock`
/// cannot express that -- it has no reset reachable from a `static` -- so this is a `Mutex`
/// around the pair, taken once per call to clone two `Arc`s.
///
/// The old pools are FORGOTTEN rather than dropped. `ThreadPool::drop` joins its workers, and
/// in the child those workers were never created, so dropping is the same hang by another
/// route. Leaking them is correct: they are a copy of memory this process never owned.
struct Pools {
    pid: u32,
    read: Arc<rayon::ThreadPool>,
    decode: Arc<rayon::ThreadPool>,
}

static POOLS: Mutex<Option<Pools>> = Mutex::new(None);

/// A size the operating system cannot give is the CALLER'S ERROR, not a process-ending panic.
///
/// `ThreadPoolBuilder::build` fails whenever a worker thread cannot be spawned --
/// `read_pool_size = 10_000_000`, or a modest value under a cgroup `pids.max`, which Slurm and
/// Kubernetes both set. This used to `expect`, and the panic unwound through the live `POOLS`
/// guard: the mutex was then poisoned, and every later read in the process died on an
/// `.expect("the pool lock is never held across a panic")` asserting the opposite of what had
/// happened. Lowering the config did not recover it; only a new interpreter did.
fn build_pool(size: usize, name: &'static str) -> PyResult<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(size)
        .thread_name(move |i| format!("zarrs-{name}-{i}"))
        .build()
        .map_err(|e| {
            PyValueError::new_err(format!(
                "codec_pipeline.{name}_pool_size = {size} could not be built: {e}"
            ))
        })
}

/// The `POOLS` guard, recovering rather than dying if some earlier caller poisoned it.
///
/// Poison means a panic happened while the lock was held. The data behind it is a
/// `Option<Pools>` -- there is no half-updated state a panic can leave, because the only write
/// is a whole-struct assignment -- so continuing is safe and refusing is not more correct, it
/// is just a second failure on top of the first.
fn pools_guard() -> std::sync::MutexGuard<'static, Option<Pools>> {
    POOLS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// The two pools, building them on first use in THIS process.
///
/// `read` blocks on storage and is sized independently of the core count for that reason;
/// `decode` occupies a core and is the one genuinely bounded by parallelism.
fn pools(
    read_size: usize,
    decode_size: usize,
) -> PyResult<(Arc<rayon::ThreadPool>, Arc<rayon::ThreadPool>)> {
    let mut guard = pools_guard();
    let pid = std::process::id();
    if guard.as_ref().is_none_or(|p| p.pid != pid) {
        // Both BEFORE either is stored, so a failure to build the second leaves the slot empty
        // rather than half-filled, and the next call retries from a clean state.
        let read = Arc::new(build_pool(read_size, "read")?);
        let decode = Arc::new(build_pool(decode_size, "decode")?);
        // See `Pools`: forget, never drop. Dropping joins threads that do not exist here.
        if let Some(stale) = guard.take() {
            std::mem::forget(stale);
        }
        *guard = Some(Pools { pid, read, decode });
    }
    let built = guard.as_ref().expect("just built");
    Ok((Arc::clone(&built.read), Arc::clone(&built.decode)))
}

/// Drop the pools so a `fork()` cannot inherit a LOCKED mutex, or threads that do not exist.
///
/// The pid check in [`pools`] rebuilds in a child -- but it runs INSIDE `POOLS.lock()`, and
/// `fork` copies a held mutex as held, with an owner thread that does not exist in the child.
/// A child that forked while any thread was inside `pools` therefore blocks on that lock for
/// ever, and the pid check never gets to run. `tests/test_fork_safety.py` cannot see this: it
/// forks with no read in flight.
///
/// So the fix is upstream of the lock. `python/zarrs/__init__.py` registers this with
/// `os.register_at_fork(before=...)`, which `multiprocessing` and therefore
/// `torch.utils.data.DataLoader` both honour. It blocks until no thread holds the lock, empties
/// the slot, and releases -- so the fork sees a free, empty mutex and both sides rebuild on
/// next use. Dropping here is correct and cheap: this is the PARENT, its workers exist, and a
/// fork happens once per epoch against pools that cost microseconds to rebuild.
pub(crate) fn release_pools_for_fork() {
    let mut guard = pools_guard();
    // Dropped, not forgotten -- the opposite of the child case. See `Pools`.
    *guard = None;
}

/// What the pools were actually BUILT with, or `None` where one has not been built yet.
///
/// Sizes are fixed at first use, so a caller that sets a size after the first read gets the
/// old value silently. Reporting the built size is what lets a benchmark tell "the knob did
/// not pay" from "the knob never arrived".
pub(crate) fn pool_sizes() -> (Option<usize>, Option<usize>) {
    let guard = pools_guard();
    match guard.as_ref() {
        // A pool built by a PARENT process is not this process's pool, and reporting its size
        // here would be the same lie `check_pool_size_arrived` exists to prevent.
        Some(p) if p.pid == std::process::id() => (
            Some(p.read.current_num_threads()),
            Some(p.decode.current_num_threads()),
        ),
        _ => (None, None),
    }
}

/// Say so when a size asked for is not the one the pools were built with.
///
/// The pools are sized by the FIRST read in the process, so a `zarr.config.set` around a later
/// read resizes nothing. That is the accepted cost of persistence; doing it SILENTLY is not.
/// From the outside "the knob did not pay" and "the knob never arrived" look identical, and
/// this project has published a number from each. [`pool_sizes`] lets a caller ask; this tells
/// one who did not.
///
/// A warning, not an error. The read is correct at the width already built, and a process that
/// opens a second array wanting a different width is doing something legitimate -- refusing it
/// would turn a sizing hint into a failed read. A caller needing the guarantee asserts on
/// [`pool_sizes`], which is what the benchmark does.
pub(crate) fn check_pool_size_arrived(
    py: Python<'_>,
    config: ReadConfig,
    strict: bool,
) -> PyResult<()> {
    // ONE acquisition of the pool lock, not one per knob. This runs on every read, and the
    // loop below used to call `pool_sizes()` in each arm.
    let (built_read, built_decode) = pool_sizes();
    for (built, asked, knob) in [
        (built_read, config.read_pool_size, "read_pool_size"),
        (built_decode, config.decode_pool_size, "decode_pool_size"),
    ] {
        // Only when a pool EXISTS and differs. Before the first read there is nothing to
        // contradict.
        let Some(built) = built.filter(|built| *built != asked) else {
            continue;
        };
        let message = format!(
            "codec_pipeline.{knob} = {asked} was ignored: the pool was built with {built} \
             threads by the first read in this process and cannot be resized. Set it before \
             the array that does the first read is opened, or call \
             zarrs.pool_sizes() for what was built."
        );
        // `codec_pipeline.strict` already means "do not paper over what this pipeline cannot
        // do" -- it turns a decline into a raise instead of a silent fallback to zarr-python.
        // A width the process cannot give is the same kind of thing, and a caller who asked
        // for strictness would rather find out here than infer it from a throughput number.
        if strict {
            return Err(PyValueError::new_err(message));
        }
        py.import("warnings")?.call_method1("warn", (message,))?;
    }
    Ok(())
}

/// One read's configuration: the pool sizes the ARRAY was opened with, plus this call's row
/// threshold. The two are read at different times on purpose -- see the fields.
#[derive(Clone, Copy)]
pub(crate) struct ReadConfig {
    /// Only the FIRST call's value in this process is used -- see [`Pools`].
    pub(crate) read_pool_size: usize,
    /// Only the FIRST call's value in this process is used -- see [`Pools`].
    pub(crate) decode_pool_size: usize,
    /// Reads a chunk may become before the row path is declined for it; see [`MAX_ROW_READS`].
    /// Per call, and honoured on every call rather than only the first.
    pub(crate) max_row_reads: usize,
}

/// A pool size as the pipeline will use it.
///
/// ZERO OR ABSENT MEANS "as many threads as the machine has", NOT "disabled". Worth saying
/// because its neighbours in the same config namespace read the other way: `file_handle_cache_size
/// = 0` and `max_row_reads_per_chunk = 0` both turn their feature off. A pool of no threads
/// cannot run a read at all, so there is nothing for zero to mean here except the default.
///
/// Public so the pipeline can resolve at OPEN, which is when these are read.
pub(crate) fn resolve_pool_size(asked: Option<usize>) -> usize {
    asked.filter(|n| *n > 0).unwrap_or_else(default_pool_size)
}

impl ReadConfig {
    /// The pool sizes as the ARRAY was opened with, already resolved, plus this call's row
    /// threshold. The two are read at different times on purpose -- see the fields.
    pub(crate) fn from_open(
        read_pool_size: usize,
        decode_pool_size: usize,
        max_row_reads: Option<usize>,
    ) -> Self {
        Self {
            read_pool_size,
            decode_pool_size,
            max_row_reads: max_row_reads.unwrap_or(MAX_ROW_READS),
        }
    }
}

/// One read, and the slice of the output its bytes belong in: an innermost chunk, or -- on
/// the row path -- one run of rows taken straight out of the chunk holding them.
struct Job<'a> {
    key: StoreKey,
    /// The chunk's byte range within its shard, or -- on the row path -- one run of rows'
    /// range inside that chunk.
    range: ByteRange,
    /// Row jobs carry the wanted bytes exactly: no decode, no scratch, no gather. Their
    /// `range` is the ROW's bytes inside the chunk rather than the whole chunk's.
    by_row: bool,
    /// The output ranges this chunk fills, ascending. ONE range while every axis after the
    /// first is taken whole -- which is every rank-1 read, so the CSR path always has one.
    /// A shard that divides a trailing axis gives an item one range per row instead.
    out: Vec<&'a mut [u8]>,
    element_offsets: &'a [u64],
    /// Elements per coordinate; 1 on the 1-D path. See `ChunkItem::run_len`.
    run_len: u64,
    /// Where each RUN starts inside a coordinate's elements, and how long a run is, when the
    /// wanted elements are not one consecutive span -- `oindex[rows, cols]` and any rank-N
    /// grid. `None` is a single contiguous run, which is every other case.
    grid: Option<(&'a [u64], u64)>,
    ctx: &'a JobContext,
}

/// Keep the FIRST failure; later ones are usually consequences of it.
fn record(failure: &Mutex<Option<String>>, message: String) {
    let mut slot = failure.lock().expect("failure slot poisoned");
    if slot.is_none() {
        *slot = Some(message);
    }
}

// Decode scratch, owned by the worker and kept for the life of the process.
//
// A decode decompresses a whole inner chunk (366 KiB sparse, 512 KiB dense here) before the
// wanted rows are copied out. Above glibc's 128 KiB threshold that allocation is an mmap, a
// memset and a fault per page, so it must not be paid per chunk. A rayon worker lives for the
// process, so its own buffer is the reuse -- no lock, and no way for it to silently not run.
//
// A plain comment, not a doc comment: rustc discards docs on a macro invocation and warns.
thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// One chunk: one store read, then its decode handed to the decode pool.
///
/// The decode goes to the DECODE pool, not this one. A reader is parked on storage for most
/// of its life and a decode wants a core, so mixing them means one slow shard holds a core a
/// ready chunk needs.
///
/// ONE JOB PER READ, deliberately. Packing jobs that share a shard into one `get_partial_many`
/// was measured and removed: it shares a file-handle lookup, not a syscall, since the store
/// maps each range to its own `read_exact_at`. It is worth revisiting on storage where a
/// request costs more than a file-handle lookup -- object stores, or anything over a network.
fn read_one<'scope, 'env>(
    job: Job<'env>,
    dec: &rayon::Scope<'scope>,
    failure: &'env Mutex<Option<String>>,
    ctx: &'env JobContext,
) where
    'env: 'scope,
{
    match ctx.store.get_partial(&job.key, job.range) {
        // `None` means the KEY is absent, which is a different thing from a range coming back
        // empty. `decode_one` already knows what an absent chunk contributes -- the fill
        // value, or an error where a shard index named it -- so that logic stays in one place.
        Ok(bytes) => spawn_decode(dec, job, bytes, failure),
        Err(e) => record(failure, format!("read {} failed: {e}", job.key)),
    }
}

/// One chunk's decode, on the decode pool.
///
/// The buffer is the worker's own `SCRATCH`, borrowed in place. A rayon worker outlives every
/// call, so one buffer serves every chunk that worker ever decodes.
fn spawn_decode<'scope, 'env>(
    dec: &rayon::Scope<'scope>,
    mut job: Job<'env>,
    bytes: MaybeBytes,
    failure: &'env Mutex<Option<String>>,
) where
    'env: 'scope,
{
    // EVERY job goes to the pool, including a row one whose "decode" is only a
    // `copy_from_slice`. Running row jobs inline on the reader was tried and reverted. An
    // earlier version of this comment gave magnitudes for that; they measured a different
    // build and were retracted. Re-measured properly the effect is inside the noise floor, so
    // the reason to keep the hand-off is the argument below, not a number.
    //
    // The argument for inlining was that a hand-off buys a queue push, a steal and a wake in
    // order to memcpy bytes already in this core's cache. That much is true and it is not the
    // point. A reader that copies inline stops issuing reads while it copies, so storage
    // latency and the copy serialise per reader instead of overlapping, and the decode pool
    // sits idle. The hand-off is not overhead -- it is what keeps reads in flight.
    dec.spawn(move |_| {
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            if let Err(e) = decode_one(&mut job, bytes, &mut scratch) {
                record(failure, e);
            }
        });
    });
}

/// Decode one innermost chunk into scratch, then gather the wanted elements into `out`.
fn decode_one(job: &mut Job<'_>, bytes: MaybeBytes, scratch: &mut Vec<u8>) -> Result<(), String> {
    let ctx = job.ctx;
    let size = ctx.element_size;
    let Some(bytes) = bytes else {
        if ctx.may_be_absent {
            for piece in &mut job.out {
                fill(piece, &ctx.fill_value, size)?;
            }
            return Ok(());
        }
        return Err(format!("{} vanished between index and read", job.key));
    };

    // A row job's read WAS the answer: its range is the row, not the chunk. No decode, no
    // scratch, no gather -- but not copy-free either, since `get_partial` hands back an owned
    // buffer and these bytes still have to be moved into the output.
    //
    // `out` is a Vec since the band split, so a row job's bytes are laid across its pieces in
    // order. `row_jobs` only ever builds ONE piece per job -- a row job is one run of one
    // row -- but walking the pieces costs nothing and means this cannot silently write only
    // the first if that ever stops being true.
    if job.by_row {
        let want: usize = job.out.iter().map(|p| p.len()).sum();
        if bytes.len() != want {
            return Err(format!(
                "{}: read {} bytes for an output of {want}",
                job.key,
                bytes.len(),
            ));
        }
        let mut at = 0;
        for piece in job.out.iter_mut() {
            piece.copy_from_slice(&bytes[at..at + piece.len()]);
            at += piece.len();
        }
        return Ok(());
    }

    let shape = ctx.decode_shape.as_slice();
    let elements: u64 = shape.iter().map(|s| s.get()).product();
    // CHECKED, because this number sizes the buffer an `unsafe` view is built over below: the
    // view's safety obligation is that the slice holds at least `product(shape) * size` bytes,
    // and a wrapped `needed_bytes` makes the view describe more elements than exist.
    let needed_bytes = elements
        .checked_mul(size as u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| {
            format!("a decode unit of {elements} elements of {size} bytes cannot be addressed")
        })?;
    // GROW only. `clear()` then `resize(needed_bytes, 0)` zero-fills the whole buffer, and
    // `decode_into` below writes every byte of it -- the view is built over
    // `new_with_shape`, the entire chunk -- so the fill is overwritten without ever being
    // read. At an inner chunk of 91,549 f32 that is 366 KiB memset per decode, and a
    // chunk_size 64 preload decodes ~2,800 chunks: about a gigabyte of zeroing per preload,
    // thrown away.
    //
    // The buffer is reused across decodes by one worker, so it settles at the largest chunk
    // it has seen and stops reallocating too. Everything below addresses `&mut scratch[..n]`
    // rather than the whole Vec, so a buffer left long by a bigger chunk cannot widen a
    // bounds check.
    //
    // What this DOES change is the failure mode if `decode_into` ever leaves part of the
    // target unwritten: the gap now carries the PREVIOUS chunk's decoded elements rather
    // than zeros, so it reads as plausible values instead of an obvious block of nothing.
    // The worker's buffer outlives the call, so that previous chunk may belong to an earlier
    // CALL rather than to this one -- the same class of staleness, over a wider provenance. It
    // is still bounded by the same condition: the view below is the whole chunk, so any codec
    // that can leave a gap is already broken, whoever wrote the bytes that show through.
    // The view below is built over `new_with_shape` -- the whole chunk -- so a codec that
    // returns `Ok` without filling it would already be broken; this makes such a bug quieter
    // rather than causing one.
    if scratch.len() < needed_bytes {
        // `try_reserve`, then resize. A plain `resize` that cannot allocate goes through
        // `handle_alloc_error`, which ABORTS -- it kills the interpreter with no Python
        // traceback and no chance to fall back. `needed_bytes` comes from array metadata, which
        // nothing here validates against the machine's memory.
        scratch
            .try_reserve(needed_bytes - scratch.len())
            .map_err(|e| format!("could not hold a decode unit of {needed_bytes} bytes: {e}"))?;
        scratch.resize(needed_bytes, 0);
    }
    // Debug builds POISON the reused region. Grow-only scratch means a codec that returns
    // `Ok` without filling its target leaves the PREVIOUS chunk's elements showing through --
    // and because the buffer belongs to a rayon worker, that chunk may be from an earlier
    // call, on a different array. Zeros would be an obvious wrong answer; real values from
    // somewhere else are a plausible one. This makes a test able to see the difference,
    // without costing a release build anything.
    #[cfg(debug_assertions)]
    scratch[..needed_bytes].fill(0xAA);
    let scratch = &mut scratch[..needed_bytes];

    let shape_u64: Vec<u64> = shape.iter().map(|s| s.get()).collect();
    {
        let slice = UnsafeCellSlice::new(&mut scratch[..]);
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
                // BORROWED. `ArrayBytesRaw` is `Cow<'_, [u8]>` and `Bytes` derefs to
                // `[u8]`, so the decode can read the fetched buffer where it lies.
                // `Cow::Owned(bytes.into())` converted it to a `Vec` first -- an allocation
                // and, whenever the `Bytes` does not uniquely own its buffer, a copy of the
                // whole compressed chunk. At ~90 KiB compressed and ~2,800 chunks that is a
                // quarter of a gigabyte per preload, to hand the decoder bytes it already
                // had.
                Cow::Borrowed(&bytes),
                shape,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                &ctx.codec_options,
            )
            .map_err(|e| e.to_string())?;
    }
    // One piece is the overwhelming case: `gather` writes into a single slice, one copy per
    // coordinate, with its own bounds checks. Several pieces only happen when a shard divides
    // a trailing axis, and `gather_pieces` merges consecutive coordinates because there a run
    // can straddle two pieces.
    let result = if let [piece] = &mut job.out[..] {
        match job.grid {
            Some((starts, run)) => {
                gather_runs(&scratch[..], job.element_offsets, starts, run, piece, size)
            }
            None => gather(&scratch[..], job.element_offsets, job.run_len, piece, size),
        }
    } else if job.grid.is_some() {
        // A grid takes the same sub-box out of every index, so its output is one range by
        // construction. Reaching here means an item was built with both, which nothing does.
        Err("a grid selection cannot also span several output pieces".to_string())
    } else {
        gather_pieces(
            &scratch[..],
            job.element_offsets,
            job.run_len,
            &mut job.out,
            size,
        )
    };
    result.map_err(|e| format!("{}: {e}", job.key))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An item's output must be ONE RUN per axis-0 index, and `output_pieces` is where that
    /// is enforced.
    ///
    /// This is the guard that was missing when a banded output first landed: 633 tests failed
    /// with "claims output bytes X..Y, which run backwards" -- `DisjointBytes` reporting the
    /// symptom several layers from the cause. A rank-3 sub-box of `(2,5,5)` in a `(6,10,10)`
    /// output is five runs of five at stride ten, not one run of twenty-five, and modelling it
    /// as one claims the next item's bytes.
    #[test]
    fn output_pieces_refuses_a_strided_sub_box() {
        let item = |subset: &[std::ops::Range<u64>], array: &[u64]| ChunkItem {
            key: StoreKey::new("c/0".to_string()).expect("a key"),
            chunk_subset: ArraySubset::new_with_ranges(subset),
            subset: ArraySubset::new_with_ranges(subset),
            shape: to_nonzero(array),
            num_elements: array.iter().product(),
            array_shape: to_nonzero(array),
            element_offsets: None,
            run_len: 1,
            grid: None,
        };
        // Strided: axis 1 takes 5 of 10 and axis 2 takes 5 of 10, so a row is not one run.
        let strided = item(&[0..2, 0..5, 0..5], &[6, 10, 10]);
        assert!(
            output_pieces(&strided, 8).is_err(),
            "a strided output sub-box must be refused, not modelled as one run"
        );
        // Taking ALL of axis 1 and part of axis 2 is also strided -- ten runs of five, not
        // one run of fifty. Written out because it is the case I got wrong first: "only the
        // last axis is partial" is not the rule; "every axis before the last partial one
        // takes a single element" is.
        let wide_then_partial = item(&[0..2, 0..10, 0..5], &[6, 10, 10]);
        assert!(
            output_pieces(&wide_then_partial, 8).is_err(),
            "a full axis above a partial one is still strided"
        );
        // One element on axis 1 and part of axis 2 IS one run per index, and is served.
        let one_run = item(&[0..2, 3..4, 0..5], &[6, 10, 10]);
        assert!(
            output_pieces(&one_run, 8).is_ok(),
            "a single element above a partial axis is one contiguous run"
        );
        // Whole trailing axes, the ordinary case, stay on the single-range path.
        let whole = item(&[0..2, 0..10, 0..10], &[6, 10, 10]);
        assert_eq!(
            output_pieces(&whole, 8).expect("whole").len(),
            1,
            "whole trailing axes are one contiguous range, not one per row"
        );
    }

    fn to_nonzero(dims: &[u64]) -> Vec<NonZeroU64> {
        dims.iter()
            .map(|d| NonZeroU64::new(*d).expect("non-zero"))
            .collect()
    }

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

    /// The pools are built at the size asked for, and the size is ONE-SHOT.
    ///
    /// Both halves matter. The first is the ordinary contract. The second is the cost of
    /// persistent pools and the thing most likely to mislead someone later: a size set
    /// after the first read is silently ignored, so `pool_sizes` has to report what was built
    /// rather than what was asked for.
    ///
    /// One test rather than several because the pools are process-wide and separate `#[test]`
    /// functions run concurrently in one process, so they would race.
    #[test]
    fn a_pool_is_built_once_at_the_size_asked_for() {
        assert_eq!(
            pool_sizes(),
            (None, None),
            "no read has run, so neither pool should exist yet"
        );

        let (read, decode) = pools(3, 2).expect("a pool of three threads");
        assert_eq!(read.current_num_threads(), 3);
        assert_eq!(decode.current_num_threads(), 2);
        assert_eq!(pool_sizes(), (Some(3), Some(2)));

        // Asking again with different sizes returns the pair already built. This is the
        // documented one-shot behaviour, not an accident, and it is why the size is reported.
        let (read, decode) = pools(64, 64).expect("already built");
        assert_eq!(read.current_num_threads(), 3);
        assert_eq!(decode.current_num_threads(), 2);
        assert_eq!(pool_sizes(), (Some(3), Some(2)));

        // BOTH OR NEITHER. An earlier version built each pool on its own first use, and a
        // test pinned that they stayed independent. They never were in practice -- the one
        // caller asks for both in the same line -- and keying them on a pid means one record
        // with one answer for "was this built in THIS process".
        assert!(
            pool_sizes().0.is_some() == pool_sizes().1.is_some(),
            "the pair is built together, so neither half can exist alone"
        );
    }
}
