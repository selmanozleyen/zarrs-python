//! Reading and decoding the innermost chunks of one call, concurrently.
//!
//! One job per innermost chunk: a reader does the blocking byte-range read, a decode worker
//! decodes the chunk and copies out the elements the selection wants. The two are separate
//! because a read waits on storage and a decode occupies a core, so the useful number of
//! each is different -- hence a separate ceiling for each.
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
use std::sync::{Arc, Mutex, OnceLock};

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

use crate::utils::{
    PyCodecErrExt as _, PyErrExt as _, gather, gather_pieces, gather_runs, key_partial_decoder,
};

/// The per-array state a decode needs, shared by every job of a call.
struct JobContext {
    /// See `CodecPipelineImpl::inner_chunk_is_raw`. When true a row's bytes are addressable
    /// inside its chunk, so a job reads the ROW rather than the chunk holding it.
    raw: bool,
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
        ceilings: Ceilings,
        codec_options: &CodecOptions,
    ) -> PyResult<Vec<&'a ChunkItem>> {
        let element_size = self.element_size()?;
        let ctx = JobContext {
            raw: self.inner_chunk_is_raw,
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

        // TWO POOLS, TWO QUEUES, NO DIVISION.
        //
        // Capacity is not partitioned between calls any more. Both pools are persistent and
        // work-stealing: a free worker takes the next task, whoever queued it. A 585-chunk
        // call and a 9-chunk call contend on the same deque and the big one simply gets more
        // workers, which is what four attempts at a per-call share never managed.
        //
        // What that deletes, and why each had to go:
        //
        //   `share = ceiling / ACTIVE_CALLS` -- a snapshot taken once, at the worst instant
        //     (every loader worker has just entered), never recomputed. Integer division also
        //     lost capacity outright: at C=90 with 32 calls the share is 2, all 32 hold 64,
        //     and 26 permits were unreachable by construction.
        //   the widening loop -- which existed ONLY to correct that staleness, and whose
        //     correction was worth 25% on a long call. A mechanism whose correction term is
        //     worth 25% has a wrong initial allocation.
        //   `Permit::insist` -- a floor over the ceiling, so the real bound was C + N rather
        //     than C, and the knob did not mean what it said.
        //
        // READS AND DECODES GET SEPARATE POOLS ON PURPOSE. A read blocks on storage; a decode
        // occupies a core. A reader parked on Lustre must occupy a READER, never a decode
        // worker, or one slow shard starves every decode in the process.
        //
        // The scopes nest so a reader can hand its chunks straight to the decode pool, and
        // `in_place_scope` runs the CALLING thread as a worker rather than leaving it asleep --
        // which is the one thing daf9ca4 got right, kept here without being a special case.
        // Both scopes block until their tasks finish, which is what keeps `&'a mut [u8]` into
        // the caller's numpy buffer valid without a raw pointer or a completion latch.
        let failure: Mutex<Option<String>> = Mutex::new(None);
        let groups = batch_by_key(jobs);

        decode_pool(ceilings.decode_ceiling).in_place_scope(|dec| {
            read_pool(ceilings.read_ceiling).in_place_scope(|rd| {
                for group in groups {
                    let (failure, ctx) = (&failure, &ctx);
                    rd.spawn(move |_| read_group(group, dec, failure, ctx));
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
                // Every axis, not just the split. The path is the cache key for a
                // subshard's decoder, and two positions differing only on a trailing axis
                // would otherwise collide on it -- returning the wrong subshard's index.
                path.extend_from_slice(&grid_index);
            }
        }
        // The item must lie inside the ONE inner chunk just located. `offset` is now its
        // position within that chunk, and `shard_shape` the chunk's own extent.
        //
        // This replaces `trailing_axes_are_whole`, which the axis-0 descent needed and which
        // this one does not -- but something has to hold, and it is not weaker. Without it an
        // item claiming rows 0..8 x cols 0..12 of a shard whose inner chunk is 8x6 locates
        // chunk (0,0), and its coordinates -- built for a 12-wide row -- address exactly the
        // 48 elements that chunk holds. In bounds, wrong data, no error. `push_entry` takes
        // arbitrary arguments from Python, so this is a trust boundary rather than an
        // invariant the caller can be assumed to keep.
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
            if item.coords.is_none() {
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

/// Split the output into the disjoint piece each located chunk writes, in offset order.
///
/// Each piece comes from `DisjointBytes::take`, whose cursor only moves forward, so a second
/// claim on the same bytes is refused rather than aliased.
///
/// Returns the jobs to read, and the pieces of chunks that were never written, which need
/// only the fill value.
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
    // An item's output is vended as ONE RUN PER axis-0 index, which holds only while the
    // sub-box is contiguous within a row: every axis before the last partial one must take a
    // single element. Anything else is a strided box, and modelling it as one run claims
    // bytes belonging to the next item -- which `DisjointBytes` reports as a backwards claim,
    // naming the symptom rather than this.
    //
    // Checked HERE because this is the funnel: `push_entry`, `push_span`, `push_grid`,
    // `push_points` and a hand-built `ChunkItem` all reach bytes through it. One guard in the
    // shared function beats one per constructor.
    if let Some(last) = shape[1..]
        .iter()
        .zip(&full[1..])
        .rposition(|(width, extent)| width != extent)
    {
        if shape[1..last + 1].iter().any(|width| *width != 1) {
            return Err(PyRuntimeError::new_err(format!(
                "{}: output {:?} of {:?} is strided within one index, and an item's output is \
                 vended as one run per index",
                item.key,
                &shape[1..],
                &full[1..]
            )));
        }
    }
    let row_stride: u64 = full[1..].iter().product();
    let run: u64 = shape[1..].iter().product();
    // Where the sub-box begins inside one row. Row-major, so the trailing starts fold in by
    // the strides below them.
    let mut elem_offset = 0u64;
    let mut stride = 1u64;
    for axis in (1..full.len()).rev() {
        elem_offset += start[axis] * stride;
        stride *= full[axis];
    }
    let rows = shape[0];
    let flat = |row: u64, at: u64| -> PyResult<usize> {
        row.checked_mul(row_stride)
            .and_then(|v| v.checked_add(at))
            .and_then(|v| usize::try_from(v).ok())
            .and_then(|v| v.checked_mul(element_size))
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("{}: output offset too large to address", item.key))
            })
    };
    let width = usize::try_from(run)
        .ok()
        .and_then(|r| r.checked_mul(element_size))
        .ok_or_else(|| {
            PyRuntimeError::new_err(format!("{}: output run too large to address", item.key))
        })?;
    // Whole trailing axes: the rows are adjacent, so they are ONE range. Kept as a special
    // case because it is the common one and because `gather` then stays on its single-slice
    // path with its own coalescing.
    if elem_offset == 0 && run == row_stride {
        let len = usize::try_from(rows)
            .ok()
            .and_then(|r| r.checked_mul(width))
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("{}: output too large to address", item.key))
            })?;
        return Ok(vec![(flat(start[0], 0)?, len)]);
    }
    (0..rows)
        .map(|k| Ok((flat(start[0] + k, elem_offset)?, width)))
        .collect()
}

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
        let coords = coords_of(item)?;
        // WHERE a piece starts comes from `subset`, and HOW LONG it is comes from `coords`.
        // Nothing ties the two together: `ChunkItem` is constructible from Python and skips
        // the element-count check when coords are present. If they disagree, a piece is
        // carved at the wrong offset and the read returns the right number of wrong elements.
        if (coords.len() as u64).checked_mul(item.run_len) != Some(item.subset.num_elements()) {
            return Err(PyRuntimeError::new_err(format!(
                "{} wants {} coordinates of {} elements but its output subset holds {}",
                item.key,
                coords.len(),
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
            // One output piece and no grid: the item is a plain run of rows, which is every
            // rank-1 read and every read whose trailing axes are whole. A banded item has one
            // piece per row and a grid item carries its own per-element offsets; neither is a
            // single contiguous claim, so both take the ordinary path rather than get a second
            // implementation here.
            Some(range)
                if ctx.raw
                    && pieces.len() == 1
                    && item.grid.is_none()
                    && raw_runs(coords_of(item)?, item.run_len) <= raw_max_reads_per_chunk() =>
            {
                let piece = pieces.into_iter().next().expect("length checked");
                raw_row_jobs(item, *range, piece, coords_of(item)?, element_size, ctx, &mut jobs)?;
            }
            Some(range) => jobs.push(Job {
                key: item.key.clone(),
                range: *range,
                raw: {
                    CHUNK_JOBS.fetch_add(1, Ordering::Relaxed);
                    false
                },
                out: pieces,
                coords: coords_of(item)?,
                run_len: item.run_len,
                grid: item.grid.as_ref().map(|(starts, run)| (&starts[..], *run)),
                ctx,
            }),
            None => absent.extend(pieces),
        }
    }
    Ok((jobs, absent))
}

/// Jobs that took the RAW path, and jobs that read a whole chunk, since the run began.
///
/// The project rule -- a knob that was set is not a knob that arrived -- applied to a code
/// path. A gate that silently refuses everything is indistinguishable from a gate that is
/// working: values stay correct either way and only the throughput differs, which reads as
/// "the raw path did not pay" rather than "the raw path was never taken". Both failures have
/// already happened here once.
pub(crate) static RAW_JOBS: AtomicU64 = AtomicU64::new(0);
pub(crate) static CHUNK_JOBS: AtomicU64 = AtomicU64::new(0);

/// How many READS this chunk's rows become once consecutive ones are merged.
///
/// The count that matters is runs, not rows: 64 consecutive rows are ONE read, and 64
/// scattered ones are 64. `coords` is non-decreasing, so a run is a maximal stretch stepping
/// by exactly `run_len`. A duplicate steps by 0 and breaks the run, which is right -- the same
/// row twice is two output pieces and cannot be one read.
pub(crate) fn raw_runs(coords: &[u64], run_len: u64) -> usize {
    if coords.is_empty() {
        return 0;
    }
    1 + coords
        .windows(2)
        .filter(|w| w[1] != w[0] + run_len)
        .count()
}

/// Reads a chunk may become before the raw path is declined for it.
///
/// The raw path trades BYTES for REQUESTS, and requests are the scarce resource. Measured at
/// 14 plates on uncompressed CSR, cs=1, with no gate at all: amplification fell 87.3x -> 2.6x
/// (20.82 GB -> 0.61 GB, matching zarr-python exactly) and throughput fell with it, 10,540
/// rows/s -> 1,360, at 2.4x the CPU. A 5.8 KB row costs nearly what a 366 KB chunk costs to
/// fetch, so 63 tiny reads lose to one large one however few bytes they move.
///
/// So the choice is PER ITEM, not per array: take the raw path only where a chunk's rows
/// collapse to a handful of reads.
///
/// TWO, and it is measured rather than picked. Sweeping stride against 64-row chunks walks the
/// gate's operating region directly, which is the thing an earlier sweep could not do -- it
/// sampled patterns at ~1, 1 and ~32 runs a chunk and concluded from points on either side of
/// the value being judged:
///
///     runs/chunk   32     16      8      4      2      1
///     ratio      0.97x  0.97x  0.95x  0.98x  1.03x  1.31x
///
/// Raw pays at <= 2 and is flat to slightly negative from 4 up. A gate of 8 therefore ADMITS
/// the 0.95x and 0.98x cases; 2 refuses them and keeps only the wins. Eight requests for a
/// fraction of the bytes is not better than one request for all of them when requests are the
/// scarce resource, which is the same wall `get_partial_many` runs into from the other side.
///
/// `ZARRS_RAW_MAX_READS_PER_CHUNK=0` disables the raw path entirely; the sweep above is in
/// `notes/read-unit-and-hints.md`.
fn raw_max_reads_per_chunk() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("ZARRS_RAW_MAX_READS_PER_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    })
}

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
#[allow(clippy::too_many_arguments)]
fn raw_row_jobs<'a>(
    item: &'a ChunkItem,
    range: ByteRange,
    piece: &'a mut [u8],
    coords: &'a [u64],
    element_size: usize,
    ctx: &'a JobContext,
    jobs: &mut Vec<Job<'a>>,
) -> PyResult<()> {
    let ByteRange::FromStart(base, _) = range else {
        return Err(PyRuntimeError::new_err(format!(
            "{}: the raw path needs a FromStart range, got {range:?}",
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
    let mut start = 0usize;
    while start < coords.len() {
        let mut end = start + 1;
        while end < coords.len() && coords[end] == coords[end - 1] + item.run_len {
            end += 1;
        }
        let span = row_bytes
            .checked_mul(end - start)
            .ok_or_else(|| PyRuntimeError::new_err(format!("{}: run too large", item.key)))?;
        let (run_out, tail) = rest.split_at_mut(span.min(rest.len()));
        rest = tail;
        let at = base
            .checked_add(coords[start] * element_size as u64)
            .ok_or_else(|| PyRuntimeError::new_err(format!("{}: offset overflow", item.key)))?;
        jobs.push(Job {
            key: item.key.clone(),
            range: ByteRange::FromStart(at, Some(span as u64)),
            raw: true,
            out: vec![run_out],
            coords: &[],
            run_len: item.run_len,
            grid: None,
            ctx,
        });
        start = end;
        RAW_JOBS.fetch_add(1, Ordering::Relaxed);
    }
    if !rest.is_empty() {
        return Err(PyRuntimeError::new_err(format!(
            "{}: {} output bytes left after {} rows",
            item.key,
            rest.len(),
            coords.len()
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


/// What the shard index cache did. Counted because nothing else can tell.
///
/// A cache that is never consulted passes every correctness test ever written -- repeated
/// reads still agree, writes still invalidate. And on a small array it is invisible in the
/// timings too: the index is already in page cache, so a hit saves a memcpy rather than the
/// full-latency round trip the cache exists to avoid. Measured that way it reported 1.03x,
/// which says the regime was wrong and nothing about whether the cache works.
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
fn default_ceiling() -> usize {
    std::thread::available_parallelism().map_or(8, std::num::NonZeroUsize::get)
}

/// The two pools, built once and shared by every call in the process.
///
/// PERSISTENT, because the alternative was measured and lost. Threads used to be spawned per
/// call and their capacity divided between calls by `ceiling / ACTIVE_CALLS` -- a snapshot
/// taken once, at the instant every loader worker had just entered, and never recomputed. A
/// call was then stuck with that share for its whole life while other calls finished and left
/// permits unclaimed, and integer division lost capacity outright (at C=90 with 32 calls the
/// share is 2, all 32 hold 64, and 26 are unreachable). Four attempts to correct that
/// staleness in flight -- a polling loop, twice re-predicated, then deleted -- shipped a
/// 200 us poll, a wrong unit, a hang, and a 25% regression between them.
///
/// A work-stealing pool has none of those failure modes because it never divides anything: a
/// free worker takes the next task, whoever queued it, so a call with more work simply gets
/// more workers.
///
/// SIZED ONCE. This is the honest cost of the change: `read_worker_ceiling` and
/// `decode_worker_ceiling` are read from `zarr.config` per call, but only the FIRST call's
/// values build the pools. A later `with zarr.config.set(...)` around a read no longer resizes
/// anything. `pool_sizes()` reports what was actually built, so a bench can assert it rather
/// than assume -- this repo's rule that a knob which was set is not a knob that arrived.
static READ_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
static DECODE_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn build_pool(size: usize, name: &'static str) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(size)
        .thread_name(move |i| format!("zarrs-{name}-{i}"))
        .build()
        .expect("a thread pool of a positive size")
}

/// Threads that BLOCK on storage. Sized independently of the core count for that reason.
fn read_pool(size: usize) -> &'static rayon::ThreadPool {
    READ_POOL.get_or_init(|| build_pool(size, "read"))
}

/// Threads that occupy a core. This is the one genuinely bounded by parallelism.
fn decode_pool(size: usize) -> &'static rayon::ThreadPool {
    DECODE_POOL.get_or_init(|| build_pool(size, "decode"))
}

/// What the pools were actually BUILT with, or `None` where one has not been built yet.
///
/// Sizes are fixed at first use, so a caller that sets a ceiling after the first read gets the
/// old value silently. Reporting the built size is what lets a benchmark tell "the knob did
/// not pay" from "the knob never arrived".
pub(crate) fn pool_sizes() -> (Option<usize>, Option<usize>) {
    (
        READ_POOL.get().map(rayon::ThreadPool::current_num_threads),
        DECODE_POOL.get().map(rayon::ThreadPool::current_num_threads),
    )
}

/// The pool sizes ONE call asks for, read from `zarr.config` when that call starts.
///
/// Only the first call's values are used -- see [`READ_POOL`].
#[derive(Clone, Copy)]
pub(crate) struct Ceilings {
    pub(crate) read_ceiling: usize,
    pub(crate) decode_ceiling: usize,
}

impl Ceilings {
    /// `None`, or a zero, takes [`default_ceiling`].
    pub(crate) fn new(read_ceiling: Option<usize>, decode_ceiling: Option<usize>) -> Self {
        Self {
            read_ceiling: read_ceiling
                .filter(|c| *c > 0)
                .unwrap_or_else(default_ceiling),
            decode_ceiling: decode_ceiling
                .filter(|c| *c > 0)
                .unwrap_or_else(default_ceiling),
        }
    }
}

/// One innermost chunk, and the slice of the output its elements belong in.
struct Job<'a> {
    key: StoreKey,
    /// The chunk's byte range, or -- on the raw path -- one ROW's range inside it.
    range: ByteRange,
    /// Raw jobs carry the wanted bytes exactly: no decode, no scratch, no gather. Their
    /// `range` is the ROW's bytes inside the chunk rather than the whole chunk's.
    raw: bool,
    /// The output ranges this chunk fills, ascending. ONE range while every axis after the
    /// first is taken whole -- which is every rank-1 read, so the CSR path always has one.
    /// A shard that divides a trailing axis gives an item one range per row instead.
    out: Vec<&'a mut [u8]>,
    coords: &'a [u64],
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

/// Consecutive jobs sharing a store key, read in one call.
///
/// `FilesystemStore::get_partial_many` opens the file, takes its read lock and queries its
/// size ONCE for the whole batch, then reads each range at its own offset. Per job that is
/// three syscalls saved; whether that is worth anything depends entirely on whether the read
/// was request-bound, which is a question for the measurement and not for this comment.
struct ReadGroup<'a> {
    key: StoreKey,
    jobs: Vec<Job<'a>>,
}

/// Cap on the bytes one group may hold in flight.
///
/// A group's results are held until its LAST job has decoded, so an ungrouped batch would let
/// one key's whole shard sit resident while the first of its jobs decodes. 64 MiB is far above
/// any single inner chunk here and far below the memory a preload already holds.
const GROUP_MAX_BYTES: u64 = 64 << 20;

/// Cap on the JOBS one group may hold, which is the one that binds.
///
/// A group is read by ONE worker, so a large group serialises what the workers would otherwise
/// have done in parallel -- and it does not release any of its bytes until its last job has
/// decoded. Measured at 14 plates: a scattered draw groups 2-3 chunks per call and gains from
/// batching (sparse_c cs=4 went 1.12x -> 1.30x), while a strided draw groups 172-382 and loses
/// heavily (sparse_c stride 32 went 2.59x -> 1.80x, and sparse_r stride 128 fell to 0.91x,
/// below zarr-python).
///
/// So the byte cap never bound on the case that needed it: 64 MiB is ~180 inner chunks here.
/// Eight keeps every group a scattered draw actually forms while cutting a strided one to a
/// size a single worker can still finish promptly.
const GROUP_MAX_JOBS: usize = 8;

/// Jobs batched, and the groups they went into, since the run began. Without these a null
/// result is ambiguous: "batching bought nothing" and "batching never engaged" produce the
/// same number, and at a scattered draw over many shards the second is the likely one.
pub(crate) static GROUP_JOBS: AtomicU64 = AtomicU64::new(0);
pub(crate) static GROUP_COUNT: AtomicU64 = AtomicU64::new(0);

/// Decode scratch outlives the CALL, because the threads that hold it do not.
///
/// `decode_loop` grows its buffer to the largest chunk that worker has seen and then reuses it
/// -- which only pays if the worker decodes more than one chunk. Workers are call-scoped, and
/// at a chunk_size of 64 a call has ~9 chunks and spawns ~9 decoders, so nearly every worker
/// paid a fresh allocation and a full `resize(needed, 0)` memset of an inner chunk (366 KiB
/// sparse, 512 KiB dense) in order to decode exactly ONE chunk. Above glibc's 128 KiB mmap
/// threshold that is an mmap, a memset, a page-fault per page, and a munmap at thread exit.
///
/// The buffers are interchangeable, so they are kept rather than freed. At chunk_size 1 a
/// worker decodes ~65 chunks and already amortised this; the pool is aimed squarely at the
/// small-batch end, and if it moves the large-batch end too then the mechanism is not what
/// this comment claims.
static SCRATCH_POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Enough for every worker one call may hold at a generous ceiling; past that, dropping a
/// buffer is cheaper than keeping a pool nobody drains.
const SCRATCH_POOL_MAX: usize = 128;

/// Buffers served from the pool, and buffers that had to be created. Without these a null
/// result cannot be told from "the pool never ran" -- the failure this project has already
/// been bitten by twice.
pub(crate) static SCRATCH_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static SCRATCH_MISSES: AtomicU64 = AtomicU64::new(0);

/// A buffer to decode into, reused if one is free. Its contents are meaningless -- every
/// caller `resize`s to what it needs before writing, and addresses only `..needed`.
fn scratch_take() -> Vec<u8> {
    match SCRATCH_POOL.lock().ok().and_then(|mut pool| pool.pop()) {
        Some(buf) => {
            SCRATCH_HITS.fetch_add(1, Ordering::Relaxed);
            buf
        }
        None => {
            SCRATCH_MISSES.fetch_add(1, Ordering::Relaxed);
            Vec::new()
        }
    }
}

/// Hands a buffer back. An empty one is not worth a lock -- it carries no allocation.
fn scratch_give(buf: Vec<u8>) {
    if buf.capacity() == 0 {
        return;
    }
    if let Ok(mut pool) = SCRATCH_POOL.lock() {
        if pool.len() < SCRATCH_POOL_MAX {
            pool.push(buf);
        }
    }
}

/// Consecutive only, and deliberately not sorted. Jobs arrive in OUTPUT order; reordering them
/// to make bigger groups would let a batch hold bytes for a job whose output lands far away,
/// keeping the whole group alive until that one is reached.
fn batch_by_key(jobs: Vec<Job<'_>>) -> Vec<ReadGroup<'_>> {
    let mut groups: Vec<ReadGroup<'_>> = Vec::new();
    let mut held: u64 = 0;
    for job in jobs {
        let len = match job.range {
            ByteRange::FromStart(_, Some(n)) | ByteRange::Suffix(n) => n,
            // An open-ended range's size is not known until the read returns, so it cannot be
            // budgeted -- give it a group of its own rather than guess.
            _ => u64::MAX,
        };
        let fits = groups.last().is_some_and(|g| {
            g.key == job.key
                && g.jobs.len() < GROUP_MAX_JOBS
                && held.saturating_add(len) <= GROUP_MAX_BYTES
        });
        if fits {
            held = held.saturating_add(len);
            groups.last_mut().expect("checked above").jobs.push(job);
        } else {
            held = len;
            groups.push(ReadGroup { key: job.key.clone(), jobs: vec![job] });
        }
    }
    GROUP_JOBS.fetch_add(
        groups.iter().map(|g| g.jobs.len() as u64).sum::<u64>(),
        Ordering::Relaxed,
    );
    GROUP_COUNT.fetch_add(groups.len() as u64, Ordering::Relaxed);
    groups
}

/// One group: one store call, then one decode task per chunk it returned.
///
/// The decode tasks go to the DECODE pool, not this one. A reader is parked on storage for
/// most of its life and a decode wants a core, so mixing them means one slow shard holds a
/// core that a ready chunk needs.
///
/// Each chunk is spawned as it arrives rather than after the whole group, so the first chunk
/// decodes while the rest of the group is still being handed over.
fn read_group<'scope, 'env>(
    group: ReadGroup<'env>,
    dec: &rayon::Scope<'scope>,
    failure: &'env Mutex<Option<String>>,
    ctx: &'env JobContext,
) where
    'env: 'scope,
{
    let ranges: Vec<ByteRange> = group.jobs.iter().map(|job| job.range).collect();
    match ctx.store.get_partial_many(&group.key, Box::new(ranges.into_iter())) {
        // `None` for the whole iterator means the KEY is absent, which is a different thing
        // from a range coming back empty. `decode_one` already knows what an absent chunk
        // contributes -- the fill value, or an error where a shard index named it -- so every
        // job in the group gets `None` and that logic stays in one place.
        Ok(None) => {
            for job in group.jobs {
                spawn_decode(dec, job, None, failure);
            }
        }
        Ok(Some(mut fetched)) => {
            for job in group.jobs {
                // One result per range, in order, or the pairing is wrong for every job in the
                // group -- and wrong bytes under the right key is not something a later check
                // would catch.
                let Some(piece) = fetched.next() else {
                    record(
                        failure,
                        format!("{}: the store returned fewer ranges than asked", job.key),
                    );
                    return;
                };
                // `MaybeBytes` is `Option<Bytes>` and the iterator yields a bare `Bytes`: a
                // range that came back IS present, and absence is the `Ok(None)` arm above,
                // which is about the key rather than the range.
                let bytes = match piece {
                    Ok(b) => Some(b),
                    Err(e) => {
                        record(failure, format!("read {} failed: {e}", job.key));
                        return;
                    }
                };
                spawn_decode(dec, job, bytes, failure);
            }
        }
        Err(e) => record(failure, format!("read {} failed: {e}", group.key)),
    }
}

/// One chunk's decode, on the decode pool.
///
/// The scratch buffer is taken from the pool and returned inside the task, so a decode worker
/// that handles chunks from many different calls reuses one buffer across all of them -- which
/// is what `SCRATCH_POOL` was for when workers were per-call and short-lived.
fn spawn_decode<'scope, 'env>(
    dec: &rayon::Scope<'scope>,
    mut job: Job<'env>,
    bytes: MaybeBytes,
    failure: &'env Mutex<Option<String>>,
) where
    'env: 'scope,
{
    dec.spawn(move |_| {
        let mut scratch = scratch_take();
        if let Err(e) = decode_one(&mut job, bytes, &mut scratch) {
            record(failure, e);
        }
        scratch_give(scratch);
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

    // A raw job's read WAS the answer: its range is the row, not the chunk. No decode, no
    // scratch, no gather -- but not copy-free either, since `get_partial` hands back an owned
    // buffer and these bytes still have to be moved into the output.
    //
    // `out` is a Vec since the band split, so a raw job's bytes are laid across its pieces in
    // order. `raw_row_jobs` only ever builds ONE piece per job -- a raw job is one run of one
    // row -- but walking the pieces costs nothing and means this cannot silently write only
    // the first if that ever stops being true.
    if job.raw {
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
    let needed = usize::try_from(elements).map_err(|e| e.to_string())? * size;
    // GROW only. `clear()` then `resize(needed, 0)` zero-fills the whole buffer, and
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
    // With `SCRATCH_POOL` that previous chunk may belong to an earlier CALL rather than to
    // this one -- the same class of staleness, over a wider provenance. It is still bounded
    // by the same condition: the view below is the whole chunk, so any codec that can leave
    // a gap is already broken, whoever wrote the bytes that show through.
    // The view below is built over `new_with_shape` -- the whole chunk -- so a codec that
    // returns `Ok` without filling it would already be broken; this makes such a bug quieter
    // rather than causing one.
    if scratch.len() < needed {
        scratch.resize(needed, 0);
    }
    let scratch = &mut scratch[..needed];

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
    // One piece is the overwhelming case and keeps `gather` exactly as it was, coalescing and
    // bounds checks included. Several pieces only happen when a shard divides a trailing axis.
    let result = if let [piece] = &mut job.out[..] {
        match job.grid {
            Some((starts, run)) => {
                gather_runs(&scratch[..], job.coords, starts, run, piece, size)
            }
            None => gather(&scratch[..], job.coords, job.run_len, piece, size),
        }
    } else if job.grid.is_some() {
        // A grid takes the same sub-box out of every index, so its output is one range by
        // construction. Reaching here means an item was built with both, which nothing does.
        Err("a grid selection cannot also span several output pieces".to_string())
    } else {
        gather_pieces(&scratch[..], job.coords, job.run_len, &mut job.out, size)
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
            coords: None,
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
        dims.iter().map(|d| NonZeroU64::new(*d).expect("non-zero")).collect()
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
    /// persistent pools and the thing most likely to mislead someone later: a ceiling set
    /// after the first read is silently ignored, so `pool_sizes` has to report what was built
    /// rather than what was asked for.
    ///
    /// One test rather than several because the pools are process-wide `OnceLock`s and
    /// separate `#[test]` functions run concurrently in one process, so they would race.
    #[test]
    fn a_pool_is_built_once_at_the_size_asked_for() {
        assert_eq!(
            pool_sizes(),
            (None, None),
            "no read has run, so neither pool should exist yet"
        );

        let pool = read_pool(3);
        assert_eq!(pool.current_num_threads(), 3);
        assert_eq!(pool_sizes().0, Some(3));

        // Asking again with a different size returns the pool already built. This is the
        // documented one-shot behaviour, not an accident, and it is why the size is reported.
        assert_eq!(read_pool(64).current_num_threads(), 3);
        assert_eq!(pool_sizes().0, Some(3));

        // THE TWO POOLS ARE SEPARATE. The read pool is built here; the decode pool must not
        // have been dragged into existence with it, or a read-side default would silently
        // become the decode width.
        assert_eq!(
            pool_sizes().1,
            None,
            "building the read pool must not build the decode pool"
        );
        assert_eq!(decode_pool(2).current_num_threads(), 2);
        assert_eq!(pool_sizes(), (Some(3), Some(2)));
    }
}
