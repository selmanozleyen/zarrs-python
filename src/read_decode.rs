//! Reading and decoding the innermost chunks of one call, concurrently.
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyAnyMethods;
use pyo3::{PyResult, Python};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{
    ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset, ArrayToBytesCodecTraits,
    CodecOptions, FillValue, ravel_indices,
};
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{MaybeBytes, ReadableStorage, StoreKey};

use crate::CodecPipelineImpl;
use crate::chunk_item::ChunkItem;
use crate::per_process::PerProcess;
use crate::shard_index::ShardInfo;
use zarrs::array::codec::api::ByteIntervalPartialDecoder;
use zarrs::array::codec::array_to_bytes::sharding::ShardingPartialDecoder;

use crate::utils::{
    PyCodecErrExt as _, PyErrExt as _, coord_runs, gather, gather_pieces, gather_runs,
    key_partial_decoder,
};

/// The per-array state a decode needs, shared by every job of a call.
struct JobContext {
    /// See `CodecPipelineImpl::inner_chunk_is_raw`. When true a row's bytes are addressable
    /// inside its chunk, so a job reads the ROW rather than the chunk holding it.
    raw: bool,
    shard: Arc<ShardInfo>,
    store: ReadableStorage,
    codec_options: CodecOptions,
    element_size: usize,
    /// What an absent chunk contributes. Needed in the workers, not just at carve time,
    /// because an unsharded chunk's absence is only discovered by the read.
    fill_value: FillValue,
    /// Whether missing bytes are ordinary. A shard index that named a chunk which is then
    /// missing means the store changed under the read, and that is worth failing on; an
    /// unsharded chunk has no index to consult, so its key simply may not exist yet.
    may_be_absent: bool,
    /// See [`RAW_MAX_READS`]. Per call, so a caller can disable the raw path for one read.
    raw_max_reads: usize,
    /// The unit decoded into scratch: the shard's inner chunk where the array is sharded, the chunk
    /// where it is not. Also per call (an array's chunks are all one shape) so it is resolved once
    /// here from the first item rather than carried on every `Job`.
    decode_shape: Vec<NonZeroU64>,
}

/// Shard and subshard decoders built during one call.
#[derive(Default)]
struct CallDecoders {
    shards: HashMap<StoreKey, Arc<ShardingPartialDecoder>>,
    subshards: HashMap<(StoreKey, Vec<u64>), Arc<ShardingPartialDecoder>>,
}

impl CodecPipelineImpl {
    /// Read and decode `items`, one job per innermost chunk, on workers scoped to this call.
    ///
    /// `items` must be chunk-unit items: one whole innermost chunk each, carrying the
    /// coordinates wanted from it. There is no second path, so an item this cannot take is an
    /// error rather than a hand-off.
    pub(crate) fn retrieve_chunk_units(
        &self,
        shard: &Arc<ShardInfo>,
        items: &[ChunkItem],
        output: UnsafeCellSlice<'_, u8>,
        output_len: usize,
        config: ReadConfig,
        pools: &(Arc<rayon::ThreadPool>, Arc<rayon::ThreadPool>),
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        let element_size = self.element_size()?;
        let ctx = JobContext {
            raw: self.inner_chunk_is_raw,
            raw_max_reads: config.raw_max_reads,
            shard: shard.clone(),
            store: self.readable_store.clone(),
            codec_options: (*codec_options).with_concurrent_target(1),
            element_size,
            fill_value: self.fill_value.clone(),
            may_be_absent: shard.depth() == 0,
            // Sharded: the shard says. Not sharded: any item does, because chunk shapes are uniform
            // across an array, and an empty batch never reaches a decode.
            decode_shape: shard.subchunk_shape.as_ref().map_or_else(
                || items.first().map(|i| i.shape.clone()).unwrap_or_default(),
                |shape| shape.to_vec(),
            ),
        };

        let located = self.locate_chunks(shard, items, &ctx)?;

        let output = DisjointBytes::new(output, output_len);
        let (jobs, absent) = carve(&output, &located, element_size, &ctx)?;
        // Disjointness is proven above; coverage is not. zarr hands us a buffer from
        // `np.empty`, so a byte no job owns is returned as whatever was in that memory.
        //
        if output.covered() != output_len {
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
            return Ok(());
        }

        // Two persistent work-stealing pools, capacity never divided between calls: a free
        // worker takes the next task whoever queued it.
        //
        // Reads and decodes get separate pools. A read blocks on storage, a decode occupies a
        // core, and a reader parked on Lustre must never hold a decode worker or one slow shard
        // starves every decode in the process.
        let failure: Mutex<Option<String>> = Mutex::new(None);

        // A call takes at most `io_workers` of the pool, so two calls asking for different
        // widths both get what they asked for.
        //
        // `iter_concurrent_limit!` on the FETCH side, which is an iteration and so bounds
        // cheaply. The decode side receives work as reads land, so it is bounded by a permit
        // inside each task instead -- see `DecodePermits`.
        let readers = config.io_workers.max(1);
        let permits = DecodePermits::new(config.codec_workers);
        let (read_pool, decode_pool) = pools;
        decode_pool.in_place_scope(|dec| {
            read_pool.install(|| {
                let (failure, ctx, permits) = (&failure, &ctx, &permits);
                iter_concurrent_limit!(readers, jobs, for_each, |job| {
                    read_one(job, dec, failure, ctx, permits);
                });
            });
        });

        if let Some(e) = failure.lock().expect("failure slot poisoned").take() {
            return Err(PyRuntimeError::new_err(e));
        }
        Ok(())
    }

    /// A decoder from `call_cache`, then `array_cache`, or built and inserted into both.
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
    fn locate(
        &self,
        shard: &ShardInfo,
        item: &ChunkItem,
        start: &[u64],
        ctx: &JobContext,
        decoders: &mut CallDecoders,
    ) -> PyResult<Option<ByteRange>> {
        // Not sharded: there is no index to read and nothing to descend, the store value is the
        // chunk. Whether the key exists is the read's business, and a missing one comes back as
        // absent bytes there, exactly as a never-written shard entry does here.
        if shard.depth() == 0 {
            return Ok(Some(ByteRange::FromStart(0, None)));
        }
        let file = key_partial_decoder(&self.readable_store, &item.key);
        let mut shard_shape = item.shape.clone();
        let mut offset: Vec<u64> = start.to_vec();
        // (offset, length) of the level being descended into, absolute in the store value.
        let mut extent: Option<(u64, u64)> = None;
        // The subchunk indices taken so far. Only built below depth 0.
        let mut path: Vec<u64> = Vec::new();

        for depth in 0..shard.depth() {
            let level_shape = shard.subchunk_shape_at(depth);
            // Every axis, because `subchunk_byte_range` takes a full grid index. Filling only
            // axis 0 addresses the right subchunk just when every other axis holds exactly one,
            // which is not true of a shard that divides a trailing axis.
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
                        key_partial_decoder(&self.readable_store, &item.key),
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
                // Every axis, not just the split. The path is the cache key for a subshard's
                // decoder, and two positions differing only on a trailing axis would otherwise
                // collide on it: returning the wrong subshard's index.
                path.extend_from_slice(&grid_index);
            }
        }
        // The item must lie inside the one inner chunk just located: `offset` is its position
        // within that chunk, `shard_shape` the chunk's own extent. Without this an item claiming
        // rows 0..8 x cols 0..12 of a shard whose inner chunk is 8x6 locates chunk (0,0) and
        // addresses exactly the 48 elements it holds: in bounds, wrong data, no error. `push_entry`
        // takes arbitrary arguments from Python, so this is a trust boundary rather than a caller
        // invariant.
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
    #[allow(clippy::type_complexity)]
    fn locate_chunks<'a>(
        &self,
        shard: &ShardInfo,
        items: &'a [ChunkItem],
        ctx: &JobContext,
    ) -> PyResult<Vec<(&'a ChunkItem, Option<ByteRange>)>> {
        let mut located = Vec::with_capacity(items.len());
        let mut decoders = CallDecoders::default();

        for item in items {
            // No second path to decline to, so this is an error where it is found.
            coords_of(item)?;
            // The whole position, not just axis 0: the descent divides on every axis now,
            // so a shard that splits a trailing one is addressed rather than refused.
            let start = item.chunk_subset.start().to_vec();
            located.push((item, self.locate(shard, item, &start, ctx, &mut decoders)?));
        }
        Ok(located)
    }
}

/// Split the output into the disjoint piece each located chunk writes, in offset order. Pieces
/// come from `DisjointBytes::take`, whose cursor only moves forward, so a second claim on the
/// same bytes is refused rather than aliased.
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
    // The arithmetic is zarrs': `contiguous_linearised_indices` walks the subset in C order, merges
    // whole trailing axes into one run, and rechecks that `full` encapsulates the subset, the
    // bounds half of the guard above.
    //
    // It does not refuse a strided sub-box, it emits more runs for one, so the refusal below is a
    // count read off that walk rather than a second copy of the contiguity rule. An item's output
    // is one run per axis-0 index, or a single run when whole trailing axes make the rows adjacent;
    // anything else is strided within a row, and vending it as one run per index would claim the
    // next item's bytes.
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
    let width = usize::try_from(runs.contiguous_elements())
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
                .map(|at| (at, width))
                .ok_or_else(|| {
                    PyRuntimeError::new_err(format!(
                        "{}: output offset too large to address",
                        item.key
                    ))
                })
        })
        .collect()
}

fn carve<'a>(
    output: &'a DisjointBytes<'a>,
    located: &[(&'a ChunkItem, Option<ByteRange>)],
    element_size: usize,
    ctx: &'a JobContext,
) -> PyResult<(Vec<Job<'a>>, Vec<&'a mut [u8]>)> {
    // Pass 1: what each item needs, and the element-count agreement. Nothing is vended yet.
    let mut plan: Vec<(usize, Vec<(usize, usize)>)> = Vec::with_capacity(located.len());
    for (i, (item, _)) in located.iter().enumerate() {
        let coords = coords_of(item)?;
        // where a piece starts comes from `subset`, and how long it is comes from `coords`.
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

    // Pass 2: vend every piece of every item in ascending output order.
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
            // `run_len` its length. The request COUNT is the same either way, so all that
            // changes is how many bytes each one moves.
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
                    // Zero DISABLES, which the threshold alone does not say: an item with no
                    // coordinates is 0 reads, and `0 <= 0` would take the path the knob was
                    // set to refuse. Nothing builds such an item today -- `push_span` returns
                    // early on an empty count -- so this makes the documented behaviour true
                    // by construction rather than by the absence of a caller.
                    && ctx.raw_max_reads > 0
                    && pieces.len() == 1
                    && item.grid.is_none()
                    && raw_runs(coords_of(item)?, item.run_len) <= ctx.raw_max_reads =>
            {
                let piece = pieces.into_iter().next().expect("length checked");
                raw_row_jobs(
                    item,
                    *range,
                    piece,
                    coords_of(item)?,
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
                    raw: false,
                    out: pieces,
                    coords: coords_of(item)?,
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

/// Jobs that took the RAW path, and jobs that read a whole chunk, since the run began. A gate
/// that silently refuses everything is indistinguishable from one that works -- values stay
/// correct either way and only throughput moves -- so the path has to be counted, not inferred.
pub(crate) static RAW_JOBS: AtomicU64 = AtomicU64::new(0);
pub(crate) static CHUNK_JOBS: AtomicU64 = AtomicU64::new(0);

/// How many READS this chunk's rows become once consecutive ones are merged: 64 consecutive
/// rows are ONE read, 64 scattered ones are 64. `raw_row_jobs` emits exactly the runs counted
/// here, from the same walk, so the gate cannot come to disagree with what it admits.
pub(crate) fn raw_runs(coords: &[u64], run_len: u64) -> usize {
    coord_runs(coords, run_len).count()
}

/// Default for `codec_pipeline.raw_max_reads_per_chunk`.
///
/// The raw path reads a row's exact bytes instead of the chunk around it, which trades BYTES
/// for REQUESTS -- and requests are the scarce resource, since a row costs nearly what the
/// chunk holding it costs to fetch. Hence a PER-ITEM gate: take it only where a chunk's wanted
/// rows collapse to a handful of reads. Two is measured; zero disables the path and costs ~75%
/// on an uncompressed scattered draw. See README and `notes/deferred-wins.md`.
const RAW_MAX_READS: usize = 2;

/// One job per RUN of consecutive rows, each reading exactly its own bytes, for a chunk that
/// is a plain byte tiling.
///
/// `coord` is already the row's element offset within the chunk and `run_len` its length, so
/// the row's byte range is arithmetic. The request COUNT is the same either way, and only the
/// bytes each one moves change.
///
/// `piece` is the item's single contiguous claim, split here rather than re-claimed, so the
/// vend-once cursor still sees exactly one take per item and coverage is still checked.
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
    for run in coord_runs(coords, item.run_len) {
        let span = row_bytes
            .checked_mul(run.len())
            .ok_or_else(|| PyRuntimeError::new_err(format!("{}: run too large", item.key)))?;
        let (run_out, tail) = rest.split_at_mut(span.min(rest.len()));
        rest = tail;
        let at = base
            .checked_add(coords[run.start] * element_size as u64)
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

/// Hands out each byte range of the output at most once. The one `unsafe` is inside `take` and
/// its argument is local: `cursor` only moves forward, so no two ranges can intersect.
struct DisjointBytes<'a> {
    slice: UnsafeCellSlice<'a, u8>,
    len: usize,
    /// A `Cell` so `take` can vend from a shared reference. It has to: each piece borrows
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

    /// How many bytes have actually been handed out. not `cursor`: that is the end of the
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
    // The guarantee is `cursor`, not the type. `UnsafeCellSlice::get_mut` carries the same
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

/// Where an item's elements land in the output, as a flat element offset. Orders jobs; never
/// places bytes. The C-order ravel of the subset's start, not row index times row length --
/// a banded item's trailing start is not zero, and two bands of one row would sort equal.
fn output_offset(item: &ChunkItem) -> u64 {
    let shape = bytemuck::must_cast_slice::<_, u64>(&item.array_shape);
    ravel_indices(item.subset.start(), shape).unwrap_or(u64::MAX)
}

/// What the shard index cache did. Counted because nothing else can: a cache that is never
/// consulted passes every correctness test ever written.
pub(crate) static INDEX_CALL_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INDEX_ARRAY_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INDEX_BUILDS: AtomicU64 = AtomicU64::new(0);

/// How much wider the I/O pool is than the CPU pool. Generous because a parked reader is a
/// stack and no CPU. A multiple of that pool rather than of the machine, so the one setting
/// that sizes it -- `RAYON_NUM_THREADS` -- sizes every width in the process.
const IO_POOL_MULTIPLIER: usize = 8;

/// The pool a read's STORE traffic runs on; decoding runs on [`crate::pool`] with encoding,
/// being CPU work. What must not share is the two SIDES -- a reader parked on Lustre must never
/// hold a worker a decode needs -- and a third pool would buy no separation these two do not.
static READ_POOL: PerProcess<rayon::ThreadPool> = PerProcess::new();

fn build_pool(size: usize, name: &'static str) -> PyResult<rayon::ThreadPool> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(size)
        .thread_name(move |i| format!("zarrs-{name}-{i}"))
        .build()
        .map_err(|e| PyRuntimeError::new_err(format!("could not create the {name} pool: {e}")))
}

/// `(io, cpu)`: the pool that fetches, and the pool that decodes. Takes the token because it
/// locks, and the GIL is what keeps that lock unheld when another thread forks.
pub(crate) fn pools(
    py: Python<'_>,
    limits: PoolLimits,
) -> PyResult<(Arc<rayon::ThreadPool>, Arc<rayon::ThreadPool>)> {
    let codec = crate::pool::pool(py, Some(limits.codec_max))?;
    let io = READ_POOL.get_or_try_init(|| build_pool(limits.io_max, "io"))?;
    Ok((io, codec))
}

/// Resolve the two widths from the three knobs, once per array at open. Both maxes explicit
/// with a total too small to hold them is refused, not clamped: three numbers that disagree is
/// a config error, and silently picking one is how a knob stops arriving.
pub(crate) fn resolve_limits(
    io_max: Option<usize>,
    codec_max: Option<usize>,
    max_workers: Option<usize>,
) -> PyResult<PoolLimits> {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let limits = match max_workers {
        // A total, and at least one side left to us: give the codec side a core each, up to the
        // total, and the rest to the side that blocks.
        Some(total) if io_max.is_none() || codec_max.is_none() => {
            let codec = codec_max.unwrap_or_else(|| total.min(cores)).max(1);
            PoolLimits {
                io_max: io_max.unwrap_or_else(|| total.saturating_sub(codec)).max(1),
                codec_max: codec,
            }
        }
        // ALWAYS a named width, never rayon's own ladder. Leaving it unset would let
        // `RAYON_NUM_THREADS` size the codec pool and nothing else -- not the I/O pool, where
        // most of the threads are -- so a variable meant to tame some other library's rayon
        // usage would silently reshape half of this one. The knobs above say what they mean.
        _ => PoolLimits {
            io_max: io_max.unwrap_or(cores * IO_POOL_MULTIPLIER).max(1),
            codec_max: codec_max.unwrap_or(cores).max(1),
        },
    };
    if let Some(total) = max_workers
        && limits.io_max + limits.codec_max > total
    {
        return Err(PyValueError::new_err(format!(
            "codec_pipeline.io_workers_max ({}) plus codec_workers_max ({}) is {} threads, above \
             codec_pipeline.max_workers ({total})",
            limits.io_max,
            limits.codec_max,
            limits.io_max + limits.codec_max
        )));
    }
    Ok(limits)
}

/// The widths the pools are asked for, resolved once per array at open.
#[derive(Clone, Copy)]
pub(crate) struct PoolLimits {
    pub(crate) io_max: usize,
    pub(crate) codec_max: usize,
}

/// Refuse a max that cannot arrive rather than ignoring it: the pools are process-wide, so a
/// later array asking for different widths gets the ones already standing. This project's rule
/// is that a knob that was set must be a knob that arrived.
fn check_limits_arrived(
    py: Python<'_>,
    io: &rayon::ThreadPool,
    codec: &rayon::ThreadPool,
    limits: PoolLimits,
    strict: bool,
) -> PyResult<()> {
    for (built, asked, knob) in [
        (io.current_num_threads(), limits.io_max, "io_workers_max"),
        (
            codec.current_num_threads(),
            limits.codec_max,
            "codec_workers_max",
        ),
    ] {
        if built == asked {
            continue;
        }
        let message = format!(
            "codec_pipeline.{knob} = {asked} did not arrive: this process built {built} and a \
             pool cannot be resized. The codec pool is fixed by the first array opened, the I/O \
             pool by the first read, so set these before either."
        );
        if strict {
            return Err(PyValueError::new_err(message));
        }
        py.import("warnings")?.call_method1("warn", (message,))?;
    }
    Ok(())
}

/// The widths the two pools were BUILT with, or `None` where one does not exist yet. Read off
/// the pools, never recomputed: a width calculated twice can disagree with the pool it names.
pub(crate) fn pool_sizes() -> (Option<usize>, Option<usize>) {
    (
        READ_POOL.peek().map(|p| p.current_num_threads()),
        crate::pool::peek().map(|p| p.current_num_threads()),
    )
}

/// Refuse a width above what a pool was BUILT with, the one width it cannot give. Fewer than it
/// holds is the ordinary case, so this compares against the pool, not against another call.
fn check_workers_arrived(
    py: Python<'_>,
    io: &rayon::ThreadPool,
    codec: &rayon::ThreadPool,
    config: ReadConfig,
) -> PyResult<()> {
    for (limit, asked, knob) in [
        (io.current_num_threads(), config.io_workers, "io_workers"),
        (
            codec.current_num_threads(),
            config.codec_workers,
            "codec_workers",
        ),
    ] {
        if asked <= limit {
            continue;
        }
        let message = format!(
            "codec_pipeline.{knob} = {asked} is above the {limit} workers this process built, so \
             {limit} will be used. A pool is built once and cannot grow; raise the matching \
             *_max_workers before the first read instead."
        );
        if config.strict {
            return Err(PyValueError::new_err(message));
        }
        py.import("warnings")?.call_method1("warn", (message,))?;
    }
    Ok(())
}

/// What one call reads from `zarr.config` when it starts.
#[derive(Clone, Copy)]
pub(crate) struct ReadConfig {
    /// Workers this call fetches with, out of the I/O pool.
    pub(crate) io_workers: usize,
    /// Decodes this call runs at once, out of the codec pool.
    pub(crate) codec_workers: usize,
    /// Reads a chunk may become before the raw path is declined for it; see [`RAW_MAX_READS`].
    pub(crate) raw_max_reads: usize,
    /// Whether a width above what the pools were built with is an error rather than a warning.
    pub(crate) strict: bool,
}

impl ReadConfig {
    /// Everything this call reads from `zarr.config`, resolved against the pools that will run
    /// it -- so an unset knob defaults to a share of the width that actually exists, and cannot
    /// default to more than the ceiling it is then checked against.
    ///
    /// Zero or absent means "the default", not "none".
    fn from_call(
        py: Python<'_>,
        io_workers: Option<usize>,
        codec_workers: Option<usize>,
        raw_max_reads: Option<usize>,
        strict: bool,
        io: &rayon::ThreadPool,
        codec: &rayon::ThreadPool,
    ) -> PyResult<Self> {
        let _ = py;
        Ok(Self {
            io_workers: io_workers
                .filter(|n| *n > 0)
                .unwrap_or(io.current_num_threads()),
            codec_workers: codec_workers
                .filter(|n| *n > 0)
                .unwrap_or(codec.current_num_threads()),
            raw_max_reads: raw_max_reads.unwrap_or(RAW_MAX_READS),
            strict,
        })
    }
}

/// Everything a call needs from the pools in one lookup: the widths it may use, and both
/// arrival checks. Three separate entry points meant three lock acquisitions per call, and a
/// check that could run against different pools from the ones the defaults came from.
pub(crate) fn config_for_call(
    py: Python<'_>,
    io_workers: Option<usize>,
    codec_workers: Option<usize>,
    raw_max_reads: Option<usize>,
    strict: bool,
    limits: PoolLimits,
) -> PyResult<ReadConfig> {
    let (io, codec) = pools(py, limits)?;
    check_limits_arrived(py, &io, &codec, limits, strict)?;
    let config = ReadConfig::from_call(
        py,
        io_workers,
        codec_workers,
        raw_max_reads,
        strict,
        &io,
        &codec,
    )?;
    check_workers_arrived(py, &io, &codec, config)?;
    Ok(config)
}
struct Job<'a> {
    key: StoreKey,
    /// The chunk's byte range within its shard, or -- on the raw path -- one run of rows'
    /// range inside that chunk.
    range: ByteRange,
    /// Raw jobs carry the wanted bytes exactly: no decode, no scratch, no gather. Their
    /// `range` is the ROW's bytes inside the chunk rather than the whole chunk's.
    raw: bool,
    /// The output ranges this chunk fills, ascending. one range while every axis after the first is
    /// taken whole, which is every rank-1 read, so the CSR path always has one. A shard that
    /// divides a trailing axis gives an item one range per row instead.
    out: Vec<&'a mut [u8]>,
    coords: &'a [u64],
    /// Elements per coordinate; 1 on the 1-D path. See `ChunkItem::run_len`.
    run_len: u64,
    /// Where each run starts inside a coordinate's elements, and how long a run is, when the wanted
    /// elements are not one consecutive span: `oindex[rows, cols]` and any rank-N grid. `None` is a
    /// single contiguous run, which is every other case.
    grid: Option<(&'a [u64], u64)>,
    ctx: &'a JobContext,
}

/// Keep the first failure; later ones are usually consequences of it.
fn record(failure: &Mutex<Option<String>>, message: String) {
    let mut slot = failure.lock().expect("failure slot poisoned");
    if slot.is_none() {
        *slot = Some(message);
    }
}

// Decode scratch, owned by the worker for the life of the process. A decode decompresses a whole
// inner chunk, and past glibc's 128 KiB mmap threshold -- which any chunk worth sharding is --
// that allocation is an mmap, a memset and a fault per page, so it must not be paid per chunk.
thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// One chunk: one store read, then its decode handed to the decode pool.
fn read_one<'scope, 'env>(
    job: Job<'env>,
    dec: &rayon::Scope<'scope>,
    failure: &'env Mutex<Option<String>>,
    ctx: &'env JobContext,
    permits: &'env DecodePermits,
) where
    'env: 'scope,
{
    match ctx.store.get_partial(&job.key, job.range) {
        // `None` means the key is absent, which is a different thing from a range coming back
        // empty. `decode_one` already knows what an absent chunk contributes (the fill value, or an
        // error where a shard index named it) so that logic stays in one place.
        Ok(bytes) => spawn_decode(dec, job, bytes, failure, permits),
        Err(e) => record(failure, format!("read {} failed: {e}", job.key)),
    }
}

/// One chunk's decode, on the decode pool.
/// How many decodes may run at once, out of a pool that could run more. The permit is taken
/// INSIDE the decode task, never before the spawn: a fetcher that waited for a decode slot would
/// couple the pools and idle the device the wide I/O pool exists to keep busy. At the default --
/// the pool's own width -- it never binds, so the cost is one uncontended lock per chunk.
struct DecodePermits {
    free: Mutex<usize>,
    returned: Condvar,
}

impl DecodePermits {
    fn new(limit: usize) -> Self {
        Self {
            free: Mutex::new(limit.max(1)),
            returned: Condvar::new(),
        }
    }

    /// Blocks until a slot is free, and gives it back when the guard drops -- including on a
    /// panic, so one failed decode cannot strand a permit and stall every later chunk.
    fn acquire(&self) -> DecodePermit<'_> {
        let mut free = self.free.lock().unwrap_or_else(PoisonError::into_inner);
        while *free == 0 {
            free = self
                .returned
                .wait(free)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *free -= 1;
        DecodePermit { permits: self }
    }
}

struct DecodePermit<'a> {
    permits: &'a DecodePermits,
}

impl Drop for DecodePermit<'_> {
    fn drop(&mut self) {
        *self
            .permits
            .free
            .lock()
            .unwrap_or_else(PoisonError::into_inner) += 1;
        self.permits.returned.notify_one();
    }
}

fn spawn_decode<'scope, 'env>(
    dec: &rayon::Scope<'scope>,
    mut job: Job<'env>,
    bytes: MaybeBytes,
    failure: &'env Mutex<Option<String>>,
    permits: &'env DecodePermits,
) where
    'env: 'scope,
{
    // every job goes to the pool, including a raw one whose "decode" is only a
    // `copy_from_slice`: a reader that copies inline stops issuing reads while it does.
    dec.spawn(move |_| {
        let _permit = permits.acquire();
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
    // Grow only. `clear()` + `resize(needed, 0)` would zero-fill a buffer that `decode_into`
    // then writes every byte of: a whole-chunk memset per decode, thrown away.
    //
    // A gap left by `decode_into` would show the previous chunk's elements rather than zeros --
    // plausible values instead of an obvious block of nothing, and possibly from an earlier call,
    // since the worker's buffer outlives this one. The view below is the whole chunk, so a codec
    // that can leave a gap is already broken; this only makes such a bug quieter.
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
                // Borrowed. `ArrayBytesRaw` is `Cow<'_, [u8]>` and `Bytes` derefs to `[u8]`,
                // so the decode reads the fetched buffer where it lies. `Cow::Owned` would
                // allocate, and copy the whole compressed chunk whenever the `Bytes` is not
                // uniquely owned, to hand the decoder bytes it already had.
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
            Some((starts, run)) => gather_runs(&scratch[..], job.coords, starts, run, piece, size),
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

    /// An item's output must be one run per axis-0 index, and `output_pieces` is where that
    /// is enforced.
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
        // Taking all of axis 1 and part of axis 2 is also strided: ten runs of five, not one run of
        // fifty. Written out because it is the case I got wrong first: "only the last axis is
        // partial" is not the rule; "every axis before the last partial one takes a single element"
        // is.
        let wide_then_partial = item(&[0..2, 0..10, 0..5], &[6, 10, 10]);
        assert!(
            output_pieces(&wide_then_partial, 8).is_err(),
            "a full axis above a partial one is still strided"
        );
        // One element on axis 1 and part of axis 2 is one run per index, and is served.
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
        // Vending over a gap is allowed (the caller may skip bytes it does not own) but it must not
        // count as covered, or the completeness check in `retrieve_chunk_units` would pass with a
        // hole and hand `np.empty` contents back as data.
        assert!(bytes.take(12, 4).is_some(), "forwards over a gap");
        assert_eq!(
            bytes.covered(),
            12,
            "4 + 4 + 4 vended; the 4-byte hole at 8..12 is not covered"
        );

        assert_eq!(buffer[0], 1);
        assert_eq!(buffer[4], 2);
    }

    /// The pools are built at the size asked for, and the size is one-shot.
    #[test]
    fn the_io_pool_is_wider_than_the_cpu_pool_and_both_report_what_they_built() {
        Python::initialize();
        Python::attach(|py| {
            // A call cannot size the pools -- it masks them. The widths are fixed per process,
            // which is what lets two calls ask differently and both be served.
            let limits = resolve_limits(None, None, None).expect("defaults resolve");
            let (io, codec) = pools(py, limits).expect("the pools must be buildable");
            assert_eq!(io.current_num_threads(), limits.io_max);
            assert_eq!(codec.current_num_threads(), limits.codec_max);
            assert_eq!(
                pool_sizes(),
                (
                    Some(io.current_num_threads()),
                    Some(codec.current_num_threads())
                ),
                "what is reported is read off the pools, not computed a second time"
            );
            // The reason there are two: a fetcher parked on storage must not hold a worker a
            // decode needs, so there are more of the former than there are cores.
            assert!(io.current_num_threads() > codec.current_num_threads());

            // The default a call gets must fit inside them, or every default read trips the
            // ceiling check meant to catch misconfiguration.
            let config = config_for_call(py, None, None, None, true, limits).expect("resolvable");
            assert!(config.io_workers <= io.current_num_threads());
            assert!(config.codec_workers <= codec.current_num_threads());
        });
    }
}

#[cfg(test)]
mod limit_tests {
    use super::*;

    /// The three knobs, and what each combination resolves to.
    #[test]
    fn the_widths_resolve_and_contradictions_are_refused() {
        let cores = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);

        let d = resolve_limits(None, None, None).expect("defaults");
        assert_eq!((d.io_max, d.codec_max), (cores * IO_POOL_MULTIPLIER, cores));

        // A total with both sides left to us: cores for the codec pool, the rest for the one
        // that blocks.
        let t = resolve_limits(None, None, Some(cores + 4)).expect("a total splits");
        assert_eq!(t.codec_max, cores);
        assert_eq!(t.io_max, 4);

        // Explicit sides that do not fit the total contradict, and saying so is the point.
        assert!(resolve_limits(Some(8), Some(8), Some(4)).is_err());
        assert!(resolve_limits(Some(8), Some(8), Some(16)).is_ok());
    }
}
