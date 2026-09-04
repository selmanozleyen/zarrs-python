//! Reading and decoding the innermost chunks of one call, concurrently.
use std::borrow::Cow;
use std::cell::Cell;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyAnyMethods;
use pyo3::{PyResult, Python};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{
    ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset, ArrayToBytesCodecTraits,
    CodecOptions, FillValue,
    ravel_indices,
};
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{MaybeBytes, ReadableStorage, StoreKey};

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
    shard: Arc<ShardInfo>,
    store: ReadableStorage,
    codec_options: CodecOptions,
    element_size: usize,
    /// What an absent chunk contributes. Needed in the workers, not just at carve time,
    /// because an UNSHARDED chunk's absence is only discovered by the read.
    fill_value: FillValue,
    /// Whether missing bytes are ordinary. A shard index that named a chunk which is then
    /// missing means the store changed under the read, and that is worth failing on; an
    /// unsharded chunk has no index to consult, so its key simply may not exist yet.
    may_be_absent: bool,
    /// The unit decoded into scratch: the shard's inner chunk where the array is sharded, the
    /// CHUNK where it is not. Also per call -- an array's chunks are all one shape -- so it is
    /// resolved once here from the first item rather than carried on every `Job`.
    decode_shape: Vec<NonZeroU64>,
}

/// Shard and subshard decoders built during ONE call.
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
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        let element_size = self.element_size()?;
        let ctx = JobContext {
            shard: shard.clone(),
            store: self.readable_store.clone(),
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

        let located = self.locate_chunks(shard, items, &ctx)?;

        let output = DisjointBytes::new(output, output_len);
        let (jobs, absent) = carve(&output, &located, element_size, &ctx)?;
        // Disjointness is proven above; COVERAGE is not. zarr hands us a buffer from
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

        // Two persistent work-stealing pools, and capacity is never divided between calls: a
        // free worker takes the next task, whoever queued it, so a call with more chunks
        // simply gets more workers.
        //
        // READS AND DECODES ARE SEPARATE POOLS. A read blocks on storage; a decode occupies a
        // core. A reader parked on Lustre must occupy a READER, never a decode worker, or one
        // slow shard starves every decode in the process.
        let failure: Mutex<Option<String>> = Mutex::new(None);

        decode_pool(config.decode_ceiling).in_place_scope(|dec| {
            read_pool(config.read_ceiling).in_place_scope(|rd| {
                for job in jobs {
                    let (failure, ctx) = (&failure, &ctx);
                    rd.spawn(move |_| read_one(job, dec, failure, ctx));
                }
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
        // Not sharded: there is no index to read and nothing to descend -- the store value is
        // the chunk. Whether the key EXISTS is the read's business, and a missing one comes
        // back as absent bytes there, exactly as a never-written shard entry does here.
        if shard.depth() == 0 {
            return Ok(Some(ByteRange::FromStart(0, None)));
        }
        let file = key_partial_decoder(&self.readable_store, &item.key);
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

/// Split the output into the disjoint piece each located chunk writes, in offset order.
///
/// Each piece comes from `DisjointBytes::take`, whose cursor only moves forward, so a second
/// claim on the same bytes is refused rather than aliased.
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
            // A range is the chunk's place in its shard; its absence means the chunk was
            // never written, and the output it owns is filled rather than read.
            Some(range) => jobs.push(Job {
                key: item.key.clone(),
                range: *range,
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

/// Hands out each byte range of the output at most once.
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
/// The C-order ravel of the subset's start, which is `ravel_indices`. This used to be the row
/// index times the row length, on the grounds that an item spans every axis after the first
/// whole -- but `output_pieces` admits a BANDED item, whose trailing start is not zero, and
/// two bands of one row then sorted equal. It orders jobs and never places bytes, so that was
/// harmless; the ravel is exact and does not need the invariant to hold.
fn output_offset(item: &ChunkItem) -> u64 {
    let shape = bytemuck::must_cast_slice::<_, u64>(&item.array_shape);
    ravel_indices(item.subset.start(), shape).unwrap_or(u64::MAX)
}

/// What the shard index cache did. Counted because nothing else can: a cache that is never
/// consulted passes every correctness test ever written.
pub(crate) static INDEX_CALL_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INDEX_ARRAY_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INDEX_BUILDS: AtomicU64 = AtomicU64::new(0);

/// The default size of either pool: the machine's parallelism.
fn default_ceiling() -> usize {
    std::thread::available_parallelism().map_or(8, std::num::NonZeroUsize::get)
}

/// The two pools, built once and shared by every call in the process.
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
pub(crate) fn pool_sizes() -> (Option<usize>, Option<usize>) {
    (
        READ_POOL.get().map(rayon::ThreadPool::current_num_threads),
        DECODE_POOL
            .get()
            .map(rayon::ThreadPool::current_num_threads),
    )
}

/// Say so when a ceiling asked for is not the one the pools were built with.
///
/// A warning, not an error. The read is correct at the width already built, and a process that
/// opens a second array wanting a different width is doing something legitimate -- refusing it
/// would turn a sizing hint into a failed read. A caller needing the guarantee asserts on
/// [`pool_sizes`], which is what the benchmark does.
pub(crate) fn check_ceiling_arrived(
    py: Python<'_>,
    config: ReadConfig,
    strict: bool,
) -> PyResult<()> {
    for (built, asked, knob) in [
        (pool_sizes().0, config.read_ceiling, "read_worker_ceiling"),
        (
            pool_sizes().1,
            config.decode_ceiling,
            "decode_worker_ceiling",
        ),
    ] {
        // Only when a pool EXISTS and differs. Before the first read there is nothing to
        // contradict, and the ordinary case costs one atomic load per pool per call.
        let Some(built) = built.filter(|built| *built != asked) else {
            continue;
        };
        let message = format!(
            "codec_pipeline.{knob} = {asked} was ignored: the pool was built with {built} \
             threads by the first read in this process and cannot be resized. Set it before \
             the array that does the first read is opened, or call \
             zarrs._internal.pool_sizes() for what was built."
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

/// What ONE call reads from `zarr.config` when it starts.
#[derive(Clone, Copy)]
pub(crate) struct ReadConfig {
    /// Only the FIRST call's value is used -- see [`READ_POOL`].
    pub(crate) read_ceiling: usize,
    /// Only the FIRST call's value is used -- see [`READ_POOL`].
    pub(crate) decode_ceiling: usize,
}

/// A ceiling as the pipeline will use it: zero or absent means "as much as the machine has".
///
/// Public so the pipeline can resolve at OPEN, which is when these are read.
pub(crate) fn resolve_ceiling(ceiling: Option<usize>) -> usize {
    ceiling.filter(|c| *c > 0).unwrap_or_else(default_ceiling)
}

impl ReadConfig {
    pub(crate) fn from_open(read_ceiling: usize, decode_ceiling: usize) -> Self {
        Self {
            read_ceiling,
            decode_ceiling,
        }
    }
}

/// One innermost chunk, and the slice of the output its elements belong in.
struct Job<'a> {
    key: StoreKey,
    /// The chunk's byte range within its shard.
    range: ByteRange,
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

// Decode scratch, owned by the worker and kept for the life of the process.
//
// A decode decompresses a whole inner chunk (366 KiB sparse, 512 KiB dense here) before the
// wanted rows are copied out. Above glibc's 128 KiB threshold that allocation is an mmap, a
// memset and a fault per page, so it must not be paid per chunk. A rayon worker lives for the
// process, so its own buffer is the reuse -- no lock, and no way for it to silently not run.
thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// One chunk: one store read, then its decode handed to the decode pool.
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
fn spawn_decode<'scope, 'env>(
    dec: &rayon::Scope<'scope>,
    mut job: Job<'env>,
    bytes: MaybeBytes,
    failure: &'env Mutex<Option<String>>,
) where
    'env: 'scope,
{
    // EVERY job goes to the pool: the reader hands off, it never decodes.
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

    /// An item's output must be ONE RUN per axis-0 index, and `output_pieces` is where that
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
