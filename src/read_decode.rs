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
    CodecOptions, FillValue, ravel_indices,
};
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{MaybeBytes, ReadableStorage, StoreKey};

use crate::CodecPipelineImpl;
use crate::chunk_item::ChunkItem;
use crate::shard_index::ShardInfo;
use zarrs::array::codec::api::ByteIntervalPartialDecoder;
use zarrs::array::codec::array_to_bytes::sharding::ShardingPartialDecoder;

use crate::utils::{
    PyCodecErrExt as _, PyErrExt as _, coord_runs, gather, gather_pieces, key_partial_decoder,
};

/// The per-array state a decode needs, shared by every job of a call.
struct JobContext {
    raw: bool,
    shard: Arc<ShardInfo>,
    store: ReadableStorage,
    codec_options: CodecOptions,
    element_size: usize,
    fill_value: FillValue,
    /// An unsharded chunk may be missing; a chunk its shard index names may not.
    may_be_absent: bool,
    /// The inner chunk when sharded, the chunk when not.
    decode_shape: Vec<NonZeroU64>,
}

/// Shard and subshard decoders built during one call.
#[derive(Default)]
struct CallDecoders {
    shards: HashMap<StoreKey, Arc<ShardingPartialDecoder>>,
    subshards: HashMap<(StoreKey, Vec<u64>), Arc<ShardingPartialDecoder>>,
}

impl CodecPipelineImpl {
    /// Read and decode `items`, one job per innermost chunk.
    pub(crate) fn retrieve_chunk_units(
        &self,
        shard: &Arc<ShardInfo>,
        items: &[ChunkItem],
        output: UnsafeCellSlice<'_, u8>,
        output_len: usize,
        pools: &(Arc<QueuePool>, Arc<QueuePool>),
    ) -> PyResult<()> {
        let element_size = self.element_size()?;
        let ctx = JobContext {
            raw: self.inner_chunk_is_raw,
            shard: shard.clone(),
            store: self.readable_store.clone(),
            codec_options: self.codec_options.with_concurrent_target(1),
            element_size,
            fill_value: self.fill_value.clone(),
            may_be_absent: shard.depth() == 0,
            // Unsharded: chunk shapes are uniform, so any item's will do.
            decode_shape: shard.subchunk_shape.as_ref().map_or_else(
                || items.first().map(|i| i.shape.clone()).unwrap_or_default(),
                |shape| shape.to_vec(),
            ),
        };

        let located = self.locate_chunks(shard, items, &ctx)?;

        let output = DisjointBytes::new(output, output_len);
        let (jobs, absent) = carve(&output, &located, element_size, &ctx)?;
        // The output is `np.empty`, so a byte no job owns would be returned uninitialised.
        if output.covered() != output_len {
            return Err(PyRuntimeError::new_err(format!(
                "the batch covers {} of {output_len} output bytes; the rest would be returned \
                 uninitialised",
                output.covered()
            )));
        }

        for piece in absent {
            fill(piece, &self.fill_value, element_size).map_py_err::<PyRuntimeError>()?;
        }
        if jobs.is_empty() {
            return Ok(());
        }

        let failure: Mutex<Option<String>> = Mutex::new(None);

        // Every job is queued at once and the pool widths are the only bound. A reader hands its
        // chunk to the decode pool, and `batch` returns once both have run.
        let (read_pool, decode_pool) = pools;
        batch(|b| {
            for job in jobs {
                let (failure, ctx, task_b) = (&failure, &ctx, b.clone());
                // SAFETY: `jobs`, `failure`, `ctx` and the pools all outlive this `batch` call.
                unsafe { b.spawn(read_pool, move || read_one(job, decode_pool, &task_b, failure, ctx)) };
            }
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
        // Not sharded: the store value is the chunk.
        if shard.depth() == 0 {
            return Ok(Some(ByteRange::FromStart(0, None)));
        }
        let file = key_partial_decoder(&self.readable_store, &item.key);
        let mut shard_shape = item.shape.clone();
        let mut offset: Vec<u64> = start.to_vec();
        // (offset, length) of the level being descended into, absolute in the store value.
        let mut extent: Option<(u64, u64)> = None;
        let mut path: Vec<u64> = Vec::new();

        for depth in 0..shard.depth() {
            let level_shape = shard.subchunk_shape_at(depth);
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
                return Ok(None);
            };
            // Always `FromStart` with an explicit length, so the `size` argument is unused.
            let base = extent.map_or(0, |(base, _)| base);
            extent = Some((base + range.start(0), range.length(0)));

            shard_shape.clone_from(shard.subchunk_shape_at(depth));
            if depth + 1 < shard.depth() {
                path.extend_from_slice(&grid_index);
            }
        }
        // `push_entry` takes arbitrary arguments, and an item overflowing its inner chunk would
        // read the wrong elements.
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
            coords_of(item)?;
            let start = item.chunk_subset.start().to_vec();
            located.push((item, self.locate(shard, item, &start, ctx, &mut decoders)?));
        }
        Ok(located)
    }
}

/// Split the output into the disjoint piece each located chunk writes, in offset order.
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
    // `decode_shape` is empty for an empty unsharded batch; the caller's coverage check reports it.
    if located.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let real_inner = (
        ctx.decode_shape[0].get(),
        ctx.decode_shape[1..].iter().map(|d| d.get()).product::<u64>(),
    );
    for (i, (item, _)) in located.iter().enumerate() {
        let coords = coords_of(item)?;
        // The coordinates are only right against the inner chunk they were built for.
        if item.claimed_inner != real_inner {
            return Err(PyRuntimeError::new_err(format!(
                "{} was described against an inner chunk of split {} and row stride {}, but \
                 this array decodes {} by {}; its coordinates would address the wrong elements",
                item.key,
                item.claimed_inner.0,
                item.claimed_inner.1,
                real_inner.0,
                real_inner.1,
            )));
        }
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
    // Queued in output order, so on high-latency storage chunks arrive roughly as wanted.
    let mut order: Vec<usize> = (0..located.len()).collect();
    order.sort_by_key(|&i| output_offset(located[i].0));
    for i in order {
        let (item, range) = &located[i];
        let pieces = std::mem::take(&mut taken[i]);
        match range {
            // Uncompressed: read each run of rows as its own byte range, if that is few reads.
            // A banded item has several pieces and takes the chunk path.
            Some(range)
                if ctx.raw
                    && pieces.len() == 1
                    && raw_runs(coords_of(item)?, item.run_len) <= RAW_MAX_READS =>
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
                let coords = coords_of(item)?;
                // Adjacent items on the same chunk share one read and decode.
                if let Some(last) = jobs
                    .last_mut()
                    .filter(|j| !j.raw && j.key == item.key && j.range == *range)
                {
                    last.shared.push(Part {
                        out: pieces,
                        coords,
                        run_len: item.run_len,
                    });
                    continue;
                }
                CHUNK_JOBS.fetch_add(1, Ordering::Relaxed);
                jobs.push(Job {
                    key: item.key.clone(),
                    range: *range,
                    raw: false,
                    out: pieces,
                    coords,
                    run_len: item.run_len,
                    shared: Vec::new(),
                    ctx,
                });
            }
            None => absent.extend(pieces),
        }
    }
    Ok((jobs, absent))
}

pub(crate) static RAW_JOBS: AtomicU64 = AtomicU64::new(0);
pub(crate) static CHUNK_JOBS: AtomicU64 = AtomicU64::new(0);

/// How many reads this chunk's rows become once consecutive ones are merged.
pub(crate) fn raw_runs(coords: &[u64], run_len: u64) -> usize {
    coord_runs(coords, run_len).count()
}

/// Past this many reads per chunk, reading the whole chunk is cheaper.
const RAW_MAX_READS: usize = 2;

/// One job per run of consecutive rows, each reading exactly its own bytes.
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
            shared: Vec::new(),
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

/// Hands out each byte range of the output at most once: `cursor` only moves forward.
struct DisjointBytes<'a> {
    slice: UnsafeCellSlice<'a, u8>,
    len: usize,
    cursor: Cell<usize>,
    /// Bytes actually vended; `cursor` would count a skipped gap.
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

    fn covered(&self) -> usize {
        self.covered.get()
    }

    /// `None` if the range runs backwards into one already handed out, or past the buffer.
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

/// Where an item's output starts, as a flat element offset. Orders jobs only.
fn output_offset(item: &ChunkItem) -> u64 {
    let shape = bytemuck::must_cast_slice::<_, u64>(&item.array_shape);
    ravel_indices(item.subset.start(), shape).unwrap_or(u64::MAX)
}

pub(crate) static INDEX_CALL_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INDEX_ARRAY_HITS: AtomicU64 = AtomicU64::new(0);
pub(crate) static INDEX_BUILDS: AtomicU64 = AtomicU64::new(0);

fn default_pool_size() -> usize {
    std::thread::available_parallelism().map_or(8, std::num::NonZeroUsize::get)
}

/// Reads kept in flight however few the cores; an idle reader is a parked thread.
const READ_FLOOR: usize = 64;

/// Default `(read, decode)` widths: decoders get every core up to 16 and half of them past that.
fn default_widths(cores: usize) -> (usize, usize) {
    (cores.max(READ_FLOOR), if cores <= 16 { cores } else { cores / 2 })
}

static READ_POOL: OnceLock<Arc<QueuePool>> = OnceLock::new();

type QueueTask = Box<dyn FnOnce() + Send + 'static>;
type Panic = Box<dyn std::any::Any + Send>;

/// Worker threads blocked on one queue, built once and never resized.
pub(crate) struct QueuePool {
    tx: crossbeam_channel::Sender<QueueTask>,
    width: usize,
}

impl QueuePool {
    fn new(width: usize, name: &'static str) -> PyResult<Self> {
        let (tx, rx) = crossbeam_channel::unbounded::<QueueTask>();
        for i in 0..width {
            let rx = rx.clone();
            std::thread::Builder::new()
                .name(format!("zarrs-{name}-{i}"))
                .spawn(move || rx.iter().for_each(|task| task()))
                .map_err(|e| PyRuntimeError::new_err(format!("could not create the {name} pool: {e}")))?;
        }
        Ok(Self { tx, width })
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }
}

/// The handle every task of one `batch` carries; the batch returns once all are dropped.
#[derive(Clone)]
pub(crate) struct Batch {
    wg: crossbeam_utils::sync::WaitGroup,
    panic: Arc<Mutex<Option<Panic>>>,
}

impl Batch {
    /// # Safety
    /// Whatever `task` borrows must outlive the `batch` call this handle came from.
    unsafe fn spawn<'a>(&self, pool: &QueuePool, task: impl FnOnce() + Send + 'a) {
        let b = self.clone();
        let task: Box<dyn FnOnce() + Send + 'a> = Box::new(move || {
            // The whole handle: a 2021 closure naming only `b.panic` captures just that field,
            // and the `WaitGroup` would be dropped before the task had run.
            let b = b;
            if let Err(p) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(task)) {
                let mut slot = b.panic.lock().expect("panic slot poisoned");
                if slot.is_none() {
                    *slot = Some(p);
                }
            }
        });
        // SAFETY: `b` goes when the task finishes, and `batch` waits for every clone of it.
        let task: QueueTask = unsafe { std::mem::transmute(task) };
        pool.tx.send(task).expect("a pool's queue is never closed");
    }
}

/// Run `submit`, which queues tasks through the handle it is given, and return once every one
/// has finished, including tasks queued by other tasks. Re-raises the first task panic.
pub(crate) fn batch<R>(submit: impl FnOnce(Batch) -> R) -> R {
    struct WaitOnDrop(crossbeam_utils::sync::WaitGroup);
    impl Drop for WaitOnDrop {
        fn drop(&mut self) {
            std::mem::take(&mut self.0).wait();
        }
    }
    let root = crossbeam_utils::sync::WaitGroup::new();
    let panic = Arc::new(Mutex::new(None));
    let b = Batch { wg: root.clone(), panic: panic.clone() };
    // Waits on drop as well, so an unwind out of `submit` still cannot outlive a task.
    let wait = WaitOnDrop(root);
    let r = submit(b);
    drop(wait);
    if let Some(p) = panic.lock().expect("panic slot poisoned").take() {
        std::panic::resume_unwind(p);
    }
    r
}

static DECODE_POOL: OnceLock<Arc<QueuePool>> = OnceLock::new();

/// `(read, decode)` pools, sized by the first call of the process and never resized.
///
/// Takes the `Python` token so it is called with the GIL held.
pub(crate) fn pools(
    _py: Python<'_>,
    config: ReadConfig,
) -> PyResult<(Arc<QueuePool>, Arc<QueuePool>)> {
    // Inherited pools have queues but no threads, so a forked child would wait forever.
    crate::fork::check()?;
    let cpu = match DECODE_POOL.get() {
        Some(p) => p.clone(),
        None => {
            let built = Arc::new(QueuePool::new(config.decode_workers, "decode")?);
            DECODE_POOL.get_or_init(|| built).clone()
        }
    };
    let io = match READ_POOL.get() {
        Some(p) => p.clone(),
        None => {
            let built = Arc::new(QueuePool::new(config.read_workers, "read")?);
            READ_POOL.get_or_init(|| built).clone()
        }
    };
    Ok((io, cpu))
}

pub(crate) fn pool_sizes() -> (Option<usize>, Option<usize>) {
    (
        READ_POOL.get().map(|p| p.width()),
        DECODE_POOL.get().map(|p| p.width()),
    )
}

/// Warn, or raise under `strict`, when a call asks for other widths than the pools have.
pub(crate) fn check_workers_arrived(
    py: Python<'_>,
    config: ReadConfig,
    (io, dec): &(Arc<QueuePool>, Arc<QueuePool>),
) -> PyResult<()> {
    for (limit, asked, knob) in [
        (io.width(), config.read_workers, "read_workers"),
        (dec.width(), config.decode_workers, "decode_workers"),
    ] {
        if asked == limit {
            continue;
        }
        let message = format!(
            "codec_pipeline.{knob} = {asked} but this process built {limit} workers, \
             so {limit} will be used. The pools are sized by the first call and cannot change."
        );
        if config.strict {
            return Err(PyValueError::new_err(message));
        }
        py.import("warnings")?.call_method1("warn", (message,))?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct ReadConfig {
    pub(crate) read_workers: usize,
    pub(crate) decode_workers: usize,
    pub(crate) strict: bool,
}

impl ReadConfig {
    /// Zero or absent means the default for this machine.
    pub(crate) fn from_call(
        read_workers: Option<usize>,
        decode_workers: Option<usize>,
        strict: bool,
    ) -> Self {
        let (read, decode) = default_widths(default_pool_size());
        Self {
            read_workers: read_workers.filter(|n| *n > 0).unwrap_or(read),
            decode_workers: decode_workers.filter(|n| *n > 0).unwrap_or(decode),
            strict,
        }
    }
}

/// One read: an innermost chunk, or on the raw path one run of rows inside it.
struct Job<'a> {
    key: StoreKey,
    range: ByteRange,
    /// The read is the output bytes: no decode, no gather.
    raw: bool,
    out: Vec<&'a mut [u8]>,
    coords: &'a [u64],
    run_len: u64,
    /// Further items gathered from the same decoded chunk.
    shared: Vec<Part<'a>>,
    ctx: &'a JobContext,
}

struct Part<'a> {
    out: Vec<&'a mut [u8]>,
    coords: &'a [u64],
    run_len: u64,
}

/// Keep the first failure; later ones are usually consequences of it.
fn record(failure: &Mutex<Option<String>>, message: String) {
    let mut slot = failure.lock().expect("failure slot poisoned");
    if slot.is_none() {
        *slot = Some(message);
    }
}

// Per-worker decode scratch, reused so a decode does not pay an mmap and page faults.
thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// One chunk: one store read, then its decode handed to the decode pool.
fn read_one<'env>(
    job: Job<'env>,
    dec: &'env QueuePool,
    b: &Batch,
    failure: &'env Mutex<Option<String>>,
    ctx: &'env JobContext,
) {
    match ctx.store.get_partial(&job.key, job.range) {
        Ok(bytes) => spawn_decode(dec, b, job, bytes, failure),
        Err(e) => record(failure, format!("read {} failed: {e}", job.key)),
    }
}

/// One chunk's decode, on the decode pool.
fn spawn_decode<'env>(
    dec: &QueuePool,
    b: &Batch,
    mut job: Job<'env>,
    bytes: MaybeBytes,
    failure: &'env Mutex<Option<String>>,
) {
    // Raw jobs too: a reader that copies inline stops issuing reads.
    // SAFETY: the job and `failure` borrow from the caller of `batch`, which outlives it.
    unsafe {
        b.spawn(dec, move || {
            SCRATCH.with(|cell| {
                let mut scratch = cell.borrow_mut();
                if let Err(e) = decode_one(&mut job, bytes, &mut scratch) {
                    record(failure, e);
                }
            });
        });
    }
}

/// Decode one innermost chunk into scratch, then gather the wanted elements into `out`.
fn decode_one(job: &mut Job<'_>, bytes: MaybeBytes, scratch: &mut Vec<u8>) -> Result<(), String> {
    let ctx = job.ctx;
    let size = ctx.element_size;
    let Some(bytes) = bytes else {
        if ctx.may_be_absent {
            let parts = job.shared.iter_mut().flat_map(|p| p.out.iter_mut());
            for piece in job.out.iter_mut().chain(parts) {
                fill(piece, &ctx.fill_value, size)?;
            }
            return Ok(());
        }
        return Err(format!("{} vanished between index and read", job.key));
    };

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

    // The item wants the whole decode unit: decode straight into the output.
    let whole_unit = job.shared.is_empty()
        && job.out.len() == 1
        && job.out[0].len() == needed
        && job.coords.first() == Some(&0)
        && job.coords.len() as u64 * job.run_len == elements
        && coord_runs(job.coords, job.run_len).nth(1).is_none();
    if whole_unit {
        let shape_u64: Vec<u64> = shape.iter().map(|s| s.get()).collect();
        let slice = UnsafeCellSlice::new(&mut job.out[0][..]);
        let mut view = unsafe {
            // SAFETY: this view is the only writer to that piece, which `DisjointBytes` vended
            // to this job alone and no other job can hold.
            ArrayBytesFixedDisjointView::new(
                slice,
                size,
                &shape_u64,
                ArraySubset::new_with_shape(shape_u64.clone()),
            )
            .map_err(|e| e.to_string())?
        };
        return ctx
            .shard
            .inner_chain
            .decode_into(
                Cow::Borrowed(&bytes),
                shape,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                &ctx.codec_options,
            )
            .map_err(|e| e.to_string());
    }
    // Grow only: `decode_into` overwrites it.
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
                Cow::Borrowed(&bytes),
                shape,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                &ctx.codec_options,
            )
            .map_err(|e| e.to_string())?;
    }
    gather_part(scratch, &mut job.out, job.coords, job.run_len, size)
        .map_err(|e| format!("{}: {e}", job.key))?;
    for part in &mut job.shared {
        gather_part(scratch, &mut part.out, part.coords, part.run_len, size)
            .map_err(|e| format!("{}: {e}", job.key))?;
    }
    Ok(())
}

/// Gather what one item wants out of a decoded chunk into its output pieces.
fn gather_part(
    scratch: &[u8],
    out: &mut [&mut [u8]],
    coords: &[u64],
    run_len: u64,
    size: usize,
) -> Result<(), String> {
    if let [piece] = out {
        gather(scratch, coords, run_len, piece, size)
    } else {
        gather_pieces(scratch, coords, run_len, out, size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            claimed_inner: (0, 0),
        };
        // Strided: axis 1 takes 5 of 10 and axis 2 takes 5 of 10, so a row is not one run.
        let strided = item(&[0..2, 0..5, 0..5], &[6, 10, 10]);
        assert!(
            output_pieces(&strided, 8).is_err(),
            "a strided output sub-box must be refused, not modelled as one run"
        );
        // Taking all of axis 1 and part of axis 2 is ten runs of five, not one run.
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
        // Vending over a gap is allowed but the gap is not covered.
        assert!(bytes.take(12, 4).is_some(), "forwards over a gap");
        assert_eq!(
            bytes.covered(),
            12,
            "4 + 4 + 4 vended; the 4-byte hole at 8..12 is not covered"
        );

        assert_eq!(buffer[0], 1);
        assert_eq!(buffer[4], 2);
    }

    /// A batch returns only after every task, including ones queued by other tasks on the
    /// other pool, and a panic comes back to the caller without costing a pool a thread.
    #[test]
    fn batch_waits_for_nested_tasks_and_survives_a_panic() {
        let (reads, decodes) = (QueuePool::new(4, "t-read").unwrap(), QueuePool::new(2, "t-dec").unwrap());
        let mut out = vec![0u64; 1000];
        batch(|b| {
            for (i, c) in out.chunks_mut(10).enumerate() {
                let (inner, decodes) = (b.clone(), &decodes);
                // SAFETY: `out` and `decodes` outlive the batch.
                unsafe {
                    b.spawn(&reads, move || {
                        std::thread::sleep(std::time::Duration::from_micros(50));
                        inner.spawn(decodes, move || {
                            // Slow enough that a batch returning early is always caught.
                            std::thread::sleep(std::time::Duration::from_millis(2));
                            c.iter_mut().for_each(|x| *x = i as u64 + 1);
                        });
                    });
                }
            }
        });
        assert!(out.chunks(10).enumerate().all(|(i, c)| c.iter().all(|&x| x == i as u64 + 1)));

        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            batch(|b| unsafe { b.spawn(&decodes, || panic!("decode failed")) });
        }));
        assert!(caught.is_err(), "the task's panic reaches the caller");
        let mut ran = [false; 8];
        batch(|b| {
            for r in ran.iter_mut() {
                // SAFETY: `ran` outlives the batch.
                unsafe { b.spawn(&decodes, move || *r = true) };
            }
        });
        assert!(ran.iter().all(|&r| r), "both decode workers are still there");
    }

    /// The defaults at the sizes they were measured on.
    #[test]
    fn default_widths_follow_the_measured_rule() {
        for (cores, want) in [
            (1, (64, 1)),
            (4, (64, 4)),
            (8, (64, 8)),
            (16, (64, 16)),
            (32, (64, 16)),
            (90, (90, 45)),
            (200, (200, 100)),
        ] {
            assert_eq!(default_widths(cores), want, "{cores} cpus");
        }
    }

    /// The pools are built at the size asked for, and the size is one-shot.
    #[test]
    fn the_io_pool_is_wider_than_the_cpu_pool_and_both_report_what_they_built() {
        Python::initialize();
        Python::attach(|py| {
            let config = ReadConfig::from_call(None, None, true);
            let (io, cpu) = pools(py, config).expect("the pools must be buildable");
            assert_eq!(
                (io.width(), cpu.width()),
                default_widths(default_pool_size()),
                "with nothing asked, the pools are the defaults for this machine"
            );
            assert_eq!(
                pool_sizes(),
                (
                    Some(io.width()),
                    Some(cpu.width())
                ),
                "what is reported is read off the pools, not computed a second time"
            );
            assert!(io.width() > cpu.width());
            assert!(config.read_workers <= io.width());
            assert!(config.decode_workers <= cpu.width());

            // A later call naming a different width is answered by the pool that exists.
            let narrower = ReadConfig::from_call(Some(1), Some(1), false);
            let (io2, cpu2) = pools(py, narrower).expect("already built");
            assert_eq!(io2.width(), io.width());
            assert_eq!(cpu2.width(), cpu.width());
        });
    }
}
