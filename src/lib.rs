#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, OnceLock};

use chunk_item::ChunkItem;
use itertools::Itertools;
use numpy::npyffi::PyArrayObject;
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use unsafe_cell_slice::UnsafeCellSlice;
use utils::is_whole_chunk;
use zarrs::array::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayMetadata,
    ArrayPartialDecoderTraits, ArraySubset, ArrayToBytesCodecTraits, CodecChain, CodecChainBound,
    CodecOptions, DataPlan, DataType, FillValue, IndexPlan, ReadPlan, copy_fill_value_into,
    update_array_bytes,
};
use zarrs::config::global_config;
use zarrs::convert::array_metadata_v2_to_v3;
use zarrs::plugin::ZarrVersion;
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{
    MaybeBytes, ReadableWritableListableStorage, StorageError, StorageHandle, StoreKey,
};

mod chunk_item;
mod concurrency;
mod runtime;
mod store;
#[cfg(test)]
mod tests;
mod utils;

use crate::concurrency::ChunkConcurrentLimitAndCodecOptions;
use crate::store::StoreConfig;
use crate::utils::{PyCodecErrExt, PyErrExt as _};

/// Fetch threads when `codec_pipeline.fetch_threads` is unset.
///
/// Sized by how many reads parallel storage will usefully carry at once, not by core
/// count -- a fetch thread parks in the storage read and burns no CPU.
const FETCH_THREADS_DEFAULT: usize = 32;

/// In-flight byte budget when `codec_pipeline.fetch_byte_budget` is unset.
///
/// 512 MiB: far more than the outstanding reads that saturate parallel storage, so the
/// bound does not bite on ordinary batches and only clips pathological ones.
const FETCH_BYTE_BUDGET_DEFAULT: u64 = 512 * 1024 * 1024;

/// How many encoded bytes one read brings in.
fn range_bytes(byte_range: ByteRange) -> u64 {
    match byte_range {
        ByteRange::FromStart(_, Some(length)) | ByteRange::Suffix(length) => length,
        // A range with no length reads to the end, which is not something a shard
        // index describes. Counted as nothing rather than guessed at.
        ByteRange::FromStart(_, None) => 0,
    }
}

/// How many encoded bytes a plan's reads will bring in.
fn plan_bytes(plan: &DataPlan) -> u64 {
    plan.reads().map(|(_, byte_range)| range_bytes(byte_range)).sum()
}

/// The pool that planned reads are issued on.
///
/// Separate from the rayon pool that decodes, so a slow store cannot starve decoding
/// and vice versa.
///
/// One pool per process, sized by the first pipeline that asks. `zarr.config` is
/// process-global, so every pipeline normally asks for the same size; a pipeline
/// constructed under a different `fetch_threads` shares the first pool rather than
/// stacking a second pool's threads on top of the same storage. Zero threads means no
/// pool, which disables planning for every pipeline in the process.
fn fetch_pool(threads: usize) -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        (threads > 0)
            .then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|index| format!("zarrs-fetch-{index}"))
                    .build()
                    .ok()
            })
            .flatten()
    })
    .as_ref()
}

/// The pool planned entries decode on, the moment their read lands.
///
/// A fetch thread must not decode -- that turns an I/O thread into a CPU consumer --
/// but `rayon::spawn` from inside the fetch pool would do exactly that, since it
/// targets the current pool. This is the explicit CPU-sized target instead: the fetch
/// thread hands the bytes here and returns to reading. One pool per process, sized by
/// the first pipeline that asks, exactly as [`fetch_pool`] is.
fn decode_pool(threads: usize) -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|index| format!("zarrs-decode-{index}"))
            .build()
            .ok()
    })
    .as_ref()
}

// TODO: Use a OnceLock for store with get_or_try_init when stabilised?
#[gen_stub_pyclass]
#[pyclass]
pub struct CodecPipelineImpl {
    pub(crate) store: ReadableWritableListableStorage,
    pub(crate) codec_chain: Arc<CodecChainBound>,
    pub(crate) codec_options: CodecOptions,
    pub(crate) chunk_concurrent_minimum: usize,
    pub(crate) chunk_concurrent_maximum: usize,
    pub(crate) num_threads: usize,
    pub(crate) fill_value: FillValue,
    pub(crate) data_type: DataType,
    /// Encoded bytes allowed in flight across all planned chunks at once.
    ///
    /// Planning makes every read in a batch visible at once, which is the point -- but
    /// issuing them all at once means the peak is the whole batch rather than a window
    /// of it. A batch is bounded by rows, not by bytes, so nothing else bounds this.
    pub(crate) fetch_byte_budget: u64,
    /// Threads the process-wide fetch pool was asked for. Zero disables planning.
    pub(crate) fetch_threads: usize,
    /// Partial decoders kept between calls, each holding its shard's decoded
    /// index.
    ///
    /// Building one reads that index from storage, so without this a minibatch
    /// loader re-reads every index it touches on every batch -- measured at
    /// ~820 shards and ~11.5 MiB per batch on a real plate, none of it changing
    /// between batches.
    ///
    /// On by default; `codec_pipeline.partial_decoder_cache = False` opts out. A cached
    /// decoder holds an index that a write invalidates, so this is only sound while
    /// every write goes through [`store_chunks_with_indices`], which evicts what it
    /// touches. A writer outside this pipeline -- another process, another library --
    /// is the case to opt out for.
    pub(crate) decoder_cache: Option<Mutex<HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>>>,
}

impl CodecPipelineImpl {
    fn retrieve_chunk_bytes<'a>(
        &self,
        item: &ChunkItem,
        codec_chain: &CodecChainBound,
        codec_options: &CodecOptions,
    ) -> PyResult<ArrayBytes<'a>> {
        let value_encoded = self.store.get(&item.key).map_py_err::<PyRuntimeError>()?;
        let value_decoded = if let Some(value_encoded) = value_encoded {
            let value_encoded: Vec<u8> = value_encoded.into(); // zero-copy in this case
            codec_chain
                .decode(value_encoded.into(), &item.shape, codec_options)
                .map_codec_err()?
        } else {
            ArrayBytes::new_fill_value(&self.data_type, item.num_elements, &self.fill_value)
                .map_py_err::<PyRuntimeError>()?
        };
        Ok(value_decoded)
    }

    fn store_chunk_bytes(
        &self,
        item: &ChunkItem,
        codec_chain: &CodecChainBound,
        value_decoded: ArrayBytes,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        value_decoded
            .validate(item.num_elements, &self.data_type)
            .map_codec_err()?;

        if value_decoded.is_fill_value(&self.fill_value) {
            self.store.erase(&item.key).map_py_err::<PyRuntimeError>()
        } else {
            let value_encoded = codec_chain
                .encode(value_decoded, &item.shape, codec_options)
                .map(Cow::into_owned)
                .map_codec_err()?;

            // Store the encoded chunk
            self.store
                .set(&item.key, value_encoded.into())
                .map_py_err::<PyRuntimeError>()
        }
    }

    fn store_chunk_subset_bytes(
        &self,
        item: &ChunkItem,
        codec_chain: &CodecChainBound,
        chunk_subset_bytes: ArrayBytes,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        let array_shape = &item.shape;
        let chunk_subset = &item.chunk_subset;
        if !chunk_subset.inbounds_shape(bytemuck::must_cast_slice(array_shape)) {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "chunk subset ({chunk_subset}) is out of bounds for array shape ({array_shape:?})"
            )));
        }
        let data_type_size = self.data_type.size();

        if is_whole_chunk(item) {
            // Fast path if the chunk subset spans the entire chunk, no read required
            self.store_chunk_bytes(item, codec_chain, chunk_subset_bytes, codec_options)
        } else {
            // Validate the chunk subset bytes
            chunk_subset_bytes
                .validate(chunk_subset.num_elements(), &self.data_type)
                .map_codec_err()?;

            // Retrieve the chunk
            let chunk_bytes_old = self.retrieve_chunk_bytes(item, codec_chain, codec_options)?;

            // Update the chunk
            let chunk_bytes_new = update_array_bytes(
                chunk_bytes_old,
                bytemuck::must_cast_slice(array_shape),
                chunk_subset,
                &chunk_subset_bytes,
                data_type_size,
            )
            .map_codec_err()?;

            // Store the updated chunk
            self.store_chunk_bytes(item, codec_chain, chunk_bytes_new, codec_options)
        }
    }

    fn py_untyped_array_to_array_object<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
    ) -> &'a PyArrayObject {
        // TODO: Upstream a PyUntypedArray.as_array_ref()?
        //       https://github.com/zarrs/zarrs-python/pull/80/files/75be39184905d688ac04a5f8bca08c5241c458cd#r1918365296
        let array_object_ptr: NonNull<PyArrayObject> = NonNull::new(value.as_array_ptr())
            .expect("bug in numpy crate: Bound<'_, PyUntypedArray>::as_array_ptr unexpectedly returned a null pointer");
        let array_object: &'a PyArrayObject = unsafe {
            // SAFETY: the array object pointed to by array_object_ptr is valid for 'a
            array_object_ptr.as_ref()
        };
        array_object
    }

    fn nparray_to_slice<'a>(value: &'a Bound<'_, PyUntypedArray>) -> Result<&'a [u8], PyErr> {
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let array_object: &PyArrayObject = Self::py_untyped_array_to_array_object(value);
        let array_data = array_object.data.cast::<u8>();
        let array_len = value.len() * value.dtype().itemsize();
        let slice = unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts(array_data, array_len)
        };
        Ok(slice)
    }

    fn nparray_to_unsafe_cell_slice<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
    ) -> Result<UnsafeCellSlice<'a, u8>, PyErr> {
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let array_object: &PyArrayObject = Self::py_untyped_array_to_array_object(value);
        let array_data = array_object.data.cast::<u8>();
        let array_len = value.len() * value.dtype().itemsize();
        let output = unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts_mut(array_data, array_len)
        };
        Ok(UnsafeCellSlice::new(output))
    }
    /// Exchange every index plan for the plan of the data it locates.
    ///
    /// A nested shard's first plan names its inner-shard indexes. All such reads
    /// across the batch are fetched in one parallel wave, then each exchange is
    /// pure computation -- and the types bound the whole thing to one round,
    /// since refining returns a `DataPlan`. The index reads are a few dozen
    /// bytes per touched subchunk, so they are not counted against the data
    /// byte budget.
    fn resolve_plans(
        &self,
        plans: Vec<Option<ReadPlan>>,
        items: &[ChunkItem],
        pool: &rayon::ThreadPool,
        codec_options: &CodecOptions,
    ) -> PyResult<Vec<Option<DataPlan>>> {
        let mut data: Vec<Option<DataPlan>> = plans.iter().map(|_| None).collect();
        let mut staged: Vec<(usize, IndexPlan)> = Vec::new();
        for (index, plan) in plans.into_iter().enumerate() {
            match plan {
                Some(ReadPlan::Data(plan)) => data[index] = Some(plan),
                Some(ReadPlan::Indexes(plan)) => staged.push((index, plan)),
                None => {}
            }
        }
        if staged.is_empty() {
            return Ok(data);
        }

        let reads: Vec<(usize, usize, ByteRange)> = staged
            .iter()
            .enumerate()
            .flat_map(|(slot, (_, plan))| {
                plan.reads().map(move |(entry, range)| (slot, entry, range))
            })
            .collect();
        let results: Vec<(usize, usize, MaybeBytes)> = pool
            .install(|| {
                reads
                    .into_par_iter()
                    .map(|(slot, entry, range)| {
                        let key = &items[staged[slot].0].key;
                        let bytes = self.store.get_partial(key, range)?;
                        Ok((slot, entry, bytes))
                    })
                    .collect::<Result<_, StorageError>>()
            })
            .map_py_err::<PyRuntimeError>()?;

        // Scatter the bytes back to plan positions; refining performs no I/O.
        let mut fetched: Vec<Vec<MaybeBytes>> = staged
            .iter()
            .map(|(_, plan)| vec![None; plan.num_entries()])
            .collect();
        for (slot, entry, bytes) in results {
            fetched[slot][entry] = bytes;
        }
        for ((index, plan), bytes) in staged.into_iter().zip(fetched) {
            data[index] = Some(plan.refine(bytes, codec_options).map_codec_err()?);
        }
        Ok(data)
    }

    /// Decode the chunks that can report their reads, dispatching each read the
    /// moment it lands.
    ///
    /// `partial_decode_into` works out which bytes it needs, reads them, and
    /// decodes them in one call, so overlapping several chunks means a thread
    /// per chunk parked on its own read. Planning first lets every chunk's
    /// reads be outstanding at once against a pool sized for storage rather
    /// than for cores -- and entries are independent, so each read decodes as
    /// it arrives instead of waiting for its chunk's slowest read. The fetch
    /// thread never decodes: it hands the bytes to the decode pool and returns
    /// to reading.
    ///
    /// Returns which items it handled. Anything it could not plan for -- a codec
    /// that does not report reads -- is left alone and takes the ordinary path.
    fn decode_planned_chunks(
        &self,
        items: &[ChunkItem],
        decoders: &HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>,
        output: UnsafeCellSlice<u8>,
        pool: &'static rayon::ThreadPool,
        codec_options: &CodecOptions,
    ) -> PyResult<Vec<bool>> {
        // Ask every chunk what it needs to read, before reading anything. The
        // shard indexes are already resident, so this touches no storage.
        let plans: Vec<Option<ReadPlan>> = items
            .iter()
            .map(|item| {
                if is_whole_chunk(item) {
                    return Ok(None);
                }
                // `into_planned` is None for any decoder that cannot describe
                // its reads, which is every codec but sharding.
                decoders
                    .get(&item.key)
                    .cloned()
                    .and_then(|decoder| decoder.into_planned())
                    .map(|planned| planned.read_plan(&item.chunk_subset, codec_options))
                    .transpose()
                    .map_codec_err()
                    .map(Option::flatten)
            })
            .collect::<PyResult<Vec<_>>>()?;

        // A nested shard's first plan names its inner-shard indexes rather than the
        // data. One batched wave of index reads exchanges them all for data plans.
        let plans = self.resolve_plans(plans, items, pool, codec_options)?;
        let handled = plans.iter().map(Option::is_some).collect::<Vec<_>>();

        let Some(decode_pool) = decode_pool(self.num_threads) else {
            return Ok(vec![false; items.len()]);
        };
        let chunks: Vec<PlannedChunk> = plans
            .into_iter()
            .zip(items)
            .filter_map(|(plan, item)| {
                plan.map(|plan| PlannedChunk {
                    key: item.key.clone(),
                    subset: item.subset.clone(),
                    array_shape: item.array_shape.clone(),
                    plan,
                })
            })
            .collect();
        if chunks.is_empty() {
            return Ok(handled);
        }
        let data_type_size = self
            .data_type
            .fixed_size()
            .ok_or("variable length data type not supported")
            .map_py_err::<PyTypeError>()?;

        // One completion per accounted task: an absent fill per chunk, plus one
        // decode (or failed fetch) per entry. Receiving exactly this many is the
        // barrier that keeps every borrow alive until the last task is done.
        let total = chunks.len()
            + chunks
                .iter()
                .map(|chunk| chunk.plan.num_entries())
                .sum::<usize>();
        let (done, done_rx) = std::sync::mpsc::channel();

        // SAFETY: `output` borrows the caller's buffer for this call. The loop below
        // receives one message per task in `total` before returning, and every spawned
        // task sends exactly one, so no task -- and no copy of this slice -- outlives
        // the borrow.
        let output: UnsafeCellSlice<'static, u8> = unsafe { std::mem::transmute(output) };

        let waiting = (0..chunks.len()).collect();
        let dispatch = Arc::new(PlannedDispatch {
            store: self.store.clone(),
            chunks,
            output,
            data_type_size,
            codec_options: *codec_options,
            budget: self.fetch_byte_budget,
            admission: Mutex::new(Admission {
                in_flight: 0,
                waiting,
            }),
            failure: Mutex::new(None),
            fetch_pool: pool,
            decode_pool,
            done,
        });

        // Fills need no fetched data, so they all run while the first reads are
        // still in flight.
        for index in 0..dispatch.chunks.len() {
            let dispatch = dispatch.clone();
            decode_pool.spawn(move || dispatch.fill_absent(index));
        }
        dispatch.admit();

        for _ in 0..total {
            done_rx
                .recv()
                .expect("dispatch holds a sender until every task is counted");
        }

        match dispatch.failure.lock().unwrap().take() {
            Some(error) => Err(error),
            None => Ok(handled),
        }
    }
}

/// One planned chunk, owned so its tasks can outlive the borrow of the batch.
struct PlannedChunk {
    key: StoreKey,
    subset: ArraySubset,
    array_shape: Vec<std::num::NonZeroU64>,
    plan: DataPlan,
}

/// How much encoded data is in flight, and the chunks waiting on room.
///
/// Reads are issued for as many chunks as fit in a byte budget, and a new chunk
/// starts only as earlier reads retire. Bounding bytes rather than chunks is
/// what makes the limit mean something: a chunk is worth whatever its selection
/// covers, which varies by orders of magnitude across one batch.
struct Admission {
    in_flight: u64,
    waiting: std::collections::VecDeque<usize>,
}

/// Everything a spawned fetch or decode task needs, shared by `Arc`.
///
/// A fetch thread reads, hands the bytes straight to the decode pool, and returns
/// to reading -- no coordinator thread buffers arrivals in between. The byte budget
/// is released as each entry's *decode* completes, which is when the fetched buffer
/// is actually dropped; releasing at arrival would admit new reads while the old
/// bytes are still held by decodes.
struct PlannedDispatch {
    store: ReadableWritableListableStorage,
    chunks: Vec<PlannedChunk>,
    /// See the SAFETY note where this is made `'static`: no task outlives the call.
    output: UnsafeCellSlice<'static, u8>,
    data_type_size: usize,
    codec_options: CodecOptions,
    budget: u64,
    admission: Mutex<Admission>,
    failure: Mutex<Option<PyErr>>,
    fetch_pool: &'static rayon::ThreadPool,
    decode_pool: &'static rayon::ThreadPool,
    /// One `()` per completed task; the dispatching call counts them to know when
    /// every borrow is dead.
    done: std::sync::mpsc::Sender<()>,
}

impl PlannedDispatch {
    fn record_failure(&self, error: PyErr) {
        self.failure.lock().unwrap().get_or_insert(error);
    }

    /// Issue reads for every waiting chunk that fits in the byte budget.
    ///
    /// Finishing beats starting, so the queue drains in order: a chunk already
    /// reading holds memory that only its own decodes release; a new one can only
    /// add to the peak. One chunk is always admitted, so a chunk bigger than the
    /// whole budget still runs.
    fn admit(self: &Arc<Self>) {
        let mut admission = self.admission.lock().unwrap();
        while let Some(&index) = admission.waiting.front() {
            let bytes = plan_bytes(&self.chunks[index].plan);
            if admission.in_flight > 0 && admission.in_flight + bytes > self.budget {
                break;
            }
            admission.waiting.pop_front();
            admission.in_flight += bytes;
            for (entry, byte_range) in self.chunks[index].plan.reads() {
                let dispatch = self.clone();
                self.fetch_pool
                    .spawn(move || dispatch.fetch_entry(index, entry, byte_range));
            }
        }
    }

    /// Retire one read's bytes and admit whatever now fits.
    fn release(self: &Arc<Self>, byte_range: ByteRange) {
        self.admission.lock().unwrap().in_flight -= range_bytes(byte_range);
        self.admit();
    }

    /// Fetch-pool side of one entry: read it, hand it to the decode pool, return.
    fn fetch_entry(self: Arc<Self>, index: usize, entry: usize, byte_range: ByteRange) {
        match self.store.get_partial(&self.chunks[index].key, byte_range) {
            Ok(bytes) => {
                let dispatch = self.clone();
                self.decode_pool
                    .spawn(move || dispatch.decode_entry(index, entry, bytes, byte_range));
            }
            Err(error) => {
                // This entry will never decode, so its accounting retires here.
                self.record_failure(PyRuntimeError::new_err(error.to_string()));
                self.release(byte_range);
                let _ = self.done.send(());
            }
        }
    }

    /// Decode-pool side of one entry, the moment its read landed.
    fn decode_entry(self: Arc<Self>, index: usize, entry: usize, bytes: MaybeBytes, byte_range: ByteRange) {
        // A batch that has already failed returns its first error; decoding more of
        // it would be work nobody reads.
        if self.failure.lock().unwrap().is_none() {
            let chunk = &self.chunks[index];
            // Each task builds its own view: distinct entries write disjoint
            // subdivisions, so concurrent views of the same buffer are the contract.
            let view = unsafe {
                // SAFETY: chunks represent disjoint array subsets
                ArrayBytesFixedDisjointView::new(
                    self.output,
                    self.data_type_size,
                    bytemuck::must_cast_slice(&chunk.array_shape),
                    chunk.subset.clone(),
                )
            }
            .map_py_err::<PyRuntimeError>();
            let result = match view {
                Ok(mut view) => chunk
                    .plan
                    .decode_entry_into(
                        entry,
                        bytes,
                        ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                        &self.codec_options,
                    )
                    .map_codec_err(),
                Err(error) => Err(error),
            };
            if let Err(error) = result {
                self.record_failure(error);
            }
        }
        // The fetched bytes drop with this call, so the budget frees now -- not at
        // arrival, when the decode still holds them.
        self.release(byte_range);
        let _ = self.done.send(());
    }

    /// Fill one chunk's absent units. Needs no fetched data, so it runs before any
    /// read returns.
    fn fill_absent(self: Arc<Self>, index: usize) {
        if self.failure.lock().unwrap().is_none() {
            let chunk = &self.chunks[index];
            let view = unsafe {
                // SAFETY: chunks represent disjoint array subsets
                ArrayBytesFixedDisjointView::new(
                    self.output,
                    self.data_type_size,
                    bytemuck::must_cast_slice(&chunk.array_shape),
                    chunk.subset.clone(),
                )
            }
            .map_py_err::<PyRuntimeError>();
            let result = match view {
                Ok(mut view) => chunk
                    .plan
                    .fill_absent_into(
                        ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                        &self.codec_options,
                    )
                    .map_codec_err(),
                Err(error) => Err(error),
            };
            if let Err(error) = result {
                self.record_failure(error);
            }
        }
        let _ = self.done.send(());
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CodecPipelineImpl {
    #[pyo3(signature = (
        array_metadata,
        store_config,
        *,
        validate_checksums=false,
        chunk_concurrent_minimum=None,
        chunk_concurrent_maximum=None,
        num_threads=None,
        direct_io=false,
        file_handle_cache_size=0,
        fetch_threads=None,
        fetch_byte_budget=None,
        partial_decoder_cache=true,
    ))]
    #[new]
    #[expect(clippy::too_many_arguments)]
    fn new(
        array_metadata: &str,
        mut store_config: StoreConfig,
        validate_checksums: bool,
        chunk_concurrent_minimum: Option<usize>,
        chunk_concurrent_maximum: Option<usize>,
        num_threads: Option<usize>,
        direct_io: bool,
        file_handle_cache_size: usize,
        fetch_threads: Option<usize>,
        fetch_byte_budget: Option<u64>,
        partial_decoder_cache: bool,
    ) -> PyResult<Self> {
        store_config.direct_io(direct_io);
        store_config.file_handle_cache_size(file_handle_cache_size);
        let metadata = serde_json::from_str(array_metadata).map_py_err::<PyTypeError>()?;
        let metadata_v3 = match &metadata {
            ArrayMetadata::V2(v2) => {
                Cow::Owned(array_metadata_v2_to_v3(v2).map_py_err::<PyTypeError>()?)
            }
            ArrayMetadata::V3(v3) => Cow::Borrowed(v3),
        };
        let codec_options = CodecOptions::default().with_validate_checksums(validate_checksums);

        let chunk_concurrent_minimum =
            chunk_concurrent_minimum.unwrap_or(global_config().chunk_concurrent_minimum());
        let chunk_concurrent_maximum =
            chunk_concurrent_maximum.unwrap_or(rayon::current_num_threads());
        let num_threads = num_threads.unwrap_or(rayon::current_num_threads());

        let store: ReadableWritableListableStorage =
            (&store_config).try_into().map_py_err::<PyTypeError>()?;

        let data_type =
            DataType::from_metadata(&metadata_v3.data_type).map_py_err::<PyTypeError>()?;
        let fill_value = data_type
            .fill_value(&metadata_v3.fill_value, ZarrVersion::V3)
            .or_else(|_| {
                Err(match &metadata {
                    ArrayMetadata::V2(metadata) => format!(
                        "incompatible fill value metadata: dtype={}, fill_value={}",
                        metadata.dtype, metadata.fill_value
                    ),
                    ArrayMetadata::V3(metadata) => format!(
                        "incompatible fill value metadata: data_type={}, fill_value={}",
                        metadata.data_type, metadata.fill_value
                    ),
                })
            })
            .map_py_err::<PyTypeError>()?;

        // The codec chain carries its data type and fill value, so every
        // decode and encode below takes them from the chain rather than
        // passing them through at each call.
        let codec_chain = CodecChain::from_metadata(&metadata_v3.codecs)
            .map_py_err::<PyTypeError>()?
            .with_context(data_type.clone(), fill_value.clone())
            .map_py_err::<PyTypeError>()?;

        Ok(Self {
            store,
            codec_chain,
            codec_options,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
            fill_value,
            data_type,
            fetch_byte_budget: fetch_byte_budget
                .filter(|bytes| *bytes > 0)
                .unwrap_or(FETCH_BYTE_BUDGET_DEFAULT),
            fetch_threads: fetch_threads.unwrap_or(FETCH_THREADS_DEFAULT),
            decoder_cache: partial_decoder_cache.then(|| Mutex::new(HashMap::new())),
        })
    }

    fn retrieve_chunks_and_apply_index(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::ChunkItem>, // FIXME: Ref / iterable?
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        // Get input array
        let output = Self::nparray_to_unsafe_cell_slice(value)?;

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(());
        };

        // Assemble partial decoders ahead of time and in parallel
        let partial_chunk_items = chunk_descriptions
            .iter()
            .filter(|item| !(is_whole_chunk(item)))
            .unique_by(|item| item.key.clone())
            .collect::<Vec<_>>();
        let mut partial_decoder_cache: HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>> =
            HashMap::new();
        if !partial_chunk_items.is_empty() {
            // Building a decoder reads that shard's index. Without keeping them
            // between calls every batch re-reads every index it touches, which
            // for a minibatch loader over a fixed array is the same bytes over
            // and over.
            let missing = match &self.decoder_cache {
                Some(cache) => {
                    let cache = cache.lock().unwrap();
                    partial_chunk_items
                        .iter()
                        .filter(|item| match cache.get(&item.key) {
                            Some(decoder) => {
                                partial_decoder_cache.insert(item.key.clone(), decoder.clone());
                                false
                            }
                            None => true,
                        })
                        .copied()
                        .collect::<Vec<_>>()
                }
                None => partial_chunk_items,
            };

            if !missing.is_empty() {
                let key_decoder_pairs =
                    iter_concurrent_limit!(chunk_concurrent_limit, missing, map, |item| {
                        let storage_handle = Arc::new(StorageHandle::new(self.store.clone()));
                        let input_handle = Arc::new((storage_handle, item.key.clone()));
                        let partial_decoder = self
                            .codec_chain
                            .clone()
                            .partial_decoder(input_handle, &item.shape, &codec_options)
                            .map_codec_err()?;
                        Ok((item.key.clone(), partial_decoder))
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                if let Some(cache) = &self.decoder_cache {
                    cache
                        .lock()
                        .unwrap()
                        .extend(key_decoder_pairs.iter().cloned());
                }
                partial_decoder_cache.extend(key_decoder_pairs);
            }
        }

        py.detach(move || {
            // With a fetch pool configured, chunks that can report their reads
            // are fetched together and decoded here; the rest fall through to
            // the loop below untouched.
            let handled = match fetch_pool(self.fetch_threads) {
                Some(pool) => self.decode_planned_chunks(
                    &chunk_descriptions,
                    &partial_decoder_cache,
                    output,
                    pool,
                    &codec_options,
                )?,
                None => vec![false; chunk_descriptions.len()],
            };

            // FIXME: the `decode_into` methods only support fixed length data types.
            // For variable length data types, need a codepath with non `_into` methods.
            // Collect all the subsets and copy into value on the Python side?
            let update_chunk_subset = |item: ChunkItem| {
                let mut output_view = unsafe {
                    // TODO: Is the following correct?
                    //       can we guarantee that when this function is called from Python with arbitrary arguments?
                    // SAFETY: chunks represent disjoint array subsets
                    ArrayBytesFixedDisjointView::new(
                        output,
                        // TODO: why is data_type in `item`, it should be derived from `output`, no?
                        self.data_type
                            .fixed_size()
                            .ok_or("variable length data type not supported")
                            .map_py_err::<PyTypeError>()?,
                        bytemuck::must_cast_slice(&item.array_shape),
                        item.subset.clone(),
                    )
                    .map_py_err::<PyRuntimeError>()?
                };
                let target = ArrayBytesDecodeIntoTarget::Fixed(&mut output_view);
                // See zarrs::array::Array::retrieve_chunk_subset_into
                if is_whole_chunk(&item) {
                    // See zarrs::array::Array::retrieve_chunk_into
                    if let Some(chunk_encoded) =
                        self.store.get(&item.key).map_py_err::<PyRuntimeError>()?
                    {
                        // Decode the encoded data into the output buffer
                        let chunk_encoded: Vec<u8> = chunk_encoded.into();
                        self.codec_chain.decode_into(
                            Cow::Owned(chunk_encoded),
                            &item.shape,
                            target,
                            &codec_options,
                        )
                    } else {
                        // The chunk is missing, write the fill value
                        copy_fill_value_into(&self.data_type, &self.fill_value, target)
                    }
                } else {
                    let key = &item.key;
                    let partial_decoder = partial_decoder_cache.get(key).ok_or_else(|| {
                        PyRuntimeError::new_err(format!("Partial decoder not found for key: {key}"))
                    })?;
                    partial_decoder.partial_decode_into(&item.chunk_subset, target, &codec_options)
                }
                .map_codec_err()
            };

            let remaining = chunk_descriptions
                .into_iter()
                .zip(handled)
                .filter_map(|(item, handled)| (!handled).then_some(item))
                .collect::<Vec<_>>();

            iter_concurrent_limit!(
                chunk_concurrent_limit,
                remaining,
                try_for_each,
                update_chunk_subset
            )?;

            Ok(())
        })
    }

    fn store_chunks_with_indices(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::ChunkItem>,
        value: &Bound<'_, PyUntypedArray>,
        write_empty_chunks: bool,
    ) -> PyResult<()> {
        // A cached decoder holds a shard index that this write invalidates, so
        // drop every shard it touches before touching it.
        if let Some(cache) = &self.decoder_cache {
            let mut cache = cache.lock().unwrap();
            for item in &chunk_descriptions {
                cache.remove(&item.key);
            }
        }

        enum InputValue<'a> {
            Array(ArrayBytes<'a>),
            Constant(FillValue),
        }

        // Get input array
        let input_slice = Self::nparray_to_slice(value)?;
        let input = if value.ndim() > 0 {
            // FIXME: Handle variable length data types, convert value to bytes and offsets
            InputValue::Array(ArrayBytes::new_flen(Cow::Borrowed(input_slice)))
        } else {
            InputValue::Constant(FillValue::new(input_slice.to_vec()))
        };

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, mut codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(());
        };
        codec_options.set_store_empty_chunks(write_empty_chunks);

        py.detach(move || {
            let store_chunk = |item: ChunkItem| match &input {
                InputValue::Array(input) => {
                    let chunk_subset_bytes = input
                        .extract_array_subset(
                            &item.subset,
                            bytemuck::must_cast_slice(&item.array_shape),
                            &self.data_type,
                        )
                        .map_codec_err()?;
                    self.store_chunk_subset_bytes(
                        &item,
                        &self.codec_chain,
                        chunk_subset_bytes,
                        &codec_options,
                    )
                }
                InputValue::Constant(constant_value) => {
                    let chunk_subset_bytes = ArrayBytes::new_fill_value(
                        &self.data_type,
                        item.chunk_subset.num_elements(),
                        constant_value,
                    )
                    .map_py_err::<PyRuntimeError>()?;

                    self.store_chunk_subset_bytes(
                        &item,
                        &self.codec_chain,
                        chunk_subset_bytes,
                        &codec_options,
                    )
                }
            };

            iter_concurrent_limit!(
                chunk_concurrent_limit,
                chunk_descriptions,
                try_for_each,
                store_chunk
            )?;

            Ok(())
        })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<CodecPipelineImpl>()?;
    m.add_class::<chunk_item::ChunkItem>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
