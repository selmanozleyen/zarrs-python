#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::{Arc, LazyLock, Mutex, Weak};
use std::time::Instant;

use chunk_item::ChunkItem;
use itertools::Itertools;
use numpy::npyffi::PyArrayObject;
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pymethods};
use rayon::ThreadPool;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use unsafe_cell_slice::UnsafeCellSlice;
use utils::is_whole_chunk;
use zarrs::array::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayCodecTraits,
    ArrayMetadata, ArrayPartialDecoderTraits, ArraySubset, ArrayToBytesCodecTraits, CodecChain,
    CodecOptions, DataType, FillValue, StoragePartialDecoder, copy_fill_value_into,
    update_array_bytes,
};
use zarrs::config::global_config;
use zarrs::convert::array_metadata_v2_to_v3;
use zarrs::plugin::ZarrVersion;
use zarrs::storage::{ReadableWritableListableStorage, StorageHandle, StoreKey};

mod chunk_item;
mod concurrency;
mod runtime;
mod store;
#[cfg(test)]
mod tests;
mod utils;
mod vindex_stats;

use crate::concurrency::ChunkConcurrentLimitAndCodecOptions;
use crate::store::{StoreConfig, partial_read_max_active, with_io_measurement};
use crate::utils::{PyCodecErrExt, PyErrExt as _};
use crate::vindex_stats::{Phase, VindexStats};

static VINDEX_DECODE_POOLS: LazyLock<Mutex<HashMap<usize, Weak<ThreadPool>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn shared_vindex_decode_pool(num_threads: usize) -> PyResult<Arc<ThreadPool>> {
    let mut pools = VINDEX_DECODE_POOLS.lock().unwrap();
    if let Some(pool) = pools.get(&num_threads).and_then(Weak::upgrade) {
        return Ok(pool);
    }
    let pool = Arc::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(|index| format!("zarrs-decode-{index}"))
            .build()
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?,
    );
    pools.insert(num_threads, Arc::downgrade(&pool));
    Ok(pool)
}

// TODO: Use a OnceLock for store with get_or_try_init when stabilised?
#[gen_stub_pyclass]
#[pyclass]
pub struct CodecPipelineImpl {
    pub(crate) store: ReadableWritableListableStorage,
    /// `store` wrapped so each logical multi-range read is dispatched onto the
    /// shared I/O pool. Used only by the scattered path, which batches a
    /// shard's ranges into one operation; see the constructor.
    pub(crate) vindex_store: ReadableWritableListableStorage,
    pub(crate) codec_chain: Arc<CodecChain>,
    pub(crate) codec_options: CodecOptions,
    pub(crate) chunk_concurrent_minimum: usize,
    pub(crate) chunk_concurrent_maximum: usize,
    pub(crate) num_threads: usize,
    pub(crate) vindex_io_concurrent_target: usize,
    pub(crate) vindex_decode_concurrent_target: usize,
    vindex_decode_pool: Arc<ThreadPool>,
    vindex_shard_index_cache_size: usize,
    partial_decoder_cache: Mutex<HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>>,
    pub(crate) fill_value: FillValue,
    pub(crate) data_type: DataType,
}

impl CodecPipelineImpl {
    fn retrieve_chunk_bytes<'a>(
        &self,
        item: &ChunkItem,
        codec_chain: &CodecChain,
        codec_options: &CodecOptions,
    ) -> PyResult<ArrayBytes<'a>> {
        let value_encoded = self.store.get(&item.key).map_py_err::<PyRuntimeError>()?;
        let value_decoded = if let Some(value_encoded) = value_encoded {
            let value_encoded: Vec<u8> = value_encoded.into(); // zero-copy in this case
            codec_chain
                .decode(
                    value_encoded.into(),
                    &item.shape,
                    &self.data_type,
                    &self.fill_value,
                    codec_options,
                )
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
        codec_chain: &CodecChain,
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
                .encode(
                    value_decoded,
                    &item.shape,
                    &self.data_type,
                    &self.fill_value,
                    codec_options,
                )
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
        codec_chain: &CodecChain,
        chunk_subset_bytes: ArrayBytes,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        if item.chunk_indices.is_some() {
            // The python side passes allow_fragmenting=False on the
            // write path so this should not be reachable; reject
            // explicitly to keep the contract enforced in Rust too.
            return Err(PyErr::new::<PyValueError, _>(
                "writing scattered ChunkItems is not supported".to_string(),
            ));
        }
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

    /// Decode a scattered (1-D vindex / `arr[idx]`) chunk item into `output`.
    ///
    /// The coordinates are grouped into contiguous runs -- annbatch's CSR shape
    /// gives one run per selected row -- then grouped again by the codec's
    /// efficient partial-decode granularity (an inner chunk for sharding).
    ///
    /// One ordinary subset decode is issued per distinct granule. This avoids
    /// both bad earlier extremes: one decode per run could decompress the same
    /// inner chunk repeatedly, while passing all runs as a generic `Indexer`
    /// made zarrs iterate and dispatch once per selected element. Groups share
    /// the item's codec budget; any budget left when there are few groups flows
    /// down into the codec.
    fn decode_scattered_into(
        item: &ChunkItem,
        partial_decoder: &dyn ArrayPartialDecoderTraits,
        output: UnsafeCellSlice<u8>,
        data_type_size: usize,
        decode_granularity: u64,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        let Some(chunk_indices) = item.chunk_indices.as_ref() else {
            return Ok(());
        };
        if chunk_indices.is_empty() {
            return Ok(());
        }
        let out_indices = item.out_indices.as_deref();
        let out_start = item.subset.start()[0];
        let (runs, order) = group_chunk_runs(chunk_indices, out_indices, out_start);
        let (runs, decode_groups) = group_runs_by_decode_granule(runs, decode_granularity);
        let array_shape: &[u64] = bytemuck::must_cast_slice(&item.array_shape);

        // A list of ArraySubsets is a generic zarrs `Indexer`. The sharding
        // decoder handles that by iterating every selected coordinate, which
        // turned this 2.96M-element workload into 2.96M tiny inner-decoder
        // calls. Instead, issue one ordinary ArraySubset per distinct decode
        // granule (an inner chunk for sharding). Each call takes the fast
        // subset path, decompresses the granule once, and copies all runs that
        // hit it from one bounded scratch buffer.
        let (group_concurrent_limit, group_codec_target) =
            split_decode_concurrency(codec_options.concurrent_target(), decode_groups.len());
        let group_codec_options = (*codec_options).with_concurrent_target(group_codec_target);
        let decode_group = |group: DecodeGroup| -> PyResult<()> {
            let decoded = partial_decoder
                .partial_decode(
                    &ArraySubset::new_with_ranges(std::slice::from_ref(&group.chunk)),
                    &group_codec_options,
                )
                .map_codec_err()?;
            let scratch = decoded.into_fixed().map_py_err::<PyTypeError>()?;
            let scratch: &[u8] = scratch.as_ref();

            for run in &runs[group.runs] {
                let len = usize::try_from(run.chunk.end - run.chunk.start)
                    .map_py_err::<PyValueError>()?;
                let nbytes = len * data_type_size;
                let src = usize::try_from(run.chunk.start - group.chunk.start)
                    .map_py_err::<PyValueError>()?
                    * data_type_size;
                if let Some(out_run) = run.out_contiguous.clone() {
                    let mut view = unsafe {
                        // SAFETY: out_run is a disjoint contiguous range within `output`.
                        ArrayBytesFixedDisjointView::new(
                            output,
                            data_type_size,
                            array_shape,
                            ArraySubset::new_with_ranges(&[out_run]),
                        )
                        .map_py_err::<PyRuntimeError>()?
                    };
                    view.copy_from_slice(&scratch[src..src + nbytes])
                        .map_py_err::<PyRuntimeError>()?;
                } else {
                    // Permuted output: place each element individually.
                    for j in 0..len {
                        let k = order[run.start_idx + j];
                        let out_index = match out_indices {
                            Some(oi) => oi[k as usize],
                            None => out_start + u64::from(k),
                        };
                        let pos = usize::try_from(out_index).map_py_err::<PyValueError>()?
                            * data_type_size;
                        let element_src = src + j * data_type_size;
                        unsafe {
                            // SAFETY: each output position occurs exactly once
                            // across all runs.
                            output.index_mut(pos..pos + data_type_size).copy_from_slice(
                                &scratch[element_src..element_src + data_type_size],
                            );
                        }
                    }
                }
            }
            Ok(())
        };
        iter_concurrent_limit!(
            group_concurrent_limit,
            decode_groups,
            try_for_each,
            decode_group
        )?;
        Ok(())
    }

    fn decode_scattered_item(
        &self,
        item: &ChunkItem,
        partial_decoder_cache: &HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>,
        output: UnsafeCellSlice<u8>,
        data_type_size: usize,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        let partial_decoder = partial_decoder_cache.get(&item.key).ok_or_else(|| {
            PyRuntimeError::new_err(format!("Partial decoder not found for key: {}", item.key))
        })?;
        let decode_granularity = self
            .codec_chain
            .partial_decode_granularity(&item.shape)
            .first()
            .ok_or_else(|| PyRuntimeError::new_err("missing 1-D decode granularity"))?
            .get();
        Self::decode_scattered_into(
            item,
            partial_decoder.as_ref(),
            output,
            data_type_size,
            decode_granularity,
            codec_options,
        )
    }

    /// Build one sparse multi-subset task per store key (one sharded chunk).
    ///
    /// Runs stay separate in the task. The zarrs sharding decoder groups them
    /// by inner chunk without expanding them to element coordinates, so each
    /// inner chunk is fetched once and the gaps are never requested.
    fn plan_scattered_decode_tasks(
        &self,
        items: Vec<ChunkItem>,
    ) -> PyResult<Vec<ScatteredDecodeTask>> {
        let mut tasks: Vec<ScatteredDecodeTask> = Vec::new();
        let mut task_indices: HashMap<StoreKey, usize> = HashMap::new();

        for item in items {
            let Some(chunk_indices) = item.chunk_indices.as_deref() else {
                return Err(PyRuntimeError::new_err(
                    "mixed scattered/basic batch cannot use the global decode planner",
                ));
            };
            if chunk_indices.is_empty() {
                continue;
            }
            let granularity = self
                .codec_chain
                .partial_decode_granularity(&item.shape)
                .first()
                .ok_or_else(|| PyRuntimeError::new_err("missing 1-D decode granularity"))?
                .get();
            let out_start = item.subset.start()[0];
            let (runs, order) =
                group_chunk_runs(chunk_indices, item.out_indices.as_deref(), out_start);
            let (runs, _groups) = group_runs_by_decode_granule(runs, granularity);
            let plan = Arc::new(ScatteredItemPlan {
                runs,
                order,
                out_indices: item.out_indices,
                out_start,
                array_shape: item.array_shape,
            });

            let task_index = *task_indices.entry(item.key.clone()).or_insert_with(|| {
                let task_index = tasks.len();
                tasks.push(ScatteredDecodeTask {
                    key: item.key.clone(),
                    subsets: Vec::new(),
                    pieces: Vec::new(),
                });
                task_index
            });
            let task = &mut tasks[task_index];
            for (run_index, run) in plan.runs.iter().enumerate() {
                task.subsets
                    .push(ArraySubset::new_with_ranges(std::slice::from_ref(
                        &run.chunk,
                    )));
                task.pieces.push(ScatteredDecodePiece {
                    plan: plan.clone(),
                    run_index,
                });
            }
        }
        Ok(tasks)
    }

    fn decode_scattered_task(
        task: ScatteredDecodeTask,
        partial_decoder_cache: &HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>,
        output: UnsafeCellSlice<u8>,
        data_type_size: usize,
        codec_options: &CodecOptions,
        stats: Option<&Arc<VindexStats>>,
    ) -> PyResult<()> {
        let partial_decoder = partial_decoder_cache.get(&task.key).ok_or_else(|| {
            PyRuntimeError::new_err(format!("Partial decoder not found for key: {}", task.key))
        })?;
        // Payload reads issued below are attributed to this thread, so the
        // storage wait can be subtracted from the decode time to estimate
        // codec CPU.
        let _scope = stats.map(|stats| vindex_stats::scope(stats, Phase::Payload));
        let decode_started = stats.map(|_| Instant::now());
        let decoded = partial_decoder
            .partial_decode_subsets(&task.subsets, codec_options)
            .map_codec_err()?;
        let scratch = decoded.into_fixed().map_py_err::<PyTypeError>()?;
        let scratch: &[u8] = scratch.as_ref();
        if let (Some(stats), Some(started)) = (stats, decode_started) {
            stats.record_partial_decode(started.elapsed());
        }

        let scatter_started = stats.map(|_| Instant::now());
        let mut scratch_offset = 0;
        for piece in task.pieces {
            let plan = piece.plan;
            let array_shape: &[u64] = bytemuck::must_cast_slice(&plan.array_shape);
            let run = &plan.runs[piece.run_index];
            let len =
                usize::try_from(run.chunk.end - run.chunk.start).map_py_err::<PyValueError>()?;
            let nbytes = len * data_type_size;
            if let Some(out_run) = run.out_contiguous.clone() {
                let mut view = unsafe {
                    // SAFETY: zarr's projections are disjoint in output
                    // space, including across distinct ChunkItems.
                    ArrayBytesFixedDisjointView::new(
                        output,
                        data_type_size,
                        array_shape,
                        ArraySubset::new_with_ranges(&[out_run]),
                    )
                    .map_py_err::<PyRuntimeError>()?
                };
                view.copy_from_slice(&scratch[scratch_offset..scratch_offset + nbytes])
                    .map_py_err::<PyRuntimeError>()?;
            } else {
                for j in 0..len {
                    let k = plan.order[run.start_idx + j];
                    let out_index = match plan.out_indices.as_deref() {
                        Some(indices) => indices[k as usize],
                        None => plan.out_start + u64::from(k),
                    };
                    let pos =
                        usize::try_from(out_index).map_py_err::<PyValueError>()? * data_type_size;
                    let element_src = scratch_offset + j * data_type_size;
                    unsafe {
                        // SAFETY: every projected output position occurs
                        // exactly once across the complete task queue.
                        output
                            .index_mut(pos..pos + data_type_size)
                            .copy_from_slice(&scratch[element_src..element_src + data_type_size]);
                    }
                }
            }
            scratch_offset += nbytes;
        }
        debug_assert_eq!(scratch_offset, scratch.len());
        if let (Some(stats), Some(started)) = (stats, scatter_started) {
            stats.record_scatter(started.elapsed());
            stats
                .scatter_bytes
                .fetch_add(scratch_offset as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn decode_scattered_batch(
        &self,
        items: Vec<ChunkItem>,
        partial_decoder_cache: &HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>,
        output: UnsafeCellSlice<u8>,
        data_type_size: usize,
        codec_options: CodecOptions,
        stats: Option<&Arc<VindexStats>>,
    ) -> PyResult<()> {
        let num_items = items.len();
        let plan_started = stats.map(|_| Instant::now());
        let tasks = self.plan_scattered_decode_tasks(items)?;
        if let (Some(stats), Some(started)) = (stats, plan_started) {
            stats.record_plan(started.elapsed());
            let subsets = tasks.iter().map(|task| task.subsets.len()).sum::<usize>();
            let subsets_max = tasks.iter().map(|task| task.subsets.len()).max().unwrap_or(0);
            stats.chunk_items.fetch_add(num_items as u64, Ordering::Relaxed);
            stats
                .sparse_subsets
                .fetch_add(subsets as u64, Ordering::Relaxed);
            stats
                .shard_tasks
                .fetch_add(tasks.len() as u64, Ordering::Relaxed);
            stats
                .subsets_per_task_max
                .fetch_max(subsets_max as u64, Ordering::Relaxed);
        }
        // Shard tasks run concurrently up to the pool size; the decode target
        // is the total codec-thread budget spread across whatever is running.
        let active_decode_tasks =
            std::cmp::min(self.vindex_io_concurrent_target, tasks.len()).max(1);
        let task_codec_target = std::cmp::max(
            1,
            self.vindex_decode_concurrent_target / active_decode_tasks,
        );
        if stats.is_some() {
            eprintln!(
                "zarrs vindex concurrency: io_target={} decode_target={} active_decode_workers={active_decode_tasks} codec_target_per_shard={task_codec_target}",
                self.vindex_io_concurrent_target, self.vindex_decode_concurrent_target,
            );
        }
        let task_codec_options = codec_options.with_concurrent_target(task_codec_target);
        let execute_started = stats.map(|_| Instant::now());
        self.vindex_decode_pool.install(|| {
            tasks.into_par_iter().try_for_each(|task| {
                Self::decode_scattered_task(
                    task,
                    partial_decoder_cache,
                    output,
                    data_type_size,
                    &task_codec_options,
                    stats,
                )
            })
        })?;
        if let (Some(stats), Some(started)) = (stats, execute_started) {
            stats.record_execute(started.elapsed());
        }
        Ok(())
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
        vindex_io_concurrent_target=None,
        vindex_decode_concurrent_target=None,
        vindex_shard_index_cache_size=0,
        direct_io=false,
        file_handle_cache_size=0,
    ))]
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        array_metadata: &str,
        mut store_config: StoreConfig,
        validate_checksums: bool,
        chunk_concurrent_minimum: Option<usize>,
        chunk_concurrent_maximum: Option<usize>,
        num_threads: Option<usize>,
        vindex_io_concurrent_target: Option<usize>,
        vindex_decode_concurrent_target: Option<usize>,
        vindex_shard_index_cache_size: usize,
        direct_io: bool,
        file_handle_cache_size: usize,
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
        let codec_chain =
            Arc::new(CodecChain::from_metadata(&metadata_v3.codecs).map_py_err::<PyTypeError>()?);
        let codec_options = CodecOptions::default().with_validate_checksums(validate_checksums);

        let chunk_concurrent_minimum =
            chunk_concurrent_minimum.unwrap_or(global_config().chunk_concurrent_minimum());
        let chunk_concurrent_maximum =
            chunk_concurrent_maximum.unwrap_or(rayon::current_num_threads());
        let num_threads = num_threads.unwrap_or(rayon::current_num_threads());
        let vindex_io_concurrent_target = vindex_io_concurrent_target.unwrap_or(num_threads).max(1);
        let vindex_decode_concurrent_target = vindex_decode_concurrent_target
            .unwrap_or(num_threads)
            .max(1);
        // One pool runs each shard task end to end: its own read, then its own
        // decode. Its size is therefore the outstanding-read depth, which is
        // what this latency-bound workload actually rewards, so it is sized
        // from the I/O target rather than the decode target. Threads waiting
        // in `pread` are cheap; codec CPU is a small share of the batch.
        let vindex_decode_pool = shared_vindex_decode_pool(vindex_io_concurrent_target)?;

        let store: ReadableWritableListableStorage =
            (&store_config).try_into().map_py_err::<PyTypeError>()?;
        // Only the scattered path is instrumented; the basic slice path keeps
        // the raw store so it pays nothing for counters it does not report.
        let vindex_store = with_io_measurement(store.clone());

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

        Ok(Self {
            store,
            vindex_store,
            codec_chain,
            codec_options,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
            vindex_io_concurrent_target,
            vindex_decode_concurrent_target,
            vindex_decode_pool,
            vindex_shard_index_cache_size,
            partial_decoder_cache: Mutex::new(HashMap::new()),
            fill_value,
            data_type,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn retrieve_chunks_and_apply_index(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::ChunkItem>, // FIXME: Ref / iterable?
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        // Get input array
        let output = Self::nparray_to_unsafe_cell_slice(value)?;
        py.detach(move || {
            // One stats object per read, shared by every worker it fans out
            // to, so concurrently reading arrays do not pool their counters.
            let stats = vindex_stats::enabled().then(|| Arc::new(VindexStats::default()));

            // Decoder construction reads sharding indexes, so it must be
            // outside the GIL along with payload I/O and codec work.
            let Some((chunk_concurrent_limit, codec_options)) =
                chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
            else {
                return Ok(());
            };

            // Decided up front because it selects the store the partial
            // decoders are built from: only the scattered path amortises the
            // I/O pool's cross-pool handoff.
            let scattered = chunk_descriptions
                .iter()
                .all(|item| item.chunk_indices.is_some());
            let read_store = if scattered {
                &self.vindex_store
            } else {
                &self.store
            };

            let partial_chunk_items = chunk_descriptions
                .iter()
                .filter(|item| !(is_whole_chunk(item)))
                .unique_by(|item| item.key.clone())
                .collect::<Vec<_>>();
            let cache_enabled = self.vindex_shard_index_cache_size > 0;
            let missing_partial_chunk_items = if cache_enabled {
                let cache = self.partial_decoder_cache.lock().unwrap();
                partial_chunk_items
                    .iter()
                    .copied()
                    .filter(|item| !cache.contains_key(&item.key))
                    .collect::<Vec<_>>()
            } else {
                partial_chunk_items.clone()
            };
            let num_cache_misses = missing_partial_chunk_items.len();
            let build_started = stats.as_ref().map(|_| Instant::now());
            let key_decoder_pairs = if num_cache_misses > 0 {
                self.vindex_decode_pool.install(|| {
                    missing_partial_chunk_items
                        .into_par_iter()
                        .map(|item| {
                            // Constructing a sharding partial decoder reads and
                            // decodes the shard index, so this scope isolates
                            // index I/O from payload I/O.
                            let _scope = stats
                                .as_ref()
                                .map(|stats| vindex_stats::scope(stats, Phase::Index));
                            let storage_handle = Arc::new(StorageHandle::new(read_store.clone()));
                            let input_handle =
                                StoragePartialDecoder::new(storage_handle, item.key.clone());
                            let partial_decoder = self
                                .codec_chain
                                .clone()
                                .partial_decoder(
                                    Arc::new(input_handle),
                                    &item.shape,
                                    &self.data_type,
                                    &self.fill_value,
                                    &codec_options,
                                )
                                .map_codec_err()?;
                            Ok((item.key.clone(), partial_decoder))
                        })
                        .collect::<PyResult<Vec<_>>>()
                })?
            } else {
                Vec::new()
            };
            let partial_decoder_cache = if cache_enabled {
                let mut cache = self.partial_decoder_cache.lock().unwrap();
                let mut local_cache = partial_chunk_items
                    .iter()
                    .filter_map(|item| {
                        cache
                            .get(&item.key)
                            .map(|decoder| (item.key.clone(), decoder.clone()))
                    })
                    .collect::<HashMap<_, _>>();
                for (key, decoder) in key_decoder_pairs {
                    local_cache.insert(key.clone(), decoder.clone());
                    if !cache.contains_key(&key)
                        && cache.len() >= self.vindex_shard_index_cache_size
                        && let Some(evicted) = cache.keys().next().cloned()
                    {
                        cache.remove(&evicted);
                    }
                    cache.entry(key).or_insert(decoder);
                }
                debug_assert_eq!(local_cache.len(), partial_chunk_items.len());
                local_cache
            } else {
                key_decoder_pairs.into_iter().collect::<HashMap<_, _>>()
            };
            if let (Some(stats), Some(started)) = (stats.as_ref(), build_started) {
                stats.record_decoder_build(started.elapsed());
                stats.decoder_cache_hits.fetch_add(
                    (partial_chunk_items.len() - num_cache_misses) as u64,
                    Ordering::Relaxed,
                );
                stats
                    .decoder_cache_misses
                    .fetch_add(num_cache_misses as u64, Ordering::Relaxed);
            }

            let data_type_size = self
                .data_type
                .fixed_size()
                .ok_or("variable length data type not supported")
                .map_py_err::<PyTypeError>()?;

            if scattered {
                let result = self.decode_scattered_batch(
                    chunk_descriptions,
                    &partial_decoder_cache,
                    output,
                    data_type_size,
                    codec_options,
                    stats.as_ref(),
                );
                if let Some(stats) = stats.as_ref() {
                    stats.report("scattered");
                    eprintln!(
                        "zarrs vindex process-global: max_active_partial_reads={}",
                        partial_read_max_active(),
                    );
                }
                return result;
            }

            // FIXME: the `decode_into` methods only support fixed length data types.
            // For variable length data types, need a codepath with non `_into` methods.
            // Collect all the subsets and copy into value on the Python side?
            let update_chunk_subset = |item: ChunkItem| -> PyResult<()> {
                // 1-D scattered fast path (vindex / `arr[idx]`).
                if item.chunk_indices.is_some() {
                    return self.decode_scattered_item(
                        &item,
                        &partial_decoder_cache,
                        output,
                        data_type_size,
                        &codec_options,
                    );
                }
                let mut output_view = unsafe {
                    // TODO: Is the following correct?
                    //       can we guarantee that when this function is called from Python with arbitrary arguments?
                    // SAFETY: chunks represent disjoint array subsets
                    ArrayBytesFixedDisjointView::new(
                        output,
                        data_type_size,
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
                            &self.data_type,
                            &self.fill_value,
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

            let result = iter_concurrent_limit!(
                chunk_concurrent_limit,
                chunk_descriptions,
                try_for_each,
                update_chunk_subset
            );
            if let Some(stats) = stats.as_ref() {
                stats.report("basic");
            }
            result?;

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
            self.partial_decoder_cache.lock().unwrap().clear();
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

/// One contiguous run of chunk positions, plus where its elements go.
pub(crate) struct Run {
    /// Contiguous positions within the chunk.
    pub chunk: std::ops::Range<u64>,
    /// Index into the `ChunkItem`'s coordinate arrays at which this run starts,
    /// so the scatter path can look up each element's output position.
    pub start_idx: usize,
    /// `Some(range)` when the output positions are contiguous too, which lets
    /// the run decode straight into the output with no scratch buffer.
    pub out_contiguous: Option<std::ops::Range<u64>>,
}

/// Runs that can be served by one subset decode of a single codec granule.
struct DecodeGroup {
    /// Minimal contiguous span covering the selected runs in this granule.
    chunk: std::ops::Range<u64>,
    /// Range of entries in the split `Run` vector belonging to this granule.
    runs: std::ops::Range<usize>,
}

/// Immutable output-placement information shared by all global tasks that
/// originated from one `ChunkItem`.
struct ScatteredItemPlan {
    runs: Vec<Run>,
    order: Vec<u32>,
    out_indices: Option<Vec<u64>>,
    out_start: u64,
    array_shape: Vec<std::num::NonZeroU64>,
}

/// One sparse subset and its output-placement plan.
struct ScatteredDecodePiece {
    plan: Arc<ScatteredItemPlan>,
    run_index: usize,
}

/// One globally scheduled sparse multi-range read/decode per store key.
struct ScatteredDecodeTask {
    key: StoreKey,
    subsets: Vec<ArraySubset>,
    pieces: Vec<ScatteredDecodePiece>,
}

/// Spend an item's thread budget on independent decode groups first, then
/// return any unspent capacity to the codec inside each group.
fn split_decode_concurrency(thread_budget: usize, num_groups: usize) -> (usize, usize) {
    let group_concurrent_limit = std::cmp::min(thread_budget, num_groups).max(1);
    let codec_target = std::cmp::max(1, thread_budget / group_concurrent_limit);
    (group_concurrent_limit, codec_target)
}

/// Split runs at codec decode-granule boundaries and group them by granule.
///
/// For a sharding codec the granule is an inner chunk. The input runs are
/// sorted by chunk position, so all pieces for a granule remain adjacent. A
/// group decodes the minimal span from its first selected position to its last;
/// compressed codecs still decode the inner chunk once, while gaps are copied
/// only into scratch and never into the caller's output.
fn group_runs_by_decode_granule(runs: Vec<Run>, granule: u64) -> (Vec<Run>, Vec<DecodeGroup>) {
    debug_assert!(granule > 0);
    let mut split = Vec::with_capacity(runs.len());
    for run in runs {
        let mut start = run.chunk.start;
        while start < run.chunk.end {
            let next_boundary = start.saturating_add(granule - start % granule);
            let end = std::cmp::min(run.chunk.end, next_boundary);
            let offset = start - run.chunk.start;
            let len = end - start;
            let out_contiguous = run
                .out_contiguous
                .as_ref()
                .map(|out| out.start + offset..out.start + offset + len);
            split.push(Run {
                chunk: start..end,
                start_idx: run.start_idx + usize::try_from(offset).unwrap(),
                out_contiguous,
            });
            start = end;
        }
    }

    let mut groups = Vec::new();
    let mut group_start = 0;
    while group_start < split.len() {
        let granule_index = split[group_start].chunk.start / granule;
        let mut group_end = group_start + 1;
        while group_end < split.len() && split[group_end].chunk.start / granule == granule_index {
            group_end += 1;
        }
        groups.push(DecodeGroup {
            chunk: split[group_start].chunk.start..split[group_end - 1].chunk.end,
            runs: group_start..group_end,
        });
        group_start = group_end;
    }
    (split, groups)
}

#[cfg(test)]
mod scattered_decode_plan_tests {
    use super::*;

    #[test]
    fn groups_runs_once_per_decode_granule() {
        let runs = vec![
            Run {
                chunk: 10..20,
                start_idx: 0,
                out_contiguous: Some(100..110),
            },
            Run {
                chunk: 30..40,
                start_idx: 10,
                out_contiguous: Some(200..210),
            },
            // This run crosses the 100-element granule boundary.
            Run {
                chunk: 95..115,
                start_idx: 20,
                out_contiguous: Some(300..320),
            },
            Run {
                chunk: 210..220,
                start_idx: 40,
                out_contiguous: None,
            },
        ];

        let (split, groups) = group_runs_by_decode_granule(runs, 100);

        // Five split runs, but only three distinct compressed inner chunks.
        assert_eq!(split.len(), 5);
        assert_eq!(groups.len(), 3);
        assert_eq!(groups[0].chunk, 10..100);
        assert_eq!(groups[0].runs, 0..3);
        assert_eq!(groups[1].chunk, 100..115);
        assert_eq!(groups[1].runs, 3..4);
        assert_eq!(groups[2].chunk, 210..220);
        assert_eq!(groups[2].runs, 4..5);

        // Splitting preserves both source-order and contiguous-output offsets.
        assert_eq!(split[2].start_idx, 20);
        assert_eq!(split[2].out_contiguous, Some(300..305));
        assert_eq!(split[3].start_idx, 25);
        assert_eq!(split[3].out_contiguous, Some(305..320));
    }

    #[test]
    fn empty_plan_has_no_decode_groups() {
        let (runs, groups) = group_runs_by_decode_granule(Vec::new(), 100);
        assert!(runs.is_empty());
        assert!(groups.is_empty());
    }

    #[test]
    fn decode_groups_get_budget_before_the_codec() {
        assert_eq!(split_decode_concurrency(16, 126), (16, 1));
        assert_eq!(split_decode_concurrency(16, 8), (8, 2));
        assert_eq!(split_decode_concurrency(16, 2), (2, 8));
        assert_eq!(split_decode_concurrency(16, 1), (1, 16));
    }
}

/// Group `chunk_indices` into contiguous runs.
///
/// Runs break on chunk-position discontinuity ONLY. An earlier version also
/// broke on output-position discontinuity, which quietly destroyed the grouping
/// whenever the caller's selection was not already sorted: zarr's
/// `CoordinateIndexer` sorts the coordinates and returns a permuted output
/// mapping, so on a shuffled selection almost every element became its own run
/// (measured: 1.88M runs for 2.96M coordinates, i.e. 1.6 elements per run,
/// against the ~1519 expected for CSR rows). The decode call count then tracks
/// element count rather than run count, which is catastrophic.
///
/// Output contiguity is recorded per run instead of forcing a split, so the
/// zero-copy decode is still used whenever it applies.
///
/// For annbatch's CSR shape -- one contiguous element range per selected row --
/// this yields one run per row regardless of the order the rows arrive in.
fn group_chunk_runs(
    chunk_indices: &[u64],
    out_indices: Option<&[u64]>,
    out_start: u64,
) -> (Vec<Run>, Vec<u32>) {
    let n = chunk_indices.len();
    let mut runs: Vec<Run> = Vec::new();
    // zarr's `CoordinateIndexer` groups an unsorted selection by chunk with
    // `np.argsort(chunks_raveled_indices)`, which defaults to an UNSTABLE
    // quicksort. The key has only one distinct value per chunk, so the sort is
    // free to permute arbitrarily within a chunk -- and it does, destroying
    // whatever order the caller had. Measured on a CSR minibatch whose
    // coordinate array contained exactly one contiguous run per row: 256 runs
    // going in, 203,661 as delivered, 256 again after re-sorting here. So sort
    // by chunk position before grouping.
    //
    // A caller that pre-sorts avoids this entirely: zarr then takes its
    // `searchsorted` fast path, leaves `sel_sort` unset, and hands back a plain
    // slice for `out_selection`, which reaches the zero-copy branch below.
    let mut order: Vec<u32> = (0..u32::try_from(n).unwrap_or(u32::MAX)).collect();
    order.sort_unstable_by_key(|&k| chunk_indices[k as usize]);
    if n == 0 {
        return (runs, order);
    }
    let chunk_at = |k: usize| chunk_indices[order[k] as usize];
    let out_at = |k: usize| match out_indices {
        Some(oi) => oi[order[k] as usize],
        None => out_start + u64::from(order[k]),
    };
    // Output positions are contiguous over `[a, b)` iff they never break.
    let push = |start: usize, end: usize, runs: &mut Vec<Run>| {
        let len = (end - start) as u64;
        let cs = chunk_at(start);
        let os = out_at(start);
        let out_contiguous = (start..end)
            .all(|k| out_at(k) == os + (k - start) as u64)
            .then(|| os..os + len);
        runs.push(Run {
            chunk: cs..cs + len,
            start_idx: start,
            out_contiguous,
        });
    };
    let mut run_start = 0usize;
    for k in 1..n {
        if chunk_at(k) != chunk_at(k - 1) + 1 {
            push(run_start, k, &mut runs);
            run_start = k;
        }
    }
    push(run_start, n, &mut runs);
    (runs, order)
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
