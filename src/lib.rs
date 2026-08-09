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
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, CodecChain, CodecChainBound, CodecOptions,
    DataType, FillValue, ReadPlan, copy_fill_value_into, update_array_bytes,
};
use zarrs::config::global_config;
use zarrs::convert::array_metadata_v2_to_v3;
use zarrs::plugin::ZarrVersion;
use zarrs::storage::{MaybeBytes, ReadableWritableListableStorage, StorageHandle, StoreKey};

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

/// Number of threads for the fetch pool. Unset or `0` disables planning, so
/// every chunk reads from inside its own decode as before.
const FETCH_THREADS: &str = "ZARRS_PYTHON_FETCH_THREADS";

/// Threads decoding planned chunks. Unset follows rayon's global pool.
///
/// The pooled path decodes on `rayon::scope`, so without this it inherits
/// whatever rayon sized itself to. That is the right number only if rayon
/// respects the CPU affinity mask -- on a cluster the host may have far more
/// CPUs than the allocation, and sizing to the host oversubscribes every
/// decode.
const DECODE_THREADS: &str = "ZARRS_PYTHON_DECODE_THREADS";

/// The pool planned chunks decode on.
///
/// Decoding is CPU-bound, so this wants to be about the core count -- unlike
/// the fetch pool, which parks in `pread` and is sized by how many reads the
/// storage will carry. Keeping them separate is the point: one is bounded by
/// cores, the other by the storage service.
fn decode_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var(DECODE_THREADS)
            .ok()
            .and_then(|threads| threads.parse::<usize>().ok())
            .filter(|threads| *threads > 0)?;
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("zarrs-decode-{index}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// Keep partial decoders, and therefore shard indexes, between calls.
///
/// Off by default: correctness depends on every write going through this
/// pipeline so the cache can be evicted. Benchmark scaffolding until that is
/// settled and it becomes a `zarr.config` key.
const DECODER_CACHE: &str = "ZARRS_PYTHON_DECODER_CACHE";

fn decoder_cache_enabled() -> bool {
    std::env::var(DECODER_CACHE).is_ok_and(|v| v != "0" && !v.is_empty())
}

/// The pool that planned reads are issued on.
///
/// A fetch thread parks in the storage read and burns no CPU, so this is
/// sized by how many reads the storage will usefully carry at once, not by
/// core count. Separate from the rayon pool that decodes, so a slow store
/// cannot starve decoding and vice versa.
fn fetch_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let threads = std::env::var(FETCH_THREADS)
            .ok()
            .and_then(|threads| threads.parse::<usize>().ok())
            .filter(|threads| *threads > 0)?;
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|index| format!("zarrs-fetch-{index}"))
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
    /// Partial decoders kept between calls, each holding its shard's decoded
    /// index.
    ///
    /// Building one reads that index from storage, so without this a minibatch
    /// loader re-reads every index it touches on every batch -- measured at
    /// ~820 shards and ~11.5 MiB per batch on a real plate, none of it changing
    /// between batches.
    ///
    /// [`None`] unless enabled: a cached decoder holds an index that a write
    /// invalidates, so this is only sound if every write goes through
    /// [`store_chunks_with_indices`], which evicts what it touches.
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
    /// Decode the chunks that can report their reads, fetching through one pool.
    ///
    /// `partial_decode_into` works out which bytes it needs, reads them, and
    /// decodes them in one call, so overlapping several chunks means a thread
    /// per chunk parked on its own read. Planning first lets every chunk's
    /// reads be outstanding at once against a pool sized for storage rather
    /// than for cores, and each chunk decodes the moment its last byte lands.
    ///
    /// Returns which items it handled. Anything it could not plan for -- a
    /// codec that does not report reads, nested sharding -- is left alone and
    /// takes the ordinary path.
    fn decode_planned_chunks(
        &self,
        items: &[ChunkItem],
        decoders: &HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>,
        output: UnsafeCellSlice<u8>,
        pool: &rayon::ThreadPool,
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
                // `as_planned` is None for any decoder that cannot describe its
                // reads, which is every codec but sharding.
                decoders
                    .get(&item.key)
                    .and_then(|decoder| decoder.as_planned())
                    .map(|planned| planned.read_plan(&item.chunk_subset, codec_options))
                    .transpose()
                    .map_codec_err()
                    .map(Option::flatten)
            })
            .collect::<PyResult<Vec<_>>>()?;

        let handled = plans.iter().map(Option::is_some).collect::<Vec<_>>();
        if !handled.iter().any(|planned| *planned) {
            return Ok(handled);
        }

        // Submit every read across every chunk at once. Nothing throttles them
        // here: depth is bounded by the pool, and decoding by the rayon pool
        // the caller is already on.
        let (tx, rx) = std::sync::mpsc::channel();
        let mut outstanding = vec![0usize; items.len()];
        // Chunks with nothing to read at all -- every inner chunk absent, so the
        // whole selection is the fill value. No message will arrive to complete
        // them, so they are decoded alongside the ones that are waiting on reads.
        let mut nothing_to_read = Vec::new();
        for (index, plan) in plans.iter().enumerate() {
            let Some(plan) = plan else { continue };
            outstanding[index] = plan.reads().count();
            if outstanding[index] == 0 {
                nothing_to_read.push(index);
                continue;
            }
            for (entry, byte_range) in plan.reads() {
                let tx = tx.clone();
                let store = self.store.clone();
                let key = items[index].key.clone();
                pool.spawn(move || {
                    let fetched = store.get_partial(&key, byte_range);
                    let _ = tx.send((index, entry, fetched));
                });
            }
        }
        drop(tx);

        // An entry with nothing to read keeps its place and decodes to the fill
        // value, so the vector is sized by entries rather than by reads.
        let mut fetched: Vec<Vec<MaybeBytes>> = plans
            .iter()
            .map(|plan| vec![None; plan.as_ref().map_or(0, ReadPlan::num_entries)])
            .collect();
        let failure: Mutex<Option<PyErr>> = Mutex::new(None);
        let drain = || {
            rayon::scope(|scope| {
                let decode = |index: usize, fetched: &mut Vec<Vec<MaybeBytes>>| {
                    let item = &items[index];
                    let plan = plans[index].as_ref().expect("only planned items get here");
                    let encoded = std::mem::take(&mut fetched[index]);
                    let failure = &failure;
                    scope.spawn(move |_| {
                        if let Err(error) = self.decode_chunk_from_bytes(
                            item,
                            plan,
                            decoders,
                            output,
                            encoded,
                            codec_options,
                        ) {
                            let mut failure = failure.lock().unwrap();
                            failure.get_or_insert(error);
                        }
                    });
                };

                for index in &nothing_to_read {
                    decode(*index, &mut fetched);
                }
                for (index, entry, result) in rx {
                    match result {
                        // Handed on as the store returned it. `Bytes` is a handle, so
                        // this side no longer copies the encoded chunk -- the inner
                        // decoder still copies each range out of it when it decodes.
                        Ok(bytes) => fetched[index][entry] = bytes,
                        Err(error) => {
                            let mut failure = failure.lock().unwrap();
                            failure
                                .get_or_insert_with(|| PyRuntimeError::new_err(error.to_string()));
                            continue;
                        }
                    }
                    outstanding[index] -= 1;
                    if outstanding[index] == 0 {
                        decode(index, &mut fetched);
                    }
                }
            });
        };
        // Decode on the configured pool when there is one, otherwise on
        // whatever rayon sized itself to.
        match decode_pool() {
            Some(pool) => pool.install(drain),
            None => drain(),
        }

        match failure.into_inner().unwrap() {
            Some(error) => Err(error),
            None => Ok(handled),
        }
    }

    /// Decode one chunk from bytes the caller already fetched, into the output.
    fn decode_chunk_from_bytes(
        &self,
        item: &ChunkItem,
        plan: &ReadPlan,
        decoders: &HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>,
        output: UnsafeCellSlice<u8>,
        encoded: Vec<MaybeBytes>,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        let key = &item.key;
        // Only chunks that returned a plan reach here, so the decoder that
        // produced it must still be able to consume the bytes.
        let planned = decoders
            .get(key)
            .and_then(|decoder| decoder.as_planned())
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!(
                    "Planning partial decoder not found for key: {key}"
                ))
            })?;
        let mut output_view = unsafe {
            // SAFETY: chunks represent disjoint array subsets
            ArrayBytesFixedDisjointView::new(
                output,
                self.data_type
                    .fixed_size()
                    .ok_or("variable length data type not supported")
                    .map_py_err::<PyTypeError>()?,
                bytemuck::must_cast_slice(&item.array_shape),
                item.subset.clone(),
            )
            .map_py_err::<PyRuntimeError>()?
        };
        // Into the output directly. Decoding to an owned buffer first would allocate
        // one per chunk and copy it into place, on the path whose whole point is
        // throughput.
        planned
            .partial_decode_from_bytes_into(
                plan,
                encoded,
                ArrayBytesDecodeIntoTarget::Fixed(&mut output_view),
                codec_options,
            )
            .map_codec_err()
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
    ))]
    #[new]
    fn new(
        array_metadata: &str,
        mut store_config: StoreConfig,
        validate_checksums: bool,
        chunk_concurrent_minimum: Option<usize>,
        chunk_concurrent_maximum: Option<usize>,
        num_threads: Option<usize>,
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
            decoder_cache: decoder_cache_enabled().then(|| Mutex::new(HashMap::new())),
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
            let handled = match fetch_pool() {
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
