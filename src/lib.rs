#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

use chunk_item::ChunkItem;
use numpy::npyffi::PyArrayObject;
use numpy::{PyArrayDescrMethods, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;
use pyo3_stub_gen::derive::{gen_stub_pyclass, gen_stub_pyfunction, gen_stub_pymethods};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use unsafe_cell_slice::UnsafeCellSlice;
use utils::is_whole_chunk;
use zarrs::array::codec::array_to_bytes::sharding::ShardingPartialDecoder;
use zarrs::array::{
    ArrayBytes, ArrayMetadata, ArrayToBytesCodecTraits, CodecChain, CodecChainBound, CodecOptions,
    DataType, FillValue, update_array_bytes,
};
use zarrs::config::global_config;
use zarrs::convert::array_metadata_v2_to_v3;
use zarrs::plugin::ZarrVersion;
use zarrs::storage::{ReadableStorage, ReadableWritableListableStorage, StoreKey};

mod chunk_item;
mod concurrency;
mod fork;
mod lustre;
mod read_decode;
mod runtime;
mod shard_index;
mod store;
#[cfg(test)]
mod tests;
mod utils;

use crate::concurrency::ChunkConcurrentLimitAndCodecOptions;
use crate::store::StoreConfig;
use crate::utils::{PyCodecErrExt, PyErrExt as _};

const NATIVE_ENDIAN: &str = if cfg!(target_endian = "little") {
    "little"
} else {
    "big"
};

/// Whether the sharding codec's inner chain is exactly `bytes` in native order, so a row's bytes
/// can be read without decoding the chunk around it.
fn inner_chunk_is_raw(array_metadata_json: &str) -> bool {
    let Ok(meta) = serde_json::from_str::<serde_json::Value>(array_metadata_json) else {
        return false;
    };
    let Some(codecs) = meta.get("codecs").and_then(|c| c.as_array()) else {
        return false;
    };
    for codec in codecs {
        if codec.get("name").and_then(|n| n.as_str()) != Some("sharding_indexed") {
            continue;
        }
        let inner = codec
            .get("configuration")
            .and_then(|c| c.get("codecs"))
            .and_then(|c| c.as_array());
        let Some(inner) = inner else { return false };
        let [only] = inner.as_slice() else {
            return false;
        };
        if only.get("name").and_then(|n| n.as_str()) != Some("bytes") {
            return false;
        }
        // Absent means little, the v3 default.
        let endian = only
            .get("configuration")
            .and_then(|c| c.get("endian"))
            .and_then(serde_json::Value::as_str);
        return endian.unwrap_or("little") == NATIVE_ENDIAN;
    }
    false
}

// TODO: Use a OnceLock for store with get_or_try_init when stabilised?
#[gen_stub_pyclass]
#[pyclass]
pub(crate) struct CodecPipelineImpl {
    /// Read handle, deliberately the READ-ONLY type: `set`/`erase` are not in its interface,
    /// so the only way to reach a write is `writable()`.
    pub(crate) readable_store: ReadableStorage,
    /// The writable handle -- `None` when zarr-python opened the store read-only. Same object
    /// as `readable_store`; the read-only case simply never keeps a writable view of it.
    pub(crate) writable_store: Option<ReadableWritableListableStorage>,
    /// Used by the first read to size the read pool from the storage.
    pub(crate) filesystem_root: Option<String>,
    pub(crate) codec_chain: Arc<CodecChainBound>,
    pub(crate) codec_options: CodecOptions,
    pub(crate) chunk_concurrent_minimum: usize,
    pub(crate) chunk_concurrent_maximum: usize,
    pub(crate) num_threads: usize,
    pub(crate) fill_value: FillValue,
    pub(crate) data_type: DataType,
    /// `None` if the read path does not serve this codec chain.
    pub(crate) shard: Option<Arc<shard_index::ShardInfo>>,
    pub(crate) shard_indexes: Mutex<HashMap<StoreKey, Arc<ShardingPartialDecoder>>>,
    pub(crate) subshard_indexes: Mutex<HashMap<(StoreKey, Vec<u64>), Arc<ShardingPartialDecoder>>>,
    /// Only for a read-only store: a write moves the bytes an index addresses.
    pub(crate) cache_shard_indexes: bool,
    /// Whether any of the chunk-concurrency options was set, which reads ignore.
    pub(crate) chunk_concurrency_asked: bool,
    pub(crate) inner_chunk_is_raw: bool,
}

impl CodecPipelineImpl {
    fn element_size(&self) -> PyResult<usize> {
        self.data_type
            .fixed_size()
            .ok_or("variable length data type not supported")
            .map_py_err::<PyTypeError>()
    }

    fn retrieve_chunk_bytes<'a>(
        &self,
        item: &ChunkItem,
        codec_chain: &CodecChainBound,
        codec_options: &CodecOptions,
    ) -> PyResult<ArrayBytes<'a>> {
        let value_encoded = self
            .readable_store
            .get(&item.key)
            .map_py_err::<PyRuntimeError>()?;
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

    /// The writable store, or zarr-python's own refusal, verbatim.
    fn writable(&self) -> PyResult<&ReadableWritableListableStorage> {
        self.writable_store.as_ref().ok_or_else(|| {
            PyValueError::new_err("store was opened in read-only mode and does not support writing")
        })
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

        let store = self.writable()?;

        if value_decoded.is_fill_value(&self.fill_value) {
            store.erase(&item.key).map_py_err::<PyRuntimeError>()
        } else {
            let value_encoded = codec_chain
                .encode(value_decoded, &item.shape, codec_options)
                .map(Cow::into_owned)
                .map_codec_err()?;

            // Store the encoded chunk
            store
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

    fn nparray_bytes(
        value: &Bound<'_, PyUntypedArray>,
        element_size: usize,
    ) -> Result<(*mut u8, usize), PyErr> {
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let itemsize = value.dtype().itemsize();
        if itemsize != element_size {
            return Err(PyErr::new::<PyValueError, _>(format!(
                "the output array holds {itemsize} bytes per element but the zarr array holds \
                 {element_size}"
            )));
        }
        let array_object: &PyArrayObject = Self::py_untyped_array_to_array_object(value);
        Ok((array_object.data.cast::<u8>(), value.len() * itemsize))
    }

    fn nparray_to_slice<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
        element_size: usize,
    ) -> Result<&'a [u8], PyErr> {
        let (array_data, array_len) = Self::nparray_bytes(value, element_size)?;
        let slice = unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts(array_data, array_len)
        };
        Ok(slice)
    }

    fn nparray_to_unsafe_cell_slice<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
        element_size: usize,
    ) -> Result<UnsafeCellSlice<'a, u8>, PyErr> {
        let (array_data, array_len) = Self::nparray_bytes(value, element_size)?;
        let output = unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts_mut(array_data, array_len)
        };
        Ok(UnsafeCellSlice::new(output))
    }

    fn warn_read_ignores_chunk_concurrency(&self, py: Python<'_>) -> PyResult<()> {
        static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !self.chunk_concurrency_asked
            || SAID.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return Ok(());
        }
        py.import("warnings")?.call_method1(
            "warn",
            ("codec_pipeline.chunk_concurrent_minimum, chunk_concurrent_maximum and \
              threading.max_workers apply to writes only; reads use codec_pipeline.read_workers \
              and decode_workers.",),
        )?;
        Ok(())
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl CodecPipelineImpl {
    /// The inner chunk shape, empty if unsharded, or `None` if the read path does not serve it.
    fn inner_chunk_shape(&self) -> Option<Vec<u64>> {
        let shard = self.shard.as_ref()?;
        Some(
            shard
                .subchunk_shape
                .as_ref()
                .map(|s| s.iter().map(|d| d.get()).collect())
                .unwrap_or_default(),
        )
    }

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
        fork::check()?;
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
            CodecChain::from_metadata(&metadata_v3.codecs).map_py_err::<PyTypeError>()?;
        let codec_options = CodecOptions::default().with_validate_checksums(validate_checksums);

        let chunk_concurrency_asked = chunk_concurrent_minimum.is_some()
            || chunk_concurrent_maximum.is_some()
            || num_threads.is_some();
        let chunk_concurrent_minimum =
            chunk_concurrent_minimum.unwrap_or(global_config().chunk_concurrent_minimum());
        let chunk_concurrent_maximum =
            chunk_concurrent_maximum.unwrap_or(rayon::current_num_threads());
        let num_threads = num_threads.unwrap_or(rayon::current_num_threads());

        let store: ReadableWritableListableStorage =
            (&store_config).try_into().map_py_err::<PyTypeError>()?;
        let writable_store = (!store_config.read_only).then(|| store.clone());
        let filesystem_root = match &store_config.kind {
            store::StoreKind::Filesystem(config) => Some(config.root.clone()),
            _ => None,
        };
        let readable_store: ReadableStorage = store.readable();

        let data_type =
            DataType::from_metadata(&metadata_v3.data_type).map_py_err::<PyTypeError>()?;
        let fill_value = data_type
            .fill_value(&metadata_v3.fill_value, ZarrVersion::V3)
            .map_err(|_| match &metadata {
                ArrayMetadata::V2(metadata) => format!(
                    "incompatible fill value metadata: dtype={}, fill_value={}",
                    metadata.dtype, metadata.fill_value
                ),
                ArrayMetadata::V3(metadata) => format!(
                    "incompatible fill value metadata: data_type={}, fill_value={}",
                    metadata.data_type, metadata.fill_value
                ),
            })
            .map_py_err::<PyTypeError>()?;

        let codec_chain = codec_chain
            .with_context(data_type.clone(), fill_value.clone())
            .map_py_err::<PyTypeError>()?;
        let shard = shard_index::ShardInfo::from_codec_chain(&codec_chain).map(Arc::new);

        Ok(Self {
            readable_store,
            filesystem_root,
            codec_chain,
            codec_options,
            chunk_concurrency_asked,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
            fill_value,
            data_type,
            shard,
            shard_indexes: Mutex::new(HashMap::new()),
            subshard_indexes: Mutex::new(HashMap::new()),
            cache_shard_indexes: writable_store.is_none(),
            writable_store,
            inner_chunk_is_raw: inner_chunk_is_raw(array_metadata),
        })
    }

    #[pyo3(signature = (chunk_items, value, read_workers=None, decode_workers=None, strict=false))]
    fn retrieve_chunk_items_and_apply_index(
        &self,
        py: Python,
        chunk_items: PyRef<'_, chunk_item::ChunkItems>,
        value: &Bound<'_, PyUntypedArray>,
        read_workers: Option<usize>,
        decode_workers: Option<usize>,
        strict: bool,
    ) -> PyResult<()> {
        let items = chunk_items.as_slice();
        // Only the first read of the process sizes the pools, so only it looks at the storage.
        let read_workers = read_workers.filter(|n| *n > 0).or_else(|| {
            read_decode::pool_sizes().0.or_else(|| {
                let (root, item) = (self.filesystem_root.as_ref()?, items.first()?);
                lustre::read_capacity(&std::path::Path::new(root).join(item.key.as_str()))
            })
        });
        let config = read_decode::ReadConfig::from_call(read_workers, decode_workers, strict);
        // Under the GIL: `pools` locks, and the GIL keeps that lock from being held across a fork.
        let pools = read_decode::pools(py, config)?;
        read_decode::check_workers_arrived(py, config, &pools)?;
        self.warn_read_ignores_chunk_concurrency(py)?;
        let shard = self.shard.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("this array's codec chain is not served by the read path")
        })?;
        let output = Self::nparray_to_unsafe_cell_slice(value, self.element_size()?)?;
        let output_len = output.len();
        py.detach(|| self.retrieve_chunk_units(shard, items, output, output_len, &pools))
    }

    fn store_chunks_with_indices(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::ChunkItem>,
        value: &Bound<'_, PyUntypedArray>,
        write_empty_chunks: bool,
    ) -> PyResult<()> {
        fork::check()?;
        // Fail before decoding anything; the write site checks again by construction.
        self.writable()?;

        enum InputValue<'a> {
            Array(ArrayBytes<'a>),
            Constant(FillValue),
        }

        // Get input array
        let input_slice = Self::nparray_to_slice(value, self.element_size()?)?;
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

/// `(call_hits, array_hits, builds)` for the shard index cache.
#[gen_stub_pyfunction]
#[pyfunction]
fn shard_index_cache_stats() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering;
    (
        read_decode::INDEX_CALL_HITS.load(Ordering::Relaxed),
        read_decode::INDEX_ARRAY_HITS.load(Ordering::Relaxed),
        read_decode::INDEX_BUILDS.load(Ordering::Relaxed),
    )
}

/// `(raw, chunk)` read jobs: rows read as their own byte range, and whole inner chunks.
#[gen_stub_pyfunction]
#[pyfunction]
fn raw_path_stats() -> (u64, u64) {
    use std::sync::atomic::Ordering;
    (
        read_decode::RAW_JOBS.load(Ordering::Relaxed),
        read_decode::CHUNK_JOBS.load(Ordering::Relaxed),
    )
}

/// The widths the read and decode pools were built with, `None` before the first read.
#[gen_stub_pyfunction]
#[pyfunction]
fn pool_sizes() -> (Option<usize>, Option<usize>) {
    read_decode::pool_sizes()
}

#[gen_stub_pyfunction]
#[pyfunction]
fn reset_shard_index_cache_stats() {
    use std::sync::atomic::Ordering;
    read_decode::INDEX_CALL_HITS.store(0, Ordering::Relaxed);
    read_decode::INDEX_ARRAY_HITS.store(0, Ordering::Relaxed);
    read_decode::INDEX_BUILDS.store(0, Ordering::Relaxed);
}

/// A Python module implemented in Rust.
#[pymodule]
pub mod _internal {
    #[pymodule_export]
    #[allow(non_upper_case_globals)]
    const __version__: &str = env!("CARGO_PKG_VERSION");
    #[pymodule_export]
    use super::CodecPipelineImpl;
    #[pymodule_export]
    use super::chunk_item::ChunkItem;
    #[pymodule_export]
    use super::chunk_item::ChunkItems;
    #[pymodule_export]
    use super::pool_sizes;
    #[pymodule_export]
    use super::raw_path_stats;
    #[pymodule_export]
    use super::reset_shard_index_cache_stats;
    #[pymodule_export]
    use super::shard_index_cache_stats;
}

define_stub_info_gatherer!(stub_info);
