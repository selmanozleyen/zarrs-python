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
    ArrayBytes, ArrayMetadata, ArrayToBytesCodecTraits, CodecChain,
    CodecChainBound, CodecOptions, DataType, FillValue, update_array_bytes,
};
use zarrs::config::global_config;
use zarrs::convert::array_metadata_v2_to_v3;
use zarrs::plugin::ZarrVersion;
use zarrs::storage::{ReadableWritableListableStorage, StoreKey};

mod chunk_item;
mod concurrency;
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

// TODO: Use a OnceLock for store with get_or_try_init when stabilised?
/// Is an innermost chunk a plain byte tiling of its elements?
///
/// True only when the sharding codec's inner chain is exactly `bytes`: no filter, no
/// compressor, nothing between an element and its bytes. Then a chunk's bytes are its
/// elements in C order, the offset of any row inside it is arithmetic, and a row can be read
/// WITHOUT reading the chunk around it.
///
/// Read off the metadata JSON the pipeline is constructed from, rather than taken as another
/// constructor flag -- that argument list already carries three bools and nine parameters.
///
/// Conservative by construction: anything unrecognised, unparsable or nested returns false
/// and the read takes the ordinary chunk path, which is always correct.
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
        let names: Vec<&str> = inner
            .iter()
            .filter_map(|c| c.get("name").and_then(|n| n.as_str()))
            .collect();
        // Exactly the byte reinterpretation. A `crc32c` or a `blosc` here means the chunk
        // cannot be entered part-way, which is the whole reason the chunk is the read unit.
        return names == ["bytes"];
    }
    false
}

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
    /// Present only for a singly-sharded array: the concurrent path locates chunks itself,
    /// so it needs the shard's index codecs and the codecs inside a shard. `None` means this
    /// array cannot take that path at all.
    pub(crate) shard: Option<Arc<shard_index::ShardInfo>>,
    /// Shard indexes read so far, for the life of the array. Reading one is a full-latency
    /// round trip on the CALLING thread, so keeping the decoder costs a shard once per array
    /// rather than once per call; a shard that does not exist is remembered too.
    ///
    /// Only for a READ-ONLY store: a write through this pipeline would move the bytes a
    /// remembered range addresses. An external writer still can, and no cache here sees it.
    pub(crate) shard_indexes: Mutex<HashMap<StoreKey, Arc<ShardingPartialDecoder>>>,
    /// The same, for levels BELOW the outermost, keyed by the path of subchunk indices that
    /// reaches them. Empty and untouched unless the array is nested-sharded, which keeps the
    /// single-level path free of the key allocation this needs.
    pub(crate) subshard_indexes: Mutex<HashMap<(StoreKey, Vec<u64>), Arc<ShardingPartialDecoder>>>,
    /// Whether to remember shard indexes at all: true only for a read-only store.
    pub(crate) cache_shard_indexes: bool,
    /// Whether zarr-python opened this store read-only. Not inferable here: `StoreConfig`
    /// builds a writable Rust store whatever mode the array was opened in.
    pub(crate) store_is_read_only: bool,
    /// Whether an innermost chunk is a plain byte tiling of its elements -- no filter, no
    /// compressor. Read off the array metadata this pipeline was built from.
    ///
    /// When it is, a row's bytes are addressable arithmetically and can be read WITHOUT
    /// reading the chunk around them. Measured at scale: 8,192 rows as exact ranges take
    /// 628 ms against 1121 for the chunks holding them, because the request COUNT is the
    /// same either way and only the bytes differ.
    pub(crate) inner_chunk_is_raw: bool,
}

impl CodecPipelineImpl {
    /// The array's element size in bytes; errors for a variable-length data type.
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

    /// The buffer's pointer and length in bytes, shared by the three `nparray_to_*`
    /// functions. The pointer is returned unread; each caller dereferences it in its own
    /// `unsafe` block, where its safety argument belongs.
    ///
    /// The length is sized in NUMPY's element size while both read paths stride the buffer in
    /// ZARR's, so the two are compared here: a mismatch scales every offset wrongly and still
    /// lands in bounds, which is silently wrong data rather than an error.
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

    fn retrieve_items_and_apply_index(
        &self,
        py: Python,
        chunk_descriptions: &[chunk_item::ChunkItem],
        value: &Bound<'_, PyUntypedArray>,
        ceilings: read_decode::Ceilings,
    ) -> PyResult<()> {
        // Every item must be a chunk-unit item and the array must present a sharding codec.
        // Both are guaranteed by `chunk_info_for_read`, which is the only route to a
        // `ChunkItems` handle; anything it cannot describe raises and falls back to
        // zarr-python in Python. The check stays because this is a `#[pymethods]` boundary,
        // and what it guards is an exclusive output slice.
        if let (true, Some(shard)) = (
            !chunk_descriptions.is_empty() && chunk_descriptions.iter().all(|i| i.coords.is_some()),
            self.shard.as_ref(),
        ) {
            // Confined to this block so no live `&mut` exists when the fallback below takes
            // its own view of the same array. The borrow checker will not catch that -- the
            // `&mut` comes from a raw pointer -- so the block is a deliberate lexical
            // guarantee rather than relying on the fallback staying unreachable.
            let element_size = self.element_size()?;
            let declined = {
                // An aliasing wrapper: no `&mut` is claimed over the whole buffer.
                // `DisjointBytes` vends the pieces from it, one range each, so each piece
                // becomes a `&mut [u8]` the compiler can check.
                let output = Self::nparray_to_unsafe_cell_slice(value, element_size)?;
                let output_len = output.len();
                py.detach(|| {
                    let Some((_, codec_options)) =
                        chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
                    else {
                        return Ok(Vec::new());
                    };
                    self.retrieve_chunk_units(
                        shard,
                        chunk_descriptions,
                        output,
                        output_len,
                        ceilings,
                        &codec_options,
                    )
                })?
            };
            if declined.is_empty() {
                return Ok(());
            }
            // Whatever that path could not take still has to be read, down the fused one.
            let declined: Vec<chunk_item::ChunkItem> = declined.into_iter().cloned().collect();
            // Unreachable: a read item always carries coordinates, because
            // `chunk_info_for_read` only produces a `ChunkItems` handle and every route into
            // that sets them. This was a hand-off to a second Rust path that no longer
            // exists, so if it fires the invariant broke and silence is the wrong answer.
            return Err(PyRuntimeError::new_err(format!(
                "{} read items arrived without coordinates; there is no fallback path and \
                 the caller should have declined to zarr-python instead",
                declined.len()
            )));
        }
        // No second path to fall through to. Reaching here means either the batch was
        // empty, or an item arrived without coordinates, or the array does not present a
        // sharding codec -- and in every one of those cases the output buffer is exactly as
        // `np.empty` left it, so returning Ok would hand the caller uninitialised memory and
        // call it data. Python declines all three before they get here; this is the assertion
        // that says so, rather than a silence that looks like success.
        Err(PyRuntimeError::new_err(format!(
            "a batch of {} items could not be served: {}",
            chunk_descriptions.len(),
            if chunk_descriptions.is_empty() {
                "it is empty"
            } else if self.shard.is_none() {
                "this array presents no sharding codec, so there is no decode unit to group by"
            } else {
                "an item arrived without coordinates"
            }
        )))
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
        store_is_read_only=false,
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
        store_is_read_only: bool,
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
        // Parsed here, as before, so an array with both bad codec metadata and a bad fill
        // value still reports the codec. Only the BINDING has to wait for the data type.
        let codec_chain =
            CodecChain::from_metadata(&metadata_v3.codecs).map_py_err::<PyTypeError>()?;
        let codec_options = CodecOptions::default().with_validate_checksums(validate_checksums);

        let chunk_concurrent_minimum =
            chunk_concurrent_minimum.unwrap_or(global_config().chunk_concurrent_minimum());
        let chunk_concurrent_maximum =
            chunk_concurrent_maximum.unwrap_or(rayon::current_num_threads());
        let num_threads = num_threads.unwrap_or(rayon::current_num_threads());

        // Both default to the available parallelism -- more readers than that is defensible
        // (a blocked reader costs no CPU) but not a library's call to make unasked, and a
        // sweep from 16 to 1024 readers measured flat. Set independently, so a caller who
        // wants reads oversubscribed can raise one alone.
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

        // A codec chain is unbound until it is given the data type and fill value it will
        // work on; `decode`, `encode`, `partial_decoder` and `recommended_concurrency` all
        // live on the bound form. Bound once here, because it is the same for every chunk
        // this pipeline touches.
        let codec_chain = codec_chain
            .with_context(data_type.clone(), fill_value.clone())
            .map_py_err::<PyTypeError>()?;
        // Read off the BOUND chain: it already holds the sharding codec with its inner and
        // index chains bound, so nothing has to be re-derived from the metadata.
        let shard = shard_index::ShardInfo::from_codec_chain(&codec_chain).map(Arc::new);

        Ok(Self {
            store,
            codec_chain,
            codec_options,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
            fill_value,
            data_type,
            shard,
            shard_indexes: Mutex::new(HashMap::new()),
            subshard_indexes: Mutex::new(HashMap::new()),
            cache_shard_indexes: store_is_read_only,
            inner_chunk_is_raw: inner_chunk_is_raw(array_metadata),
            store_is_read_only,
        })
    }

    /// The one read entry point.
    ///
    /// Takes a `ChunkItems` handle rather than a `Vec<ChunkItem>`: the vector costs one
    /// pyclass allocation per item on the way out of the builder and one extraction per item
    /// on the way in, where a handle costs one of each per call whatever the selection.
    ///
    /// There was a second entry point until 2026-08-30 -- a partial decoder per chunk over
    /// rayon, for selections this path declined. An audit of the public indexing surface found
    /// nothing reaching it, so it went, and a decline is now a fall back to zarr-python rather
    /// than a slower second Rust path.
    #[pyo3(signature = (chunk_items, value, read_worker_ceiling=None, decode_worker_ceiling=None))]
    fn retrieve_chunk_items_and_apply_index(
        &self,
        py: Python,
        chunk_items: PyRef<'_, chunk_item::ChunkItems>,
        value: &Bound<'_, PyUntypedArray>,
        read_worker_ceiling: Option<usize>,
        decode_worker_ceiling: Option<usize>,
    ) -> PyResult<()> {
        let ceilings = read_decode::Ceilings::new(read_worker_ceiling, decode_worker_ceiling);
        self.retrieve_items_and_apply_index(py, chunk_items.as_slice(), value, ceilings)
    }

    fn store_chunks_with_indices(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::ChunkItem>,
        value: &Bound<'_, PyUntypedArray>,
        write_empty_chunks: bool,
    ) -> PyResult<()> {
        if self.store_is_read_only {
            return Err(PyValueError::new_err(
                "store was opened in read-only mode and does not support writing",
            ));
        }
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
            // The two inputs differ in how the bytes are OBTAINED, not in what is done with
            // them, so the store call is written once below rather than in each arm.
            let store_chunk = |item: ChunkItem| {
                let chunk_subset_bytes = match &input {
                    InputValue::Array(input) => input
                        .extract_array_subset(
                            &item.subset,
                            bytemuck::must_cast_slice(&item.array_shape),
                            &self.data_type,
                        )
                        .map_codec_err()?,
                    InputValue::Constant(constant_value) => ArrayBytes::new_fill_value(
                        &self.data_type,
                        item.chunk_subset.num_elements(),
                        constant_value,
                    )
                    .map_py_err::<PyRuntimeError>()?,
                };
                self.store_chunk_subset_bytes(
                    &item,
                    &self.codec_chain,
                    chunk_subset_bytes,
                    &codec_options,
                )
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

/// `(call_hits, array_hits, builds)` for the shard index cache, since the run began.
///
/// A build is an index actually read and decoded; the two hit counts are the per-call cache
/// and the per-array one, which are separate because the second only engages on a read-only
/// store. Exposed so a test can assert the cache DID something -- correctness and timing both
/// pass a cache that is never consulted.
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

/// `(raw, chunk)` jobs since the run began: rows read as their own byte range, against whole
/// inner chunks read and decoded.
///
/// Exposed so a test can assert the raw path was TAKEN. Correctness cannot: both paths return
/// the same values, so a gate that refuses everything passes every values test.
#[gen_stub_pyfunction]
#[pyfunction]
fn raw_path_stats() -> (u64, u64) {
    use std::sync::atomic::Ordering;
    (
        read_decode::RAW_JOBS.load(Ordering::Relaxed),
        read_decode::CHUNK_JOBS.load(Ordering::Relaxed),
    )
}

/// `(issued, served)`: store calls made, and inner chunks they served, since the run began.
///
/// `jobs / groups` is the mean batch size, and it is the number that says whether batching
/// ENGAGED at all. At a scattered draw over many shards consecutive jobs rarely share a key,
/// so the ratio approaches 1 and the change is a no-op -- which is a different finding from
/// "batching did not pay", and indistinguishable from it without this.
#[gen_stub_pyfunction]
#[pyfunction]
fn read_merge_stats() -> (u64, u64) {
    use std::sync::atomic::Ordering;
    // (issued, served): store CALLS, then inner chunks. The probe's existing hook reads them
    // in that order and prints the ratio.
    (
        read_decode::GROUP_COUNT.load(Ordering::Relaxed),
        read_decode::GROUP_JOBS.load(Ordering::Relaxed),
    )
}

/// Zero the counters, so one test's numbers are its own.
#[gen_stub_pyfunction]
#[pyfunction]
fn reset_shard_index_cache_stats() {
    use std::sync::atomic::Ordering;
    read_decode::INDEX_CALL_HITS.store(0, Ordering::Relaxed);
    read_decode::INDEX_ARRAY_HITS.store(0, Ordering::Relaxed);
    read_decode::INDEX_BUILDS.store(0, Ordering::Relaxed);
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(shard_index_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(reset_shard_index_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(raw_path_stats, m)?)?;
    m.add_function(wrap_pyfunction!(read_merge_stats, m)?)?;
    m.add_class::<CodecPipelineImpl>()?;
    m.add_class::<chunk_item::ChunkItem>()?;
    m.add_class::<chunk_item::ChunkItems>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
