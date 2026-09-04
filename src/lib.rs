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

use crate::store::StoreConfig;
use crate::utils::{PyCodecErrExt, PyErrExt as _};

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
    /// Present unless this array's codec chain is refused outright: the concurrent path locates
    /// so it needs the shard's index codecs and the codecs inside a shard. `None` means this
    /// array cannot take that path at all.
    pub(crate) shard: Option<Arc<shard_index::ShardInfo>>,
    /// Shard indexes read so far, for the life of the array. Reading one is a full-latency
    /// round trip on the CALLING thread, so keeping the decoder costs a shard once per array
    /// rather than once per call; a shard that does not exist is remembered too.
    ///
    /// Only for a READ-ONLY store: a write through this pipeline would move the bytes a
    /// remembered range addresses. An external writer still can, and no cache here sees it.
    pub(crate) shard_decoders: Mutex<HashMap<StoreKey, Arc<ShardingPartialDecoder>>>,
    /// The same, for levels BELOW the outermost, keyed by the path of subchunk indices that
    /// reaches them. Empty and untouched unless the array is nested-sharded, which keeps the
    /// single-level path free of the key allocation this needs.
    pub(crate) subshard_decoders: Mutex<HashMap<(StoreKey, Vec<u64>), Arc<ShardingPartialDecoder>>>,
    /// Whether to remember shard indexes at all: true only for a read-only store.
    pub(crate) cache_shard_indexes: bool,
    /// Whether zarr-python opened this store read-only. Not inferable here: `StoreConfig`
    /// builds a writable Rust store whatever mode the array was opened in.
    pub(crate) store_is_read_only: bool,
    /// The pool sizes this array was OPENED with.
    ///
    /// Read once, here, exactly as `num_threads` and the chunk-concurrency bounds are. They
    /// size process-wide pools that only the first read builds, so reading them per call
    /// would offer a caller a choice that cannot be honoured.
    pub(crate) read_pool_size: usize,
    pub(crate) decode_pool_size: usize,
    /// Whether a size that cannot be honoured is an ERROR rather than a warning.
    ///
    /// `codec_pipeline.strict` already means "do not paper over something this pipeline
    /// cannot do" -- it turns a decline into a raise instead of a silent fallback. A pool size
    /// the process cannot give is the same kind of thing, so it answers to the same switch.
    pub(crate) strict: bool,
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
        // WRITABLE, which `from_raw_parts_mut` requires and which nothing else established.
        // A read-only target -- a read-only mmap, `arr.flags.writeable = False`, a view over an
        // immutable buffer -- was written through anyway: a segfault, or a silently diverging
        // copy-on-write page, neither of them a Python exception. Contiguity and the element
        // size are both checked in `nparray_bytes`, which is what made this read as an
        // oversight rather than a decision.
        //
        // HERE and not in `nparray_bytes`, because the write path reads its INPUT array through
        // that same helper and a read-only source is perfectly legitimate there.
        let array_object: &PyArrayObject = Self::py_untyped_array_to_array_object(value);
        if array_object.flags & numpy::npyffi::NPY_ARRAY_WRITEABLE == 0 {
            return Err(PyErr::new::<PyValueError, _>(
                "the output array is not writable".to_string(),
            ));
        }
        let (array_data, array_len) = Self::nparray_bytes(value, element_size)?;
        let output = unsafe {
            // SAFETY: `array_data` points at `array_len` bytes of a C-contiguous, writable
            // array, checked above; `UnsafeCellSlice` is what makes the aliasing sound.
            assert!(
                !array_data.is_null(),
                "numpy handed over a null data pointer"
            );
            std::slice::from_raw_parts_mut(array_data, array_len)
        };
        Ok(UnsafeCellSlice::new(output))
    }

    fn retrieve_items_and_apply_index(
        &self,
        py: Python,
        chunk_descriptions: &[chunk_item::ChunkItem],
        value: &Bound<'_, PyUntypedArray>,
        config: read_decode::ReadConfig,
    ) -> PyResult<()> {
        // An empty batch is a read of NOTHING -- `X[[]]` -- and nothing is servable, so this
        // returns rather than falling to the refusal below, which is where it used to land.
        // Python serves it too (`chunk_info_for_read` hands back an empty handle); both halves
        // have to agree or the fix is half a fix.
        //
        // AND THE OUTPUT MUST BE EMPTY TOO. This return skips `retrieve_chunk_units`, and with
        // it the `covered() != output_len` check -- the one that stops `np.empty` contents
        // being handed back as data. An earlier version returned unconditionally, so
        // `push nothing` against a 1,000-element buffer was a silent success full of whatever
        // was in that memory. Nothing Python builds reaches it (every describer refuses an
        // empty index array), but this is a `#[pymethods]` boundary and that is the only
        // reason the hole was ever closed elsewhere.
        if chunk_descriptions.is_empty() {
            if value.len() != 0 {
                return Err(PyRuntimeError::new_err(format!(
                    "an empty batch cannot fill {} output elements; they would be returned \
                     uninitialised",
                    value.len()
                )));
            }
            return Ok(());
        }
        // The codec chain must be one this path accepts; `chunk_info_for_read` is the only route to a
        // `ChunkItems` handle and it refuses anything else, but this is a `#[pymethods]`
        // boundary and what it guards is an exclusive output slice.
        //
        // Whether every item is a CHUNK-UNIT item is no longer checked here. It used to be, and
        // then the batch was handed over anyway and a list of the ones it could not take came
        // back to be turned into an error -- a round trip whose result was always empty.
        // `locate_chunks` raises on the first such item instead, naming its key.
        if let Some(shard) = self.shard.as_ref() {
            // Confined to this block so no live `&mut` exists when the fallback below takes
            // its own view of the same array. The borrow checker will not catch that -- the
            // `&mut` comes from a raw pointer -- so the block is a deliberate lexical
            // guarantee rather than relying on the fallback staying unreachable.
            let element_size = self.element_size()?;
            {
                // An aliasing wrapper: no `&mut` is claimed over the whole buffer.
                // `DisjointBytes` vends the pieces from it, one range each, so each piece
                // becomes a `&mut [u8]` the compiler can check.
                let output = Self::nparray_to_unsafe_cell_slice(value, element_size)?;
                py.detach(|| {
                    // The pipeline's own options, NOT
                    // `get_chunk_concurrent_limit_and_codec_options`. That helper's whole
                    // product is a concurrent target for a chunk-at-a-time codec loop, and
                    // this path has no such loop -- `retrieve_chunk_units` overrides the
                    // target to 1 because parallelism here comes from the two pools, one job
                    // per inner chunk. Calling it would have priced `recommended_concurrency`
                    // on the codec chain once per batch to produce a number thrown away on
                    // the next line.
                    self.retrieve_chunk_units(
                        shard,
                        chunk_descriptions,
                        output,
                        config,
                        &self.codec_options,
                    )
                })?;
            }
            return Ok(());
        }
        // No second path to fall through to. Reaching here means the codec chain was REFUSED
        // -- a codec beside the sharding codec -- and the output buffer is exactly as
        // `np.empty` left it, so
        // returning Ok would hand the caller uninitialised memory and call it data. Python
        // declines this before it gets here (`_inner_chunk_shape` returns `None` for the same
        // chains `ShardInfo::from_codec_chain` refuses); this is the assertion that says so,
        // rather than a silence that looks like success.
        Err(PyRuntimeError::new_err(format!(
            "a batch of {} items could not be served: this array carries a codec beside its \
             sharding codec, so the shard index no longer addresses the bytes on disk",
            chunk_descriptions.len(),
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
        read_pool_size=None,
        decode_pool_size=None,
        strict=false,
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
        read_pool_size: Option<usize>,
        decode_pool_size: Option<usize>,
        strict: bool,
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
            shard_decoders: Mutex::new(HashMap::new()),
            subshard_decoders: Mutex::new(HashMap::new()),
            cache_shard_indexes: store_is_read_only,
            store_is_read_only,
            read_pool_size: read_decode::resolve_pool_size(read_pool_size),
            decode_pool_size: read_decode::resolve_pool_size(decode_pool_size),
            strict,
        })
    }

    /// The innermost unit this array's codec chain decodes, or `None` to refuse the array.
    ///
    /// THREE ANSWERS, and Python needs all three: a shape is the inner chunk of a sharded
    /// array; an EMPTY shape means the array is not sharded, so its chunk is its own decode
    /// unit and only a batch entry knows that shape; `None` means this chain cannot be served
    /// at all -- a codec beside the sharding codec, which leaves the shard index addressing
    /// bytes that are no longer there.
    ///
    // ASKED HERE RATHER THAN DERIVED TWICE. Python used to answer this itself by walking
    // zarr's codec OBJECTS while `ShardInfo::from_codec_chain` answered it from the bound
    // chain, and the two could disagree -- at which point Python built a description this side
    // refuses, and the refusal surfaced as an uncaught `PyRuntimeError` from a call made
    // outside `read`'s `try`. A third-party codec registered for `sharding_indexed` reopens
    // that gap however carefully the Python walk is written. `//` rather than `///`: this is
    // why the function exists, not what it promises, and only the latter belongs in `help()`.
    #[getter]
    fn inner_chunk_shape(&self) -> Option<Vec<u64>> {
        self.shard.as_ref().map(|shard| {
            shard
                .subchunk_shape
                .as_ref()
                .map_or_else(Vec::new, |shape| {
                    shape.iter().map(|extent| extent.get()).collect()
                })
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
    #[pyo3(signature = (chunk_items, value))]
    fn retrieve_chunk_items_and_apply_index(
        &self,
        py: Python,
        chunk_items: PyRef<'_, chunk_item::ChunkItems>,
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        // The pool sizes come from the array, not from this call: they were read when it was
        // opened.
        let config = read_decode::ReadConfig::from_open(self.read_pool_size, self.decode_pool_size);
        // The pools are sized once, by the first read. A size arriving after that cannot
        // be honoured, and a caller who is not told believes it was.
        read_decode::check_pool_size_arrived(py, config, self.strict)?;
        self.retrieve_items_and_apply_index(py, chunk_items.as_slice(), value, config)
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
            concurrency::chunk_concurrency(&chunk_descriptions, self)?
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

/// Drop the two worker pools, so a `fork()` about to happen cannot inherit a held lock.
///
/// Registered from Python with `os.register_at_fork(before=...)`. The per-process check that
/// rebuilds these in a child is not enough on its own: it runs under the very lock that `fork`
/// would copy as held, so a child forked mid-read would block on it for ever.
#[gen_stub_pyfunction]
#[pyfunction]
fn release_pools_for_fork() {
    read_decode::release_pools_for_fork();
    // The tokio runtime has the identical hazard and is reached by any object-store or HTTP
    // backed array. Covering one and not the other is what makes a fork hook a false promise.
    runtime::release_runtime_for_fork();
}

/// The sizes the two worker pools were BUILT with, or `None` where one has not been built.
///
/// Pools are sized by the first read in the process, so a size set later is silently
/// ignored. A benchmark that sets one and reports a number has to be able to say which of the
/// two happened -- the repo's rule that a knob which was set is not a knob that arrived.
#[gen_stub_pyfunction]
#[pyfunction]
fn pool_sizes() -> (Option<usize>, Option<usize>) {
    read_decode::pool_sizes()
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(shard_index_cache_stats, m)?)?;
    m.add_function(wrap_pyfunction!(pool_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(release_pools_for_fork, m)?)?;
    m.add_class::<CodecPipelineImpl>()?;
    m.add_class::<chunk_item::ChunkItem>()?;
    m.add_class::<chunk_item::ChunkItems>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
