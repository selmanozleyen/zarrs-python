#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

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
use zarrs::array::codec::array_to_bytes::sharding::ShardingPartialDecoder;
use zarrs::array::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayMetadata,
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, CodecChain, CodecChainBound, CodecOptions,
    DataType, FillValue, copy_fill_value_into, update_array_bytes,
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
use crate::utils::{PyCodecErrExt, PyErrExt as _, gather, key_partial_decoder};

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
    /// How many readers and decoders ONE call of this array's may run at once. A per-array
    /// setting, because the workers belong to the call: nothing here is shared with another
    /// array, and the only process-wide bound is the worker ceiling in `read_decode`.
    pub(crate) read_concurrency: usize,
    pub(crate) decode_concurrency: usize,
    /// Present only for a singly-sharded array: the concurrent path locates chunks itself,
    /// so it needs the shard's index codecs and the codecs inside a shard. `None` means this
    /// array cannot take that path at all.
    pub(crate) shard: Option<Arc<shard_index::ShardInfo>>,
    /// Shard indexes read so far, for the life of this pipeline -- which is the life of the
    /// array, since zarr builds one pipeline per array.
    ///
    /// A `ShardingPartialDecoder` holds one shard's decoded index, and reading that index
    /// is a full-latency round trip on the CALLING thread before any job reaches the reader
    /// pool. Keeping the decoder means a shard is paid for once per array rather than once
    /// per call. A shard that does not exist is remembered too -- its decoder answers every
    /// subchunk as absent, which is also not worth asking twice.
    ///
    /// Only populated when the store is READ-ONLY, which is the only state in which a
    /// remembered range cannot be invalidated behind our back by our own caller. A
    /// read-only store rejects writes, so nothing this process does can move the bytes a
    /// range addresses; an external writer still can, and no cache here can see that.
    pub(crate) shard_indexes: Mutex<HashMap<StoreKey, Arc<ShardingPartialDecoder>>>,
    /// The same, for levels BELOW the outermost, keyed by the path of subchunk indices that
    /// reaches them. Empty and untouched unless the array is nested-sharded, which keeps the
    /// single-level path free of the key allocation this needs.
    pub(crate) subshard_indexes: Mutex<HashMap<(StoreKey, Vec<u64>), Arc<ShardingPartialDecoder>>>,
    /// Whether to remember shard indexes at all: true only for a read-only store.
    pub(crate) cache_shard_indexes: bool,
}

impl CodecPipelineImpl {
    /// The width of one element, or the error every caller was spelling out for itself.
    ///
    /// Everything on the read path indexes buffers by element, so this is asked for at each
    /// entry point; a variable-length dtype has no answer and none of them can proceed.
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

    /// The buffer's pointer and length in bytes, once it is known to be C contiguous.
    ///
    /// The three `nparray_to_*` functions below differ only in what they hand back -- a
    /// shared slice, an exclusive one, or an aliasing wrapper -- and each restated this
    /// prologue. The pointer is returned unread; every caller dereferences it inside its own
    /// `unsafe` block, with its own safety argument, which is where that argument belongs.
    fn nparray_bytes(value: &Bound<'_, PyUntypedArray>) -> Result<(*mut u8, usize), PyErr> {
        if !value.is_c_contiguous() {
            return Err(PyErr::new::<PyValueError, _>(
                "input array must be a C contiguous array".to_string(),
            ));
        }
        let array_object: &PyArrayObject = Self::py_untyped_array_to_array_object(value);
        Ok((
            array_object.data.cast::<u8>(),
            value.len() * value.dtype().itemsize(),
        ))
    }

    fn nparray_to_slice<'a>(value: &'a Bound<'_, PyUntypedArray>) -> Result<&'a [u8], PyErr> {
        let (array_data, array_len) = Self::nparray_bytes(value)?;
        let slice = unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts(array_data, array_len)
        };
        Ok(slice)
    }

    /// The output as one exclusive slice, for the path that sub-splits it safely.
    ///
    /// Same pointer and length as `nparray_to_unsafe_cell_slice`; the difference is what
    /// the caller may do with it. One `&mut [u8]` that `split_at_mut` then divides is
    /// checked by the compiler, where `UnsafeCellSlice` hands out aliasing views whose
    /// disjointness is only asserted.
    ///
    /// # Safety
    ///
    /// Produces a `&mut` from a `&`, so the caller must be the array's only writer for the
    /// lifetime of the slice: no second call to this or to `nparray_to_unsafe_cell_slice`
    /// on the same array, and no Python code running that could write to it. Holding the
    /// GIL for the duration is what makes the second part true.
    // A `&mut` from a `&` is exactly what this does, and the lint is right that it cannot be
    // checked. The numpy buffer is owned by Python and reached through a shared handle, so
    // there is no `&mut Bound` to take; the contract above is what makes it sound, and the
    // one call site holds the GIL and takes no other view.
    #[allow(clippy::mut_from_ref)]
    unsafe fn nparray_to_mut_slice<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
    ) -> Result<&'a mut [u8], PyErr> {
        let (array_data, array_len) = Self::nparray_bytes(value)?;
        Ok(unsafe {
            // SAFETY: array_data is a valid pointer to a u8 array of length array_len, and
            // Python holds no other writer to it for the duration of this call.
            debug_assert!(!array_data.is_null());
            std::slice::from_raw_parts_mut(array_data, array_len)
        })
    }

    fn nparray_to_unsafe_cell_slice<'a>(
        value: &'a Bound<'_, PyUntypedArray>,
    ) -> Result<UnsafeCellSlice<'a, u8>, PyErr> {
        let (array_data, array_len) = Self::nparray_bytes(value)?;
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
    ) -> PyResult<()> {
        // The concurrent path, when every item is a chunk-unit item. It needs an exclusive
        // output slice, so it cannot share the aliasing wrapper the fused path takes --
        // hence the dispatch here rather than inside the loop.
        if let (true, Some(shard)) = (
            !chunk_descriptions.is_empty() && chunk_descriptions.iter().all(|i| i.coords.is_some()),
            self.shard.as_ref(),
        ) {
            // SAFETY: this is the only view taken of `value` in this call, and the GIL is
            // still held here -- `py.detach` is entered below, after the slice exists.
            let output = unsafe { Self::nparray_to_mut_slice(value)? };
            let declined = py.detach(|| {
                let Some((_, codec_options)) =
                    chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
                else {
                    return Ok(Vec::new());
                };
                self.retrieve_chunk_units(shard, chunk_descriptions, output, &codec_options)
            })?;
            if declined.is_empty() {
                return Ok(());
            }
            // Whatever that path could not take still has to be read, down the fused one.
            let declined: Vec<chunk_item::ChunkItem> = declined.into_iter().cloned().collect();
            return self.retrieve_chunks_and_apply_index_fused(py, declined, value);
        }
        // Not the hot path: the batch is mixed, or arrived as a list.
        self.retrieve_chunks_and_apply_index_fused(py, chunk_descriptions.to_vec(), value)
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
        read_concurrency=None,
        decode_concurrency=None,
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
        read_concurrency: Option<usize>,
        decode_concurrency: Option<usize>,
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
        let codec_options = CodecOptions::default().with_validate_checksums(validate_checksums);

        let chunk_concurrent_minimum =
            chunk_concurrent_minimum.unwrap_or(global_config().chunk_concurrent_minimum());
        let chunk_concurrent_maximum =
            chunk_concurrent_maximum.unwrap_or(rayon::current_num_threads());
        let num_threads = num_threads.unwrap_or(rayon::current_num_threads());

        // Both default to the available parallelism, which is what rayon and
        // `available_parallelism` do, and what a library should spend without being asked.
        //
        // Reads are latency-bound and decodes are CPU-bound, so the two are still set
        // INDEPENDENTLY -- a caller who wants reads oversubscribed can raise
        // `read_concurrency` alone. The default used to be `4 * num_threads` readers on the
        // grounds that a blocked reader costs no CPU. True, but it is not a library's call to
        // make: on a machine with no affinity mask (a login node, a bare box, a container
        // without a cpuset) that is 4x every core in threads nobody asked for, and a sweep
        // from 16 to 1024 readers measured FLAT on the strided shape anyway.
        //
        // Above ~items-per-call, more readers do nothing regardless: a call cannot have more
        // reads outstanding than it has chunks.
        let read_concurrency = read_concurrency.unwrap_or(num_threads).max(1);
        let decode_concurrency = decode_concurrency.unwrap_or(num_threads).max(1);
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
        let codec_chain = CodecChain::from_metadata(&metadata_v3.codecs)
            .map_py_err::<PyTypeError>()?
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
            read_concurrency,
            decode_concurrency,
            shard,
            shard_indexes: Mutex::new(HashMap::new()),
            subshard_indexes: Mutex::new(HashMap::new()),
            cache_shard_indexes: store_is_read_only,
        })
    }

    fn retrieve_chunks_and_apply_index(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::ChunkItem>, // FIXME: Ref / iterable?
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        self.retrieve_items_and_apply_index(py, &chunk_descriptions, value)
    }

    /// The same read as `retrieve_chunks_and_apply_index`, from a `ChunkItems` handle.
    ///
    /// A `Vec<ChunkItem>` argument costs one pyclass allocation per item on the way out
    /// of the builder and one extraction per item on the way in here. A handle costs one
    /// of each per call, whatever the selection.
    fn retrieve_chunk_items_and_apply_index(
        &self,
        py: Python,
        chunk_items: PyRef<'_, chunk_item::ChunkItems>,
        value: &Bound<'_, PyUntypedArray>,
    ) -> PyResult<()> {
        self.retrieve_items_and_apply_index(py, chunk_items.as_slice(), value)
    }

    fn retrieve_chunks_and_apply_index_fused(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::ChunkItem>,
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
            let key_decoder_pairs =
                iter_concurrent_limit!(chunk_concurrent_limit, partial_chunk_items, map, |item| {
                    let input_handle = key_partial_decoder(&self.store, &item.key);
                    let partial_decoder = self
                        .codec_chain
                        .clone()
                        .partial_decoder(input_handle, &item.shape, &codec_options)
                        .map_codec_err()?;
                    Ok((item.key.clone(), partial_decoder))
                })
                .collect::<PyResult<Vec<_>>>()?;
            partial_decoder_cache.extend(key_decoder_pairs);
        }

        py.detach(move || {
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
                        self.element_size()?,
                        bytemuck::must_cast_slice(&item.array_shape),
                        item.subset.clone(),
                    )
                    .map_py_err::<PyRuntimeError>()?
                };
                if let Some(coords) = &item.coords {
                    // The chunk is the read and decode unit. `chunk_subset` is exactly one
                    // inner chunk, and an ArraySubset indexer is what takes zarrs'
                    // chunks-in-subset path, where the chunk is fetched and decoded once.
                    //
                    // Handing zarrs the coordinates instead reaches
                    // `partial_decode_fixed_indexer`, which walks them one at a time: two
                    // ArrayIndices allocations, a subchunk-decoder cache lookup and a
                    // `partial_decode` call PER ELEMENT. `partial_decode_into` is not usable
                    // here either -- it requires indexer.len() == output elements, and a
                    // whole chunk is deliberately larger than what is wanted from it.
                    let partial_decoder =
                        cached_partial_decoder(&partial_decoder_cache, &item.key)?;
                    let decoded = partial_decoder
                        .partial_decode(&item.chunk_subset, &codec_options)
                        .map_codec_err()?;
                    let ArrayBytes::Fixed(raw) = decoded else {
                        return Err(PyTypeError::new_err(
                            "variable length data type not supported",
                        ));
                    };
                    let size = self.element_size()?;
                    // The gather zarr-python does with one numpy fancy index, over a buffer
                    // that is already decoded: a load and a store per element. The output
                    // side is contiguous because the indices reached us non-decreasing, so
                    // one chunk's elements are one run of the output.
                    // The view takes one contiguous run, so the elements are gathered into
                    // a buffer first; a job writes into its own slice and skips that.
                    let mut gathered = vec![0u8; coords.len() * size];
                    gather(&raw, coords, &mut gathered, size)
                        .map_err(|e| PyRuntimeError::new_err(format!("{}: {e}", item.key)))?;
                    output_view
                        .copy_from_slice(&gathered)
                        .map_py_err::<PyRuntimeError>()?;
                    return Ok(());
                }
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
                    let partial_decoder =
                        cached_partial_decoder(&partial_decoder_cache, &item.key)?;
                    partial_decoder.partial_decode_into(&item.chunk_subset, target, &codec_options)
                }
                .map_codec_err()
            };

            iter_concurrent_limit!(
                chunk_concurrent_limit,
                chunk_descriptions,
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

/// The partial decoder assembled for this key earlier in the call.
///
/// Looked up twice on the read path -- once by the coordinate gather, once by the ordinary
/// subset decode -- and a miss is a bug in this function rather than a bad input, so both
/// want the same message.
fn cached_partial_decoder<'a>(
    cache: &'a HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>,
    key: &StoreKey,
) -> PyResult<&'a Arc<dyn ArrayPartialDecoderTraits>> {
    cache
        .get(key)
        .ok_or_else(|| PyRuntimeError::new_err(format!("Partial decoder not found for key: {key}")))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<CodecPipelineImpl>()?;
    m.add_class::<chunk_item::ChunkItem>()?;
    m.add_class::<chunk_item::ChunkItems>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
