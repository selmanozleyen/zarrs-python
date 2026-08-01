#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Arc;

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
    ArrayPartialDecoderTraits, ArraySubset, ArrayToBytesCodecTraits, CodecChain, CodecOptions,
    DataType, FillValue, StoragePartialDecoder, copy_fill_value_into, update_array_bytes,
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

use crate::concurrency::ChunkConcurrentLimitAndCodecOptions;
use crate::store::StoreConfig;
use crate::utils::{PyCodecErrExt, PyErrExt as _};

// TODO: Use a OnceLock for store with get_or_try_init when stabilised?
#[gen_stub_pyclass]
#[pyclass]
pub struct CodecPipelineImpl {
    pub(crate) store: ReadableWritableListableStorage,
    pub(crate) codec_chain: Arc<CodecChain>,
    pub(crate) codec_options: CodecOptions,
    pub(crate) chunk_concurrent_minimum: usize,
    pub(crate) chunk_concurrent_maximum: usize,
    pub(crate) num_threads: usize,
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
    /// gives one run per selected row -- and every run is handed to the decoder
    /// in ONE `partial_decode` call.
    ///
    /// Decoding runs one at a time (previously via rayon) leaves the sharding
    /// decoder nothing to work with: a run is typically a single inner chunk, so
    /// N separate calls each re-plan the read from scratch (shard index lookup,
    /// byte range assembly) and cannot share an inner-chunk decode with one
    /// another. With N large that dominates -- a 1.5k-element run inside a ~97k
    /// element inner chunk decodes the whole inner chunk, once per call.
    /// Measured on a shuffled CSR minibatch, per-run decoding was 18x slower
    /// than the slice path. Handing every subset over at once lets the decoder
    /// plan them together and dedup inner-chunk decodes via its per-call cache.
    ///
    /// The parallelism itself comes from processing several items at once; see
    /// `recommended_outer_concurrency`.
    fn decode_scattered_into(
        item: &ChunkItem,
        partial_decoder: &dyn ArrayPartialDecoderTraits,
        output: UnsafeCellSlice<u8>,
        data_type_size: usize,
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
        let array_shape: &[u64] = bytemuck::must_cast_slice(&item.array_shape);

        let subsets: Vec<ArraySubset> = runs
            .iter()
            .map(|r| ArraySubset::new_with_ranges(std::slice::from_ref(&r.chunk)))
            .collect();
        let decoded = partial_decoder
            .partial_decode(&subsets, codec_options)
            .map_codec_err()?;
        let scratch = decoded.into_fixed().map_py_err::<PyTypeError>()?;
        let scratch: &[u8] = scratch.as_ref();

        // `partial_decode` returns the subsets concatenated in request order,
        // so walk them in lockstep with `runs`.
        let mut offset: usize = 0;
        for run in &runs {
            let len = usize::try_from(run.chunk.end - run.chunk.start).map_py_err::<PyValueError>()?;
            let nbytes = len * data_type_size;
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
                view.copy_from_slice(&scratch[offset..offset + nbytes])
                    .map_py_err::<PyRuntimeError>()?;
            } else {
                // Permuted output: place each element individually.
                for j in 0..len {
                    let k = order[run.start_idx + j];
                    let out_index = match out_indices {
                        Some(oi) => oi[k as usize],
                        None => out_start + u64::from(k),
                    };
                    let pos =
                        usize::try_from(out_index).map_py_err::<PyValueError>()? * data_type_size;
                    let src = offset + j * data_type_size;
                    unsafe {
                        // SAFETY: each output position occurs exactly once
                        // across all runs.
                        output
                            .index_mut(pos..pos + data_type_size)
                            .copy_from_slice(&scratch[src..src + data_type_size]);
                    }
                }
            }
            offset += nbytes;
        }
        debug_assert_eq!(offset, scratch.len());
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
        direct_io=false,
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
    ) -> PyResult<Self> {
        store_config.direct_io(direct_io);
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

        Ok(Self {
            store,
            codec_chain,
            codec_options,
            chunk_concurrent_minimum,
            chunk_concurrent_maximum,
            num_threads,
            fill_value,
            data_type,
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
            let key_decoder_pairs =
                iter_concurrent_limit!(chunk_concurrent_limit, partial_chunk_items, map, |item| {
                    let storage_handle = Arc::new(StorageHandle::new(self.store.clone()));
                    let input_handle = StoragePartialDecoder::new(storage_handle, item.key.clone());
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
                .collect::<PyResult<Vec<_>>>()?;
            partial_decoder_cache.extend(key_decoder_pairs);
        }

        let data_type_size = self
            .data_type
            .fixed_size()
            .ok_or("variable length data type not supported")
            .map_py_err::<PyTypeError>()?;

        py.detach(move || {
            // FIXME: the `decode_into` methods only support fixed length data types.
            // For variable length data types, need a codepath with non `_into` methods.
            // Collect all the subsets and copy into value on the Python side?
            let update_chunk_subset = |item: ChunkItem| -> PyResult<()> {
                // 1-D scattered fast path (vindex / `arr[idx]`).
                if item.chunk_indices.is_some() {
                    let partial_decoder =
                        partial_decoder_cache.get(&item.key).ok_or_else(|| {
                            PyRuntimeError::new_err(format!(
                                "Partial decoder not found for key: {}",
                                item.key,
                            ))
                        })?;
                    return Self::decode_scattered_into(
                        &item,
                        partial_decoder.as_ref(),
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
