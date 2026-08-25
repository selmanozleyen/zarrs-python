#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use chunk_item::ChunkItem;
use itertools::Itertools;
use lru::LruCache;
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
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{Bytes, ReadableWritableListableStorage, StorageHandle, StoreKey};

mod chunk_item;
mod concurrency;
mod io_pool;
mod runtime;
mod shard_index;
mod store;
#[cfg(test)]
mod tests;
mod utils;

use crate::concurrency::ChunkConcurrentLimitAndCodecOptions;
use crate::shard_index::{ReadGroup, ShardLayout};
use crate::store::StoreConfig;
use crate::utils::{PyCodecErrExt, PyErrExt as _};

/// Bytes allowed to sit fetched but undecoded when the caller does not say.
const DEFAULT_FETCH_BYTE_BUDGET: u64 = 256 << 20;

/// How a planned read is to be run, beyond the fact that it is planned.
#[derive(Clone, Copy)]
pub(crate) struct PlanOptions {
    /// Threads issuing reads. Separate from the decode pool because these block on IO.
    pub fetch_threads: usize,
    /// Ceiling on bytes fetched but not yet decoded.
    pub fetch_byte_budget: u64,
}

/// What every stage of a planned read needs, so a stage takes one argument rather than six.
struct ReadContext<'a> {
    layout: &'a ShardLayout,
    /// Inner chunk shape as plain `u64`s, which is what the index arithmetic wants.
    inner_shape: Vec<u64>,
    /// Inner chunks along each dimension of a shard.
    chunks_per_shard: Vec<u64>,
    output: UnsafeCellSlice<'a, u8>,
    data_type_size: usize,
    codec_options: &'a CodecOptions,
}

/// One inner chunk that can be read a piece at a time: its shard, which chunk, where it sits in
/// the shard, and which of the shard's items want it.
type SubChunkUnit<'a> = (usize, u64, Option<ByteRange>, &'a [usize]);

/// One store read: its shard, the inner chunks it covers, and which items want each of them.
type WholeUnit<'a> = (usize, &'a ReadGroup, &'a HashMap<u64, Vec<usize>>);

/// What one shard contributes to a read, decided before any of its data is fetched.
struct ShardPlan {
    /// Which of the shard's items want each inner chunk, as indices into its item list.
    ///
    /// Built here because working it out is free — deciding which chunks to read means walking
    /// every item's chunks anyway. Without it, decoding a chunk means asking every item of the
    /// shard whether it overlaps, which is `items × chunks` work for `items` worth of answers.
    wanted_by: HashMap<u64, Vec<usize>>,
    reads: ShardReads,
}

enum ShardReads {
    /// The shard does not exist, so everything asked of it is the fill value.
    Absent,
    /// Inner chunks that can be read a piece at a time, with the byte range of each.
    SubChunk(Vec<(u64, Option<ByteRange>)>),
    /// Reads that must fetch whole inner chunks, and the chunks each one covers.
    Whole(Vec<ReadGroup>),
}

/// The items that want one inner chunk, or nothing if it turns out none do.
fn wanted(wanted_by: &HashMap<u64, Vec<usize>>, subchunk: u64) -> &[usize] {
    wanted_by.get(&subchunk).map_or(&[], Vec::as_slice)
}

/// Threads that issue reads, kept apart from the decode pool.
///
/// A fetch thread can block waiting for the byte budget, and only a decode can release it. Sharing
/// one pool would let blocked fetches occupy every thread the decodes need, so the pipeline would
/// stall on itself. Process-wide rather than per array: a caller reading the several arrays of a
/// sparse matrix should not get one pool per array.
static FETCH_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn fetch_pool(threads: usize) -> &'static rayon::ThreadPool {
    FETCH_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("zarrs-fetch-{i}"))
            .build()
            .expect("the fetch pool is built once, with a valid thread count")
    })
}

/// A ceiling on bytes that have been fetched but not yet decoded.
///
/// Without it a read plan large enough to cover an array would fetch all of it before the first
/// decode finished. Credit is held by every decode of a read and returned when the last of them
/// drops it, so the ceiling counts exactly the bytes waiting to be decoded.
struct ByteBudget {
    limit: u64,
    in_flight: Mutex<u64>,
    freed: Condvar,
}

impl ByteBudget {
    fn new(limit: u64) -> Self {
        Self {
            limit,
            in_flight: Mutex::new(0),
            freed: Condvar::new(),
        }
    }

    /// Wait until `bytes` fit under the ceiling, then hold them until the returned credit drops.
    ///
    /// A read larger than the whole ceiling is admitted on its own rather than never, which is
    /// what keeps a ceiling smaller than one read from wedging the pipeline.
    /// Shared rather than borrowed: a credit outlives the fetch that took it, travelling into decode
    /// tasks whose lifetime rayon cannot relate to a local borrow. One `Arc` clone per read is
    /// nothing beside the read, and it costs the type its lifetime parameter, which is what lets a
    /// credit be handed to `Scope::spawn` at all.
    fn admit(self: &Arc<Self>, bytes: u64) -> Credit {
        let mut in_flight = self.in_flight.lock().unwrap();
        while *in_flight > 0 && in_flight.saturating_add(bytes) > self.limit {
            in_flight = self.freed.wait(in_flight).unwrap();
        }
        *in_flight += bytes;
        Credit {
            budget: Arc::clone(self),
            bytes,
        }
    }
}

struct Credit {
    budget: Arc<ByteBudget>,
    bytes: u64,
}

impl Drop for Credit {
    fn drop(&mut self) {
        *self.budget.in_flight.lock().unwrap() -= self.bytes;
        self.budget.freed.notify_all();
    }
}

/// Indices of the inner chunks a subset of a shard overlaps.
fn subchunks_overlapping(subset: &ArraySubset, inner_shape: &[u64]) -> Vec<Vec<u64>> {
    let ranges: Vec<_> = subset
        .to_ranges()
        .iter()
        .zip(inner_shape)
        .map(|(range, size)| (range.start / size)..=((range.end - 1) / size))
        .collect();
    ranges
        .into_iter()
        .multi_cartesian_product()
        .collect::<Vec<_>>()
}

fn ravel(indices: &[u64], shape: &[u64]) -> Option<u64> {
    let mut raveled = 0u64;
    for (index, size) in indices.iter().zip(shape) {
        if index >= size {
            return None;
        }
        raveled = raveled.checked_mul(*size)?.checked_add(*index)?;
    }
    Some(raveled)
}

fn unravel(mut raveled: u64, shape: &[u64]) -> Vec<u64> {
    let mut indices = vec![0; shape.len()];
    for (index, size) in indices.iter_mut().zip(shape).rev() {
        *index = raveled % size;
        raveled /= size;
    }
    indices
}

/// A view of `subset` within the output array `item` writes into.
fn new_output_view<'a>(
    output: UnsafeCellSlice<'a, u8>,
    data_type_size: usize,
    item: &'a ChunkItem,
    subset: &ArraySubset,
) -> PyResult<ArrayBytesFixedDisjointView<'a>> {
    unsafe {
        // SAFETY: the boxes written by one read are disjoint subsets of the output
        ArrayBytesFixedDisjointView::new(
            output,
            data_type_size,
            bytemuck::must_cast_slice(&item.array_shape),
            subset.clone(),
        )
        .map_py_err::<PyRuntimeError>()
    }
}

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
    /// Partial decoders retained across calls, keyed by chunk key. A sharding partial decoder
    /// holds the shard index it decoded, so this is a shard index cache. `None` when disabled.
    pub(crate) shard_index_cache:
        Option<Mutex<LruCache<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>>>,
    /// Shard indexes parsed for the read planner, which needs the offsets themselves rather than
    /// a decoder holding them. Same capacity as `shard_index_cache`; the two would be one cache
    /// once planning replaces the partial-decoder path.
    pub(crate) parsed_index_cache: Option<Mutex<LruCache<StoreKey, Arc<Vec<u64>>>>>,
    /// How this array's shards are laid out, worked out on the first read that needs it.
    /// `None` once resolved means the chunk is not a plain shard, so planning does not apply.
    pub(crate) shard_layout: OnceLock<Option<ShardLayout>>,
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

        // Both shard index caches, and the read planner that uses them, exist for read-only work.
        // Rather than track which cached index a write invalidates, every write empties them. A
        // write moves where a shard's inner chunks live, and being wrong about that is silently
        // wrong data, so this trades cache warmth for having no coherence rules to get wrong.
        if let Some(cache) = &self.shard_index_cache {
            cache.lock().unwrap().clear();
        }
        if let Some(cache) = &self.parsed_index_cache {
            cache.lock().unwrap().clear();
        }

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

    /// Assemble the partial decoders for the chunks that are not read whole, ahead of time and in
    /// parallel. Constructing one decodes the shard index of a sharded chunk, so decoders are
    /// reused from `shard_index_cache` where possible, and put back for the next call.
    fn partial_decoders(
        &self,
        chunk_descriptions: &[ChunkItem],
        chunk_concurrent_limit: usize,
        codec_options: &CodecOptions,
    ) -> PyResult<HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>>> {
        let mut partial_decoders: HashMap<StoreKey, Arc<dyn ArrayPartialDecoderTraits>> =
            HashMap::new();
        let mut missing = chunk_descriptions
            .iter()
            .filter(|item| !(is_whole_chunk(item)))
            .unique_by(|item| item.key.clone())
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(partial_decoders);
        }

        if let Some(cache) = &self.shard_index_cache {
            let mut cache = cache.lock().unwrap();
            missing.retain(|item| match cache.get(&item.key) {
                Some(partial_decoder) => {
                    partial_decoders.insert(item.key.clone(), partial_decoder.clone());
                    false
                }
                None => true,
            });
        }

        let key_decoder_pairs =
            iter_concurrent_limit!(chunk_concurrent_limit, missing, map, |item: &ChunkItem| {
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
                        codec_options,
                    )
                    .map_codec_err()?;
                Ok((item.key.clone(), partial_decoder))
            })
            .collect::<PyResult<Vec<_>>>()?;

        if let Some(cache) = &self.shard_index_cache {
            let mut cache = cache.lock().unwrap();
            for (key, partial_decoder) in &key_decoder_pairs {
                cache.put(key.clone(), partial_decoder.clone());
            }
        }
        partial_decoders.extend(key_decoder_pairs);
        Ok(partial_decoders)
    }

    /// The shard index of `key`, read from the store and kept for later calls.
    fn shard_index(
        &self,
        key: &StoreKey,
        layout: &ShardLayout,
        codec_options: &CodecOptions,
    ) -> PyResult<Option<Arc<Vec<u64>>>> {
        if let Some(index) = self
            .parsed_index_cache
            .as_ref()
            .and_then(|cache| cache.lock().unwrap().get(key).cloned())
        {
            return Ok(Some(index));
        }
        let Some(encoded) = self
            .store
            .get_partial(key, layout.index_byte_range())
            .map_py_err::<PyRuntimeError>()?
        else {
            return Ok(None); // the shard does not exist, so every chunk in it is the fill value
        };
        let index = Arc::new(layout.decode_index(&encoded, codec_options)?);
        if let Some(cache) = &self.parsed_index_cache {
            cache.lock().unwrap().put(key.clone(), index.clone());
        }
        Ok(Some(index))
    }

    /// Everything one shard contributes to a read, worked out before any data is fetched.
    ///
    /// The unit in both variants is a single inner chunk, which is what keeps the pipeline from
    /// deadlocking: a unit depends on exactly one read, so a budget that admits only one read at a
    /// time still makes progress. A requested run spanning two inner chunks becomes two units, not
    /// one unit waiting on two reads.
    fn plan_shard(
        &self,
        ctx: &ReadContext<'_>,
        key: &StoreKey,
        items: &[&ChunkItem],
    ) -> PyResult<ShardPlan> {
        let mut wanted_by: HashMap<u64, Vec<usize>> = HashMap::new();
        for (item_index, item) in items.iter().enumerate() {
            for subchunk in subchunks_overlapping(&item.chunk_subset, &ctx.inner_shape) {
                let subchunk = ravel(&subchunk, &ctx.chunks_per_shard)
                    .ok_or_else(|| PyRuntimeError::new_err("chunk index out of bounds"))?;
                wanted_by.entry(subchunk).or_default().push(item_index);
            }
        }
        let mut needed: Vec<u64> = wanted_by.keys().copied().collect();
        needed.sort_unstable();

        let Some(index) = self.shard_index(key, ctx.layout, ctx.codec_options)? else {
            return Ok(ShardPlan {
                wanted_by,
                reads: ShardReads::Absent,
            });
        };
        let reads = if ctx.layout.can_read_sub_chunk() {
            // Reads happen inside the chunk's decoder, which asks for only the bytes wanted, so
            // there is nothing to merge and nothing to split from the decode.
            ShardReads::SubChunk(
                needed
                    .into_iter()
                    .map(|subchunk| (subchunk, ShardLayout::subchunk_byte_range(&index, subchunk)))
                    .collect(),
            )
        } else {
            ShardReads::Whole(ShardLayout::merge_reads(&index, needed))
        };
        Ok(ShardPlan { wanted_by, reads })
    }

    /// Write the fill value over everything a shard was asked for, for a shard that does not exist.
    fn fill_shard(&self, ctx: &ReadContext<'_>, items: &[&ChunkItem]) -> PyResult<()> {
        for item in items {
            let mut view = new_output_view(ctx.output, ctx.data_type_size, item, &item.subset)?;
            copy_fill_value_into(
                &self.data_type,
                &self.fill_value,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
            )
            .map_codec_err()?;
        }
        Ok(())
    }

    /// Write each requested box's share of one inner chunk into the output.
    ///
    /// `decoder` is `None` when the chunk is absent from the shard, in which case the boxes get the
    /// fill value. Otherwise it decodes straight into the output view — no intermediate buffer, and
    /// one decoder for the whole chunk, so a chain that must inflate it does so once however many
    /// boxes fall inside.
    fn decode_subchunk_into(
        &self,
        ctx: &ReadContext<'_>,
        items: &[&ChunkItem],
        wanted: &[usize],
        subchunk_start: &[u64],
        decoder: Option<&Arc<dyn ArrayPartialDecoderTraits>>,
    ) -> PyResult<()> {
        let subchunk_subset =
            ArraySubset::new_with_start_shape(subchunk_start.to_vec(), ctx.inner_shape.clone())
                .map_py_err::<PyRuntimeError>()?;
        for item in wanted.iter().map(|&index| items[index]) {
            let overlap = match item.chunk_subset.overlap(&subchunk_subset) {
                Ok(overlap) if overlap.num_elements() > 0 => overlap,
                _ => continue,
            };
            let destination = overlap
                .relative_to(item.chunk_subset.start())
                .and_then(|relative| relative.offset(item.subset.start()))
                .map_py_err::<PyRuntimeError>()?;
            let mut view = new_output_view(ctx.output, ctx.data_type_size, item, &destination)?;
            let target = ArrayBytesDecodeIntoTarget::Fixed(&mut view);
            match decoder {
                Some(decoder) => {
                    let source = overlap
                        .relative_to(subchunk_start)
                        .map_py_err::<PyRuntimeError>()?;
                    decoder
                        .partial_decode_into(&source, target, ctx.codec_options)
                        .map_codec_err()?;
                }
                None => copy_fill_value_into(&self.data_type, &self.fill_value, target)
                    .map_codec_err()?,
            }
        }
        Ok(())
    }

    /// Where one inner chunk starts within its shard.
    fn subchunk_start(ctx: &ReadContext<'_>, subchunk: u64) -> Vec<u64> {
        unravel(subchunk, &ctx.chunks_per_shard)
            .iter()
            .zip(&ctx.inner_shape)
            .map(|(index, size)| index * size)
            .collect()
    }

    /// Serve a read by planning byte ranges against the shard index instead of asking the sharding
    /// partial decoder for one region at a time.
    ///
    /// Every inner chunk a read touches is fetched and decoded once however many of the requested
    /// boxes fall inside it, and nothing outside those boxes is copied. Units from every shard go
    /// into one pool, so reads of one shard overlap each other as well as other shards'.
    ///
    /// When inner chunks must be fetched whole, fetching and decoding run on separate pools: a
    /// landed read hands one decode task per chunk straight to the decode pool rather than waiting
    /// for its siblings, and the bytes it holds are returned to the budget when that decode
    /// finishes. The fetch pool is separate from the decode pool precisely so that a fetch thread
    /// blocked on the budget cannot starve the decode that would release it.
    fn retrieve_planned(
        &self,
        layout: &ShardLayout,
        chunk_descriptions: &[ChunkItem],
        output: UnsafeCellSlice<u8>,
        chunk_concurrent_limit: usize,
        codec_options: &CodecOptions,
        options: PlanOptions,
    ) -> PyResult<()> {
        let ctx = &ReadContext {
            layout,
            inner_shape: layout.inner_chunk_shape.iter().map(|d| d.get()).collect(),
            chunks_per_shard: layout.chunks_per_shard.iter().map(|d| d.get()).collect(),
            output,
            data_type_size: self
                .data_type
                .fixed_size()
                .ok_or("variable length data type not supported")
                .map_py_err::<PyTypeError>()?,
            codec_options,
        };

        let mut by_shard: HashMap<&StoreKey, Vec<&ChunkItem>> = HashMap::new();
        for item in chunk_descriptions {
            by_shard.entry(&item.key).or_default().push(item);
        }
        let shards: Vec<(&StoreKey, Vec<&ChunkItem>)> = by_shard.into_iter().collect();
        let shard_indices: Vec<usize> = (0..shards.len()).collect();

        // Plan every shard first, and in parallel: an index read is small and usually cached, but
        // it is still a read and should not queue behind another shard's.
        let plans = iter_concurrent_limit!(
            chunk_concurrent_limit,
            shard_indices,
            map,
            |shard: usize| -> PyResult<(usize, ShardPlan)> {
                let (key, items) = &shards[shard];
                Ok((shard, self.plan_shard(ctx, key, items)?))
            }
        )
        .collect::<PyResult<Vec<_>>>()?;

        let mut sub_chunk_units = Vec::new();
        let mut whole_units = Vec::new();
        for (shard, plan) in &plans {
            match &plan.reads {
                ShardReads::Absent => {
                    self.fill_shard(ctx, &shards[*shard].1)?;
                }
                ShardReads::SubChunk(subchunks) => {
                    sub_chunk_units.extend(subchunks.iter().map(|&(subchunk, range)| {
                        (*shard, subchunk, range, wanted(&plan.wanted_by, subchunk))
                    }));
                }
                ShardReads::Whole(groups) => {
                    whole_units.extend(groups.iter().map(|group| (*shard, group, &plan.wanted_by)));
                }
            }
        }

        self.read_sub_chunk_units(ctx, &shards, &sub_chunk_units, options)?;
        self.read_whole_units(ctx, &shards, &whole_units, options)?;
        Ok(())
    }

    /// Read the inner chunks that can be fetched a piece at a time.
    ///
    /// Reading and decoding are one step here: the chunk's decoder asks the store for only the
    /// bytes each box wants, so there is nothing to hand to a separate decode stage.
    fn read_sub_chunk_units(
        &self,
        ctx: &ReadContext<'_>,
        shards: &[(&StoreKey, Vec<&ChunkItem>)],
        sub_chunk_units: &[SubChunkUnit<'_>],
        options: PlanOptions,
    ) -> PyResult<()> {
        if sub_chunk_units.is_empty() {
            return Ok(());
        }
        let budget = Arc::new(ByteBudget::new(options.fetch_byte_budget));
        let read_unit = |&(shard, subchunk, range, wanted): &SubChunkUnit<'_>| {
            let bytes = match range {
                Some(ByteRange::FromStart(_, Some(length))) => length,
                _ => 0,
            };
            // Reading and decoding are one step here, so the credit covers both.
            let _credit = budget.admit(bytes);
            let (key, items) = &shards[shard];
            let decoder = match range {
                Some(ByteRange::FromStart(offset, Some(length))) => {
                    Some(ctx.layout.subchunk_decoder(
                        &self.store,
                        key,
                        (offset, length),
                        &self.data_type,
                        &self.fill_value,
                        ctx.codec_options,
                    )?)
                }
                _ => None, // absent from the shard, so it is all fill value
            };
            self.decode_subchunk_into(
                ctx,
                items,
                wanted,
                &Self::subchunk_start(ctx, subchunk),
                decoder.as_ref(),
            )
        };
        fetch_pool(options.fetch_threads).install(|| {
            iter_concurrent_limit!(
                options.fetch_threads,
                sub_chunk_units,
                try_for_each,
                read_unit
            )
        })?;
        Ok(())
    }

    /// Read the inner chunks that must be fetched whole, fetching and decoding on separate pools.
    fn read_whole_units(
        &self,
        ctx: &ReadContext<'_>,
        shards: &[(&StoreKey, Vec<&ChunkItem>)],
        whole_units: &[WholeUnit<'_>],
        options: PlanOptions,
    ) -> PyResult<()> {
        if whole_units.is_empty() {
            return Ok(());
        }
        let largest = whole_units
            .iter()
            .map(|(_, group, _)| group.length)
            .max()
            .unwrap_or(0);
        let budget = Arc::new(ByteBudget::new(options.fetch_byte_budget.max(largest)));
        let failure: Mutex<Option<PyErr>> = Mutex::new(None);
        // The rayon scope is the DECODE side and only that. Fetching never runs on it: a rayon
        // worker parked in a read cannot steal, and a fetch blocked on the byte budget holds a
        // worker that only a decode can release, so sharing one pool lets the fetch side starve the
        // decode side of the threads that would unblock it. `Scope::spawn` is callable from a
        // foreign thread, which is what lets a fetch thread hand a landed buffer straight to a core
        // and go back to reading.
        rayon::scope(|scope| {
            // Everything after the bytes arrive: one decode task per inner chunk, queued the
            // moment its read lands rather than when its siblings do.
            let spawn_decodes = |unit: usize, fetched: Option<Arc<Bytes>>, credit: Credit| {
                let (shard, group, wanted_by) = whole_units[unit];
                let (_, items) = &shards[shard];
                let credit = Arc::new(credit);
                for &(subchunk, offset, length) in &group.subchunks {
                    let credit = credit.clone();
                    let fetched = fetched.clone();
                    let failure = &failure;
                    scope.spawn(move |_| {
                        let decoder = match &fetched {
                            Some(buffer) => {
                                match ctx.layout.fetched_subchunk_decoder(
                                    buffer.clone(),
                                    usize::try_from(offset).unwrap_or(usize::MAX),
                                    usize::try_from(length).unwrap_or(0),
                                    &self.data_type,
                                    &self.fill_value,
                                    ctx.codec_options,
                                ) {
                                    Ok(decoder) => Some(decoder),
                                    Err(err) => {
                                        *failure.lock().unwrap() = Some(err);
                                        return;
                                    }
                                }
                            }
                            None => None,
                        };
                        if let Err(err) = self.decode_subchunk_into(
                            ctx,
                            items,
                            wanted(wanted_by, subchunk),
                            &Self::subchunk_start(ctx, subchunk),
                            decoder.as_ref(),
                        ) {
                            *failure.lock().unwrap() = Some(err);
                        }
                        // Dropping the credit here, and not before, is what makes the budget mean
                        // "bytes fetched but not yet decoded".
                        drop(credit);
                    });
                }
            };

            io_pool::for_each_blocking(whole_units.len(), options.fetch_threads, |unit| {
                if failure.lock().unwrap().is_some() {
                    return;
                }
                let (shard, group, _) = whole_units[unit];
                // Admitted before the read, so the ceiling bounds bytes in flight
                // rather than bytes already spent.
                let credit = budget.admit(group.length);
                let (key, _) = &shards[shard];
                let fetched = if group.length == 0 {
                    None // absent from the shard, so it is all fill value
                } else {
                    match self.store.get_partial(key, group.byte_range()) {
                        Ok(Some(bytes)) => Some(Arc::new(bytes)),
                        Ok(None) => None,
                        Err(err) => {
                            *failure.lock().unwrap() =
                                Some(PyRuntimeError::new_err(err.to_string()));
                            return;
                        }
                    }
                };
                spawn_decodes(unit, fetched, credit);
            });
        });
        if let Some(err) = failure.into_inner().unwrap() {
            return Err(err);
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
        direct_io=false,
        file_handle_cache_size=0,
        shard_index_cache_size=0,
    ))]
    // The argument list is the Python constructor's, so its width is not ours to choose.
    #[allow(clippy::too_many_arguments)]
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
        shard_index_cache_size: usize,
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
            shard_index_cache: NonZeroUsize::new(shard_index_cache_size)
                .map(|capacity| Mutex::new(LruCache::new(capacity))),
            parsed_index_cache: NonZeroUsize::new(shard_index_cache_size)
                .map(|capacity| Mutex::new(LruCache::new(capacity))),
            shard_layout: OnceLock::new(),
        })
    }

    #[pyo3(signature = (chunk_descriptions, value, *, plan_reads=false, fetch_threads=0, fetch_byte_budget=0))]
    fn retrieve_chunks_and_apply_index(
        &self,
        py: Python,
        chunk_descriptions: Vec<chunk_item::ChunkItem>, // FIXME: Ref / iterable?
        value: &Bound<'_, PyUntypedArray>,
        plan_reads: bool,
        fetch_threads: usize,
        fetch_byte_budget: u64,
    ) -> PyResult<()> {
        // Get input array
        let output = Self::nparray_to_unsafe_cell_slice(value)?;

        // Adjust the concurrency based on the codec chain and the first chunk description
        let Some((chunk_concurrent_limit, codec_options)) =
            chunk_descriptions.get_chunk_concurrent_limit_and_codec_options(self)?
        else {
            return Ok(());
        };

        if plan_reads {
            let layout = self.shard_layout.get_or_init(|| {
                ShardLayout::new(&self.codec_chain, &chunk_descriptions[0].shape)
                    .ok()
                    .flatten()
            });
            if let Some(layout) = layout {
                return py.detach(|| {
                    self.retrieve_planned(
                        layout,
                        &chunk_descriptions,
                        output,
                        chunk_concurrent_limit,
                        &codec_options,
                        PlanOptions {
                            fetch_threads: if fetch_threads == 0 {
                                rayon::current_num_threads()
                            } else {
                                fetch_threads
                            },
                            fetch_byte_budget: if fetch_byte_budget == 0 {
                                DEFAULT_FETCH_BYTE_BUDGET
                            } else {
                                fetch_byte_budget
                            },
                        },
                    )
                });
            }
        }

        let partial_decoder_cache =
            self.partial_decoders(&chunk_descriptions, chunk_concurrent_limit, &codec_options)?;

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

/// A Python module implemented in Rust.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<CodecPipelineImpl>()?;
    m.add_class::<chunk_item::ChunkItem>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);
