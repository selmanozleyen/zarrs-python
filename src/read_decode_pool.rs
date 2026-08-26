//! A reader pool and a decode pool: one job per innermost chunk. A reader fetches that
//! chunk's bytes and signals the decode pool; a decode worker decodes the chunk once and
//! copies out the elements the selection wants from it.
//!
//! Why the halves are split at all: `partial_decode` reads *and* decodes inside one call
//! on one worker, so a worker parked on the filesystem cannot decode a chunk another
//! worker already fetched. A read here costs ~16x a decode (~2.9 ms against ~0.185 ms for
//! a 358 KB blosc chunk), so what matters is how many reads are outstanding -- which is
//! what `read_concurrency` sets, independently of the decode width.
//!
//! Shape:
//!
//! - **Two persistent `rayon::ThreadPool`s**, built once on the pipeline. The pools
//!   persist; `pool.scope()` per call makes the borrows scoped. Plain `std::thread` cannot
//!   do both: scoped threads must join before the borrow ends, and persistent threads need
//!   `'static` messages, which a borrowed output piece is not.
//! - **Readers get their own pool**, oversubscribed past the core count on purpose: a read
//!   is latency-bound and a parked thread costs a stack, not a core. Because the pool is
//!   dedicated, a reader parked in `get_partial` starves no decoder.
//! - **Long-lived workers, not a task per chunk.** Each pool spawns its width once per
//!   call and each worker drains the queue until it closes, so a decode worker reuses one
//!   scratch buffer across every chunk it handles.
//! - **Nothing nests.** The codec chain gets `concurrent_target(1)`, so a decode worker
//!   spawns no work underneath itself. All the parallelism is R readers plus D decoders.
//!
//! Byte extents come from `shard_index`, which reads sharding's own offset/size table --
//! so this runs against the RELEASED zarrs, with no patched crate anywhere.
//!
//! Safety: the output is carved with `split_at_mut` before any job runs and each piece
//! travels *inside* its job. Moving a `&mut [u8]` through the channel is what transfers
//! exclusivity, so disjointness is checked by the compiler instead of asserted in a
//! comment. No `UnsafeCellSlice` on this path; the one `unsafe` is the scratch view, which
//! has a single writer on its own thread.
//!
//! ponytail: the pool's scope is ONE call, so a chunk touched by two selections is read
//! and decoded twice, and read concurrency is capped by the items in one call (~70 here).
//! Widening the scope to a whole batch fixes both at once -- a job would carry several
//! (piece, coords) targets and dedup would be exact -- but it needs the Python side to
//! hand over more than one selection at a time. Next step, deliberately not this one.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crossbeam_channel::{Receiver, Sender, unbounded};
use pyo3::PyResult;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{
    ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset, ArrayToBytesCodecTraits,
    CodecOptions,
};
use zarrs::storage::StoreKey;
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::MaybeBytes;

use crate::CodecPipelineImpl;
use crate::chunk_item::ChunkItem;
use crate::shard_index::ShardInfo;
use crate::utils::{PyCodecErrExt as _, PyErrExt as _};

/// What the pool did, so a run can prove this path executed rather than assuming it.
///
/// The fused path produces correct output too, so only a count separates "the pool ran"
/// from "the pool was configured and something else ran". `chunks == reads == decodes` is
/// the invariant: one job per innermost chunk, read once, decoded once. Absent chunks are
/// counted separately because they have no read to issue.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct PoolCounts {
    pub chunks: usize,
    pub reads: usize,
    pub decodes: usize,
    pub absent: usize,
    pub shard_indexes: usize,
    /// Items this path could not take; these fall back to the fused path.
    pub declined: usize,
}

/// One innermost chunk: fetch `range` from `key`, decode it, copy `coords` into `piece`.
struct Job<'a> {
    key: StoreKey,
    range: ByteRange,
    /// Exclusively owned by this job. Travelling inside the job is what makes the
    /// disjointness a move rather than an assertion.
    piece: &'a mut [u8],
    coords: &'a [u64],
}

/// An item whose chunk has been located in its shard. `None` range = the chunk is absent.
struct Located<'a> {
    item: &'a ChunkItem,
    range: Option<ByteRange>,
}

impl CodecPipelineImpl {
    /// Read and decode `items` with a reader pool feeding a decode pool.
    ///
    /// `items` must be chunk-unit items: one whole innermost chunk each, carrying the
    /// coordinates wanted from it. Returns the items this path could not take, for the
    /// caller to run down the fused path.
    pub(crate) fn retrieve_read_decode_pool<'a>(
        &self,
        shard: &ShardInfo,
        items: &'a [ChunkItem],
        output: &mut [u8],
        codec_options: &CodecOptions,
    ) -> PyResult<(Vec<&'a ChunkItem>, PoolCounts)> {
        let size = self
            .data_type
            .fixed_size()
            .ok_or("variable length data type not supported")
            .map_py_err::<PyTypeError>()?;

        // Flat by construction: a target of one means a decode worker spawns nothing
        // underneath itself. Inheriting the fused path's target would renest silently --
        // and that target is the 4 of the (4, 4) split `calc_concurrency_outer_inner`
        // produces here, which is what leaves the fused path using a quarter of its
        // threads.
        let codec_options = codec_options.clone().with_concurrent_target(1);
        let codec_options = &codec_options;

        // ------------------------------------------------- locate each chunk in its shard
        let (located, declined, shard_indexes) = self.locate_chunks(shard, items, codec_options)?;
        let declined_n = declined.len();
        if located.is_empty() {
            return Ok((
                declined,
                PoolCounts {
                    declined: declined_n,
                    shard_indexes,
                    ..PoolCounts::default()
                },
            ));
        }

        // --------------------------------------------------- the output, split up front
        // Each item's slice of the output, carved at the offset its own subset names --
        // NOT sequentially in item order. A call carries several entries, one per shard,
        // each with its own out_selection start, and declined items leave holes that
        // belong to the fused path. Carving in item order would hand a worker somebody
        // else's bytes.
        //
        // This path is 1-D by construction (`_chunk_unit_items` declines anything else),
        // so an output offset is just `subset.start * size`.
        let mut order: Vec<usize> = (0..located.len()).collect();
        order.sort_by_key(|&i| output_offset(located[i].item));

        let mut carved: Vec<(usize, &mut [u8])> = Vec::with_capacity(order.len());
        let mut rest: &mut [u8] = output;
        let mut cursor = 0usize;
        for &i in &order {
            let item = located[i].item;
            let off = output_offset(item) * size;
            let len = coords_of(item)?.len() * size;
            if off < cursor {
                return Err(PyRuntimeError::new_err(format!(
                    "{} overlaps an earlier item's output: pieces must be disjoint",
                    item.key
                )));
            }
            let skip = off - cursor;
            if skip + len > rest.len() {
                return Err(PyRuntimeError::new_err(format!(
                    "{} names output bytes {off}..{} beyond the {} available",
                    item.key,
                    off + len,
                    cursor + rest.len()
                )));
            }
            let (_hole, tail) = rest.split_at_mut(skip);
            let (piece, tail) = tail.split_at_mut(len);
            carved.push((i, piece));
            rest = tail;
            cursor = off + len;
        }

        // ------------------------------------------------------------------ the job list
        let mut jobs: Vec<Job<'_>> = Vec::with_capacity(carved.len());
        let mut absent = 0usize;
        for (i, piece) in carved {
            let item = located[i].item;
            let coords = coords_of(item)?;
            match located[i].range {
                Some(range) => jobs.push(Job {
                    key: item.key.clone(),
                    range,
                    piece,
                    coords,
                }),
                // Absent chunk: every element the selection wants from it is the fill
                // value, written now. There is no read and no decode to do.
                None => {
                    absent += 1;
                    fill_piece(piece, &self.fill_value, size)?;
                }
            }
        }

        let chunks = jobs.len();
        if jobs.is_empty() {
            return Ok((
                declined,
                PoolCounts {
                    absent,
                    shard_indexes,
                    declined: declined_n,
                    ..PoolCounts::default()
                },
            ));
        }

        // ------------------------------------------------------------ read, then decode
        let (job_tx, job_rx): (Sender<Job<'_>>, Receiver<Job<'_>>) = unbounded();
        for job in jobs {
            job_tx.send(job).expect("receiver is alive");
        }
        drop(job_tx); // readers see the queue close once it drains

        // Unbounded on purpose: a reader never waits for a decoder. Peak resident is
        // items-per-call x encoded chunk size, small at ~70 items per call; it scales with
        // the call size, so revisit if the scope widens to a whole batch.
        let (done_tx, done_rx): (Sender<(Job<'_>, MaybeBytes)>, Receiver<_>) = unbounded();

        let reads_counter = AtomicUsize::new(0);
        let decodes_counter = AtomicUsize::new(0);
        let first_error: Mutex<Option<String>> = Mutex::new(None);
        let record_owned = |e: String| {
            let mut slot = first_error.lock().expect("error slot poisoned");
            if slot.is_none() {
                *slot = Some(e);
            }
        };
        // Every worker closure is `move`, and a loop body would otherwise move the counter
        // into the first iteration. `&AtomicUsize` is Copy, so hand each closure a
        // reference instead.
        let reads = &reads_counter;
        let decodes = &decodes_counter;
        let record = &record_owned;

        self.decode_pool.scope(|decoders| {
            // Long-lived workers: one spawn per worker per call, not one per chunk, so a
            // worker's scratch buffer is reused across every chunk it handles.
            for _ in 0..self.decode_pool.current_num_threads() {
                let done_rx = done_rx.clone();
                decoders.spawn(move |_| {
                    let mut scratch: Vec<u8> = Vec::new();
                    while let Ok((job, bytes)) = done_rx.recv() {
                        match self.decode_into_piece(
                            shard,
                            bytes,
                            job.piece,
                            job.coords,
                            size,
                            &mut scratch,
                            codec_options,
                        ) {
                            Ok(()) => {
                                decodes.fetch_add(1, Ordering::Relaxed);
                            }
                            Err(e) => record(format!("decode {} failed: {e}", job.key)),
                        }
                    }
                });
            }
            drop(done_rx);

            self.read_pool.scope(|readers| {
                for _ in 0..self.read_pool.current_num_threads() {
                    let job_rx = job_rx.clone();
                    let done_tx = done_tx.clone();
                    readers.spawn(move |_| {
                        while let Ok(job) = job_rx.recv() {
                            match self.store.get_partial(&job.key, job.range) {
                                Ok(bytes) => {
                                    reads.fetch_add(1, Ordering::Relaxed);
                                    // The signal: handing the bytes over wakes a decode
                                    // worker. The reader takes its next job at once.
                                    if done_tx.send((job, bytes)).is_err() {
                                        return; // decoders gave up on an error
                                    }
                                }
                                Err(e) => record(format!("read {} failed: {e}", job.key)),
                            }
                        }
                    });
                }
            });
            // Every reader has returned, so no more sends: closing this ends the decoders.
            drop(done_tx);
        });

        if let Some(e) = first_error.lock().expect("error slot poisoned").take() {
            return Err(PyRuntimeError::new_err(e));
        }
        Ok((
            declined,
            PoolCounts {
                chunks,
                reads: reads.load(Ordering::Relaxed),
                decodes: decodes.load(Ordering::Relaxed),
                absent,
                shard_indexes,
                declined: declined_n,
            },
        ))
    }

    /// Where each item's innermost chunk lives, from its shard's own offset/size table.
    ///
    /// One index read per distinct shard, cached for the call, so N items sharing a shard
    /// cost one index read rather than N.
    fn locate_chunks<'a>(
        &self,
        shard: &ShardInfo,
        items: &'a [ChunkItem],
        codec_options: &CodecOptions,
    ) -> PyResult<(Vec<Located<'a>>, Vec<&'a ChunkItem>, usize)> {
        let mut indexes: HashMap<StoreKey, Option<Vec<u64>>> = HashMap::new();
        let mut located = Vec::with_capacity(items.len());
        let mut declined = Vec::new();

        for item in items {
            if item.coords.is_none() {
                declined.push(item);
                continue;
            }
            if !indexes.contains_key(&item.key) {
                let chunks_per_shard = shard.chunks_per_shard(&item.shape)?;
                let index =
                    shard.read_index(&self.store, &item.key, &chunks_per_shard, codec_options)?;
                indexes.insert(item.key.clone(), index);
            }
            let Some(index) = indexes.get(&item.key).expect("just inserted") else {
                // No shard at all: everything in it is the fill value.
                located.push(Located { item, range: None });
                continue;
            };
            let linear = shard.linear_index_1d(item.chunk_subset.start().first().copied().unwrap_or(0));
            located.push(Located {
                item,
                range: ShardInfo::chunk_range(index, linear)?,
            });
        }
        let shard_indexes = indexes.len();
        Ok((located, declined, shard_indexes))
    }

    /// Decode one whole innermost chunk into `scratch`, then copy the wanted elements out.
    ///
    /// The whole chunk, deliberately: blosc partial decode is quantised to blocks, which
    /// prices one getitem per element at 23,233x a full decode plus a gather. The chunk is
    /// the decode unit and the gather picks out of it.
    #[allow(clippy::too_many_arguments)]
    fn decode_into_piece(
        &self,
        shard: &ShardInfo,
        bytes: MaybeBytes,
        piece: &mut [u8],
        coords: &[u64],
        size: usize,
        scratch: &mut Vec<u8>,
        codec_options: &CodecOptions,
    ) -> PyResult<()> {
        let Some(bytes) = bytes else {
            // The index named an extent but the store has nothing there.
            return Err(PyRuntimeError::new_err(
                "the shard index named a byte range the store does not have",
            ));
        };

        let shape = &shard.inner_shape;
        let elements: u64 = shape.iter().map(|s| s.get()).product();
        let needed = usize::try_from(elements).map_py_err::<PyRuntimeError>()? * size;
        scratch.clear();
        scratch.resize(needed, 0);

        let shape_u64: Vec<u64> = shape.iter().map(|s| s.get()).collect();
        let slice = UnsafeCellSlice::new(scratch.as_mut_slice());
        let mut view = unsafe {
            // SAFETY: this view is the only writer to `scratch`, which this worker owns.
            ArrayBytesFixedDisjointView::new(
                slice,
                size,
                &shape_u64,
                ArraySubset::new_with_shape(shape_u64.clone()),
            )
            .map_py_err::<PyRuntimeError>()?
        };
        shard
            .inner_chain
            .decode_into(
                Cow::Owned(bytes.into()),
                shape,
                &self.data_type,
                &self.fill_value,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                codec_options,
            )
            .map_codec_err()?;

        gather(scratch, coords, piece, size)
    }
}

/// The gather zarr-python does with one numpy fancy index, over an already-decoded buffer.
///
/// `piece` is exactly `coords.len()` elements and contiguous, because the indices reached
/// us non-decreasing -- so this writes straight into the output with no temporary.
fn gather(scratch: &[u8], coords: &[u64], piece: &mut [u8], size: usize) -> PyResult<()> {
    if piece.len() != coords.len() * size {
        return Err(PyRuntimeError::new_err(
            "output piece does not match the coordinate count",
        ));
    }
    for (n, &c) in coords.iter().enumerate() {
        let src = usize::try_from(c).map_py_err::<PyRuntimeError>()? * size;
        let Some(element) = scratch.get(src..src + size) else {
            return Err(PyRuntimeError::new_err(format!(
                "coordinate {c} is outside the {} elements decoded",
                scratch.len() / size
            )));
        };
        piece[n * size..(n + 1) * size].copy_from_slice(element);
    }
    Ok(())
}

/// An absent chunk contributes only fill value, repeated across the piece.
fn fill_piece(piece: &mut [u8], fill_value: &zarrs::array::FillValue, size: usize) -> PyResult<()> {
    let bytes = fill_value.as_ne_bytes();
    if bytes.len() != size {
        return Err(PyRuntimeError::new_err(
            "the fill value is not one element wide",
        ));
    }
    for slot in piece.chunks_exact_mut(size) {
        slot.copy_from_slice(bytes);
    }
    Ok(())
}

fn coords_of(item: &ChunkItem) -> PyResult<&Vec<u64>> {
    item.coords
        .as_ref()
        .ok_or("this path requires chunk-unit items, which carry coordinates")
        .map_py_err::<PyRuntimeError>()
}

/// Where an item's elements land in the output, in elements.
///
/// 1-D only, which is what this path accepts: `_chunk_unit_items` declines any selection
/// whose chunk subset, chunk selection or output selection is not one-dimensional.
fn output_offset(item: &ChunkItem) -> usize {
    usize::try_from(item.subset.start().first().copied().unwrap_or(0)).unwrap_or(usize::MAX)
}
