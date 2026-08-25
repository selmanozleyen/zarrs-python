//! Reading a shard's index directly, so a scattered read can be planned as byte ranges.
//!
//! `zarrs` keeps the shard index inside its sharding partial decoder and offers no way to ask
//! "where is inner chunk k", so a read expressed through that decoder can only ever say "decode
//! this region for me" — one call per region, each fetching and inflating whole inner chunks
//! again. With the index in hand the ranges can be deduplicated and merged before any of them is
//! issued, which is what the `zarr-python` pipeline does.
//!
//! Only the layout this covers is handled: a chunk whose codec chain is exactly the sharding
//! codec. An array-to-array or bytes-to-bytes codec wrapping it means the shard bytes are not
//! laid out as the index describes, so `ShardLayout::new` returns `None` and the caller keeps to
//! the partial decoder.

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use pyo3::PyResult;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use zarrs::array::codec::api::{
    ArrayBytesRaw, ByteIntervalPartialDecoder, BytesPartialDecoderTraits, CodecError,
    CodecTraits as _,
};
use zarrs::array::{
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits as _, BytesRepresentation, CodecChain,
    CodecMetadataOptions, CodecOptions, DataType, FillValue, StoragePartialDecoder,
    data_type::uint64,
};
use zarrs::metadata::ConfigurationSerialize;
use zarrs::metadata_ext::codec::sharding::{ShardingCodecConfiguration, ShardingIndexLocation};
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator, InvalidByteRangeError};
use zarrs::storage::{
    Bytes, ReadableWritableListableStorage, StorageError, StorageHandle, StoreKey,
};

use crate::utils::{PyCodecErrExt as _, PyErrExt as _};

/// An inner chunk that is absent from the shard is marked with this offset and length.
const EMPTY: u64 = u64::MAX;

/// One store read, and the inner chunks it covers as `(index, offset within the read, length)`.
/// A zero `length` group is an inner chunk the shard omits, which needs no read at all.
pub struct ReadGroup {
    pub offset: u64,
    pub length: u64,
    pub subchunks: Vec<(u64, u64, u64)>,
}

impl ReadGroup {
    fn end(&self) -> u64 {
        self.offset + self.length
    }

    pub fn byte_range(&self) -> ByteRange {
        ByteRange::FromStart(self.offset, Some(self.length))
    }
}

pub struct ShardLayout {
    /// Shape of an inner chunk, the smallest independently decodable unit of the shard.
    pub inner_chunk_shape: Vec<NonZeroU64>,
    /// Number of inner chunks along each dimension of the shard.
    pub chunks_per_shard: Vec<NonZeroU64>,
    /// Codecs of an inner chunk, for decoding one once it has been fetched.
    pub inner_codecs: Arc<CodecChain>,
    index_codecs: CodecChain,
    index_shape: Vec<NonZeroU64>,
    index_encoded_size: u64,
    index_location: ShardingIndexLocation,
}

impl ShardLayout {
    /// Describe the sharding of a chunk with this codec chain, or `None` if the chain is anything
    /// other than a bare sharding codec.
    pub fn new(codec_chain: &CodecChain, shard_shape: &[NonZeroU64]) -> PyResult<Option<Self>> {
        if !codec_chain.array_to_array_codecs().is_empty()
            || !codec_chain.bytes_to_bytes_codecs().is_empty()
        {
            return Ok(None);
        }
        let Some(configuration) = codec_chain
            .array_to_bytes_codec()
            .configuration_v3(&CodecMetadataOptions::default())
        else {
            return Ok(None);
        };
        let Ok(configuration) = ShardingCodecConfiguration::try_from_configuration(configuration)
        else {
            return Ok(None);
        };
        let ShardingCodecConfiguration::V1(configuration) = configuration else {
            return Ok(None);
        };

        let inner_chunk_shape: Vec<NonZeroU64> = configuration.chunk_shape.as_slice().into();
        if inner_chunk_shape.len() != shard_shape.len() {
            return Ok(None);
        }
        // A shard shape that is not a whole number of inner chunks is not a shard we can index.
        let mut chunks_per_shard = Vec::with_capacity(shard_shape.len());
        for (shard, inner) in shard_shape.iter().zip(&inner_chunk_shape) {
            if shard.get() % inner.get() != 0 {
                return Ok(None);
            }
            chunks_per_shard.push(
                NonZeroU64::new(shard.get() / inner.get()).expect("nonzero divided by a divisor"),
            );
        }

        let inner_codecs =
            Arc::new(CodecChain::from_metadata(&configuration.codecs).map_py_err::<PyTypeError>()?);
        let index_codecs =
            CodecChain::from_metadata(&configuration.index_codecs).map_py_err::<PyTypeError>()?;

        // The index is a `chunks_per_shard × 2` array of u64 offset/length pairs.
        let mut index_shape = chunks_per_shard.clone();
        index_shape.push(NonZeroU64::new(2).expect("2 is nonzero"));
        let index_encoded_size = match index_codecs
            .encoded_representation(&index_shape, &uint64(), &FillValue::from(EMPTY))
            .map_codec_err()?
        {
            BytesRepresentation::FixedSize(size) => size,
            // A variable-size index cannot be located without decoding what surrounds it.
            BytesRepresentation::BoundedSize(_) | BytesRepresentation::UnboundedSize => {
                return Ok(None);
            }
        };

        Ok(Some(Self {
            inner_chunk_shape,
            chunks_per_shard,
            inner_codecs,
            index_codecs,
            index_shape,
            index_encoded_size,
            index_location: configuration.index_location,
        }))
    }

    /// Where the index sits within a shard.
    pub fn index_byte_range(&self) -> ByteRange {
        match self.index_location {
            ShardingIndexLocation::Start => ByteRange::FromStart(0, Some(self.index_encoded_size)),
            ShardingIndexLocation::End => ByteRange::Suffix(self.index_encoded_size),
        }
    }

    /// Decode an index into `2 * chunks_per_shard` alternating offsets and lengths.
    pub fn decode_index(&self, encoded: &[u8], options: &CodecOptions) -> PyResult<Vec<u64>> {
        let decoded = self
            .index_codecs
            .decode(
                encoded.into(),
                &self.index_shape,
                &uint64(),
                &FillValue::from(EMPTY),
                options,
            )
            .map_codec_err()?;
        let decoded = decoded.into_fixed().map_py_err::<PyTypeError>()?;
        let (chunks, remainder) = decoded.as_chunks::<8>();
        if !remainder.is_empty() {
            return Err(PyRuntimeError::new_err(
                "decoded shard index is not a whole number of u64 values",
            ));
        }
        Ok(chunks.iter().map(|v| u64::from_ne_bytes(*v)).collect())
    }

    /// Byte range of one inner chunk within its shard, or `None` if the shard omits it.
    pub fn subchunk_byte_range(index: &[u64], subchunk: u64) -> Option<ByteRange> {
        let at = usize::try_from(subchunk).ok()? * 2;
        let (&offset, &length) = (index.get(at)?, index.get(at + 1)?);
        if offset == EMPTY && length == EMPTY {
            None
        } else {
            Some(ByteRange::FromStart(offset, Some(length)))
        }
    }

    /// Group the inner chunks a read needs into store reads.
    ///
    /// Two inner chunks share a read when they touch, which costs nothing: not a byte outside the
    /// request is read.
    ///
    /// Adjacency is common rather than incidental: a shard written in one pass lays its inner
    /// chunks down in index order, so a run of consecutive inner chunks is usually one span, and
    /// needing all of them collapses to a single read of the whole shard.
    ///
    /// A merged read never spans more than one shard, so the largest it can grow to is one shard.
    pub fn merge_reads(index: &[u64], mut subchunks: Vec<u64>) -> Vec<ReadGroup> {
        subchunks.sort_unstable();
        subchunks.dedup();
        let mut groups: Vec<ReadGroup> = Vec::new();
        for subchunk in subchunks {
            let Some(ByteRange::FromStart(offset, Some(length))) =
                Self::subchunk_byte_range(index, subchunk)
            else {
                // Absent from the shard, so there is nothing to read for it.
                groups.push(ReadGroup {
                    offset: 0,
                    length: 0,
                    subchunks: vec![(subchunk, 0, 0)],
                });
                continue;
            };
            match groups.last_mut() {
                // Touching the read behind it, so extend that rather than start another. Chunks
                // laid out of index order fail the `==` and simply stay separate.
                Some(group) if group.length > 0 && offset == group.end() => {
                    group
                        .subchunks
                        .push((subchunk, offset - group.offset, length));
                    group.length = offset + length - group.offset;
                }
                _ => groups.push(ReadGroup {
                    offset,
                    length,
                    subchunks: vec![(subchunk, 0, length)],
                }),
            }
        }
        groups
    }

    /// Whether an inner chunk can be read a piece at a time.
    ///
    /// True when nothing in the inner codec chain has to see all of its input before decoding:
    /// with no compressor, an element's position in the chunk is arithmetic, so only the bytes
    /// asked for need to be read. A compressor makes the chunk atomic — it is fetched and
    /// inflated whole however little of it was wanted.
    pub fn can_read_sub_chunk(&self) -> bool {
        self.inner_codecs.partial_decoder_capability().partial_read
    }

    /// A decoder for one inner chunk, reading through the shard at that chunk's byte range.
    ///
    /// How much it reads is the codec chain's decision, not ours: a chain that can partial-read
    /// fetches only the ranges each `partial_decode` asks for, while one that cannot has a cache
    /// inserted by `CodecChain`, so the first call fetches and inflates the chunk and every later
    /// call is served from that. Reusing one decoder for all the regions in a chunk is therefore
    /// what keeps a compressed chunk to a single fetch and a single inflate.
    pub fn subchunk_decoder(
        &self,
        store: &ReadableWritableListableStorage,
        key: &StoreKey,
        // Where the chunk sits within the shard, as `(offset, length)`.
        range: (u64, u64),
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> PyResult<Arc<dyn ArrayPartialDecoderTraits>> {
        let storage_handle = Arc::new(StorageHandle::new(store.clone()));
        let shard = Arc::new(StoragePartialDecoder::new(storage_handle, key.clone()));
        self.inner_codecs
            .clone()
            .partial_decoder(
                Arc::new(ByteIntervalPartialDecoder::new(shard, range.0, range.1)),
                &self.inner_chunk_shape,
                data_type,
                fill_value,
                options,
            )
            .map_codec_err()
    }

    /// A decoder for one inner chunk whose bytes have already been read.
    ///
    /// The same construction as [`Self::subchunk_decoder`], differing only in where the bytes come
    /// from, so a chunk that arrived in a merged read is decoded by the same machinery as one read
    /// on its own — including the cache `CodecChain` inserts when the chain cannot partial-read,
    /// which is what keeps a compressed chunk to one inflate however many regions want it.
    pub fn fetched_subchunk_decoder(
        &self,
        bytes: Arc<Bytes>,
        offset: usize,
        length: usize,
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> PyResult<Arc<dyn ArrayPartialDecoderTraits>> {
        self.inner_codecs
            .clone()
            .partial_decoder(
                Arc::new(FetchedChunk {
                    bytes,
                    offset,
                    length,
                }),
                &self.inner_chunk_shape,
                data_type,
                fill_value,
                options,
            )
            .map_codec_err()
    }
}

/// One inner chunk's slice of a read that has already landed.
struct FetchedChunk {
    bytes: Arc<Bytes>,
    offset: usize,
    length: usize,
}

impl FetchedChunk {
    fn chunk(&self) -> &[u8] {
        &self.bytes[self.offset..self.offset + self.length]
    }
}

impl BytesPartialDecoderTraits for FetchedChunk {
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.length
    }

    /// The bytes are already here, so any region of them can be served without reading more.
    fn supports_partial_decode(&self) -> bool {
        true
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        let chunk = self.chunk();
        let size = self.length as u64;
        decoded_regions
            .map(|byte_range| {
                let start = usize::try_from(byte_range.start(size))
                    .map_err(|err| CodecError::Other(err.to_string()))?;
                let end = usize::try_from(byte_range.end(size))
                    .map_err(|err| CodecError::Other(err.to_string()))?;
                chunk
                    .get(start..end)
                    .map(Cow::Borrowed)
                    .ok_or_else(|| InvalidByteRangeError::new(byte_range, size).into())
            })
            .collect::<Result<Vec<_>, CodecError>>()
            .map(Some)
    }
}
