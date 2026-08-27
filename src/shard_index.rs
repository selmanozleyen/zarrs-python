//! Where an innermost chunk lives inside its shard, read directly.
//!
//! The reader pool needs a chunk's byte extent *without* decoding it, and the released
//! crate's `partial_decode` will only do both at once. So this reads the shard index
//! itself: the offset/size table sharding already writes at the start or end of every
//! shard. Two codec chains come out of the array metadata -- one for the index, one for the
//! inner chunks -- and both are built with the crate's own `CodecChain`, so nothing here
//! reimplements a codec. What is reimplemented is only the arithmetic the crate keeps
//! private: which bytes the index occupies, and which pair of u64s belongs to a chunk.
//!
//! This is what lets the pool run against the RELEASED zarrs rather than a patched one.
//!
//! Refused on purpose (both fall back to the fused path rather than being measured wrong):
//!
//! - **Nested sharding.** If the inner codec list shards again, `chunk_shape` is not the
//!   innermost chunk, and one item would span several decode units -- the grouping and the
//!   dedup claim would both quietly stop holding.
//! - **A non-fixed index size.** The index has to be at a known extent to be read without
//!   reading the shard.

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use pyo3::PyResult;
use pyo3::exceptions::{PyRuntimeError, PyTypeError};
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecConfiguration;
use zarrs::array::data_type::uint64;
use zarrs::array::{
    ArrayBytes, ArrayToBytesCodecTraits, BytesRepresentation, CodecChain, CodecChainBound,
    CodecOptions, DataType, FillValue,
};
use zarrs::metadata::v3::MetadataV3;
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{ReadableWritableListableStorage, StoreKey};

use crate::utils::{PyCodecErrExt as _, PyErrExt as _};

/// An absent inner chunk: sharding writes this pair instead of an offset and a size.
const ABSENT: u64 = u64::MAX;

/// What the pool needs to know about a sharded array to fetch one chunk at a time.
pub(crate) struct ShardInfo {
    /// The innermost chunk shape -- the unit the codec chain decodes.
    pub inner_shape: Vec<NonZeroU64>,
    /// The codecs *inside* a shard, which decode one innermost chunk, bound to the array's
    /// data type and fill value.
    pub inner_chain: Arc<CodecChainBound>,
    /// The codecs the offset/size table is encoded with, bound to u64 and the absent marker.
    index_chain: Arc<CodecChainBound>,
    index_location: ShardingIndexLocation,
}

impl ShardInfo {
    /// Read the sharding codec out of the array metadata, or `None` if this array is not
    /// singly sharded.
    /// `data_type` and `fill_value` are the ARRAY's: a codec chain is unbound until it is
    /// told what it decodes into, and the chain inside a shard decodes the array's elements.
    /// The index chain binds to u64 and the absent marker instead, which are fixed by the
    /// sharding codec rather than by the array.
    pub fn from_codecs(
        codecs: &[MetadataV3],
        data_type: &DataType,
        fill_value: &FillValue,
    ) -> PyResult<Option<Self>> {
        for codec in codecs {
            if !is_sharding(codec) {
                continue;
            }
            let configuration: ShardingCodecConfiguration =
                codec.to_configuration().map_py_err::<PyTypeError>()?;
            let ShardingCodecConfiguration::V1(configuration) = configuration else {
                return Ok(None);
            };
            // Nested sharding: `chunk_shape` would name a subshard, not the innermost
            // chunk. Refuse rather than group at the wrong granularity.
            if configuration.codecs.iter().any(is_sharding) {
                return Ok(None);
            }
            let inner_chain = CodecChain::from_metadata(&configuration.codecs)
                .map_py_err::<PyTypeError>()?
                .with_context(data_type.clone(), fill_value.clone())
                .map_py_err::<PyTypeError>()?;
            let index_chain = CodecChain::from_metadata(&configuration.index_codecs)
                .map_py_err::<PyTypeError>()?
                .with_context(uint64(), FillValue::from(ABSENT))
                .map_py_err::<PyTypeError>()?;
            return Ok(Some(Self {
                inner_shape: configuration.chunk_shape.clone(),
                inner_chain,
                index_chain,
                index_location: configuration.index_location,
            }));
        }
        Ok(None)
    }

    /// How many innermost chunks a shard of `shard_shape` holds, per dimension.
    pub fn chunks_per_shard(&self, shard_shape: &[NonZeroU64]) -> PyResult<Vec<NonZeroU64>> {
        if shard_shape.len() != self.inner_shape.len() {
            return Err(PyRuntimeError::new_err(
                "shard shape and inner chunk shape disagree on dimensionality",
            ));
        }
        shard_shape
            .iter()
            .zip(&self.inner_shape)
            .map(|(shard, inner)| {
                NonZeroU64::new(shard.get().div_ceil(inner.get()))
                    .ok_or_else(|| PyRuntimeError::new_err("a shard with no inner chunks"))
            })
            .collect()
    }

    /// The bytes the encoded index occupies. Computed the way the crate computes it, from
    /// the index codecs applied to a `chunks_per_shard + [2]` array of u64.
    fn index_encoded_size(&self, chunks_per_shard: &[NonZeroU64]) -> PyResult<u64> {
        let mut index_shape = chunks_per_shard.to_vec();
        index_shape.push(NonZeroU64::new(2).expect("2 is not zero"));
        let representation = self
            .index_chain
            .encoded_representation(&index_shape)
            .map_codec_err()?;
        match representation {
            BytesRepresentation::FixedSize(size) => Ok(size),
            // Without a fixed extent the index cannot be located without reading the shard,
            // which is the whole thing this avoids.
            BytesRepresentation::BoundedSize(_) | BytesRepresentation::UnboundedSize => Err(
                PyRuntimeError::new_err("the shard index does not have a fixed encoded size"),
            ),
        }
    }

    /// Read and decode one shard's offset/size table.
    ///
    /// Two u64 per innermost chunk, in the shard's chunk order. Costs one read of the index
    /// extent, not of the shard.
    pub fn read_index(
        &self,
        store: &ReadableWritableListableStorage,
        key: &StoreKey,
        chunks_per_shard: &[NonZeroU64],
        options: &CodecOptions,
    ) -> PyResult<Option<Vec<u64>>> {
        let size = self.index_encoded_size(chunks_per_shard)?;
        let range = match self.index_location {
            ShardingIndexLocation::Start => ByteRange::FromStart(0, Some(size)),
            ShardingIndexLocation::End => ByteRange::Suffix(size),
        };
        let Some(encoded) = store
            .get_partial(key, range)
            .map_py_err::<PyRuntimeError>()?
        else {
            return Ok(None); // no shard at all: every chunk in it is absent
        };

        let mut index_shape = chunks_per_shard.to_vec();
        index_shape.push(NonZeroU64::new(2).expect("2 is not zero"));
        let decoded = self
            .index_chain
            .decode(Cow::Owned(encoded.into()), &index_shape, options)
            .map_codec_err()?;
        let ArrayBytes::Fixed(raw) = decoded else {
            return Err(PyTypeError::new_err(
                "the shard index decoded as variable length",
            ));
        };
        if raw.len() % 8 != 0 {
            return Err(PyRuntimeError::new_err(
                "the decoded shard index is not a whole number of u64",
            ));
        }
        Ok(Some(
            raw.chunks_exact(8)
                .map(|b| u64::from_le_bytes(b.try_into().expect("chunks_exact(8)")))
                .collect(),
        ))
    }

    /// Where innermost chunk `linear` lives in the shard, or `None` if it is absent.
    pub fn chunk_range(index: &[u64], linear: usize) -> PyResult<Option<ByteRange>> {
        let (Some(&offset), Some(&size)) = (index.get(2 * linear), index.get(2 * linear + 1))
        else {
            return Err(PyRuntimeError::new_err(format!(
                "inner chunk {linear} is past the {} the shard index holds",
                index.len() / 2
            )));
        };
        if offset == ABSENT && size == ABSENT {
            return Ok(None);
        }
        Ok(Some(ByteRange::FromStart(offset, Some(size))))
    }

    /// The linear index of the innermost chunk starting at `start`, in element coordinates.
    ///
    /// 1-D only, which is what the chunk-unit path accepts.
    pub fn linear_index_1d(&self, start: u64) -> usize {
        usize::try_from(start / self.inner_shape[0].get()).unwrap_or(usize::MAX)
    }
}

fn is_sharding(codec: &MetadataV3) -> bool {
    // The registered name, plus the pre-registration alias some stores still carry.
    matches!(codec.name(), "sharding_indexed" | "zarrs.sharding_indexed")
}
