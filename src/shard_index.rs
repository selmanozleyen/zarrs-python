//! Locating one innermost chunk inside a shard.
//!
//! This module used to parse the shard index by hand: work out the encoded index size from
//! the index codecs, read it from the start or the end of the shard depending on
//! `index_location`, decode it, and pick the offset/size pair out of it -- including the
//! `u64::MAX, u64::MAX` marker that means the chunk was never written. All of that duplicated
//! zarrs, and it existed only because the public API for it hung off `Array`, which needs a
//! node path that `CodecPipeline.from_array_metadata_and_store` is never given.
//!
//! zarrs 0.24 made `ShardingPartialDecoder` public with a `new` that takes a `(storage, key)`
//! handle instead of an `Array`, so the duplication is gone. What is left here is the part
//! that is genuinely ours: which codec chain decodes an inner chunk, and how a 1-D element
//! offset becomes a subchunk index.

use std::sync::Arc;

use pyo3::PyResult;
use zarrs::array::codec::array_to_bytes::sharding::{
    ShardingCodecBound, ShardingCodecOptions, ShardingPartialDecoder,
};
use zarrs::array::{ChunkShape, CodecChainBound, CodecOptions};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::{ReadableWritableListableStorage, StoreKey};

use crate::utils::{PyCodecErrExt as _, key_partial_decoder};

/// What the pool needs to know about a sharded array to fetch one inner chunk at a time.
pub(crate) struct ShardInfo {
    /// The innermost chunk shape -- the unit the codec chain decodes.
    pub subchunk_shape: ChunkShape,
    /// The codecs *inside* a shard, which decode one innermost chunk, bound to the array's
    /// data type and fill value.
    pub inner_chain: Arc<CodecChainBound>,
    index_codecs: Arc<CodecChainBound>,
    index_location: ShardingIndexLocation,
    sharding_options: ShardingCodecOptions,
}

impl ShardInfo {
    /// Read off the array's BOUND codec chain, or `None` if this array is not singly sharded.
    ///
    /// The bound chain already holds the sharding codec with its inner and index chains bound
    /// to the right data types, which is why this takes the chain rather than the metadata:
    /// the alternative was re-deriving all of it from `MetadataV3`.
    pub fn from_codec_chain(chain: &CodecChainBound) -> Option<Self> {
        // Only an EXCLUSIVELY sharded array. A codec either side of the sharding codec means
        // the shard's bytes on disk are not the shard the index describes: a bytes-to-bytes
        // codec after it compresses the whole shard, so a byte range into the file addresses
        // compressed bytes, and an array-to-array codec before it reshapes what the inner
        // chunks contain. Either way the index still parses and the read still returns
        // something, which is exactly why this is checked rather than assumed. zarrs calls
        // this `is_exclusively_sharded`, but that predicate hangs off `Array`.
        if !chain.array_to_array_codecs().is_empty() || !chain.bytes_to_bytes_codecs().is_empty() {
            return None;
        }
        let sharding = chain
            .array_to_bytes_codec()
            .as_any()
            .downcast_ref::<ShardingCodecBound>()?;
        // Nested sharding: the inner chunk shape would name a subshard rather than the
        // innermost chunk. Refuse rather than group at the wrong granularity.
        if sharding
            .inner_codecs()
            .array_to_bytes_codec()
            .as_any()
            .downcast_ref::<ShardingCodecBound>()
            .is_some()
        {
            return None;
        }
        Some(Self {
            subchunk_shape: sharding.subchunk_shape().clone(),
            inner_chain: sharding.inner_codecs().clone(),
            index_codecs: sharding.index_codecs().clone(),
            index_location: sharding.index_location(),
            sharding_options: sharding.options().clone(),
        })
    }

    /// A partial decoder for one shard.
    ///
    /// Constructing it READS that shard's index, which is a full-latency round trip on the
    /// calling thread -- the cost the whole two-phase arrangement here is built around, and
    /// the reason these are remembered for the life of a read-only array.
    pub fn decoder(
        &self,
        store: &ReadableWritableListableStorage,
        key: &StoreKey,
        shard_shape: ChunkShape,
        options: &CodecOptions,
    ) -> PyResult<ShardingPartialDecoder> {
        // Reads the key directly: no `Array`, hence no node path, which is the whole reason
        // the deletion described above was possible.
        ShardingPartialDecoder::new(
            key_partial_decoder(store, key),
            shard_shape,
            self.subchunk_shape.clone(),
            self.inner_chain.clone(),
            &self.index_codecs,
            self.index_location,
            options,
            self.sharding_options.clone(),
        )
        .map_codec_err()
    }

    /// The index of the innermost chunk holding element `start`.
    ///
    /// 1-D only, which is what the chunk-unit path accepts.
    pub fn subchunk_index_1d(&self, start: u64) -> u64 {
        start / self.subchunk_shape[0].get()
    }
}
