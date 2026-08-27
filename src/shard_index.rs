//! Where an innermost chunk lives inside its shard, however many levels of sharding are
//! between them.
//!
//! This module used to parse the shard index by hand: derive the encoded index size from the
//! index codecs, read it from the start or end per `index_location`, decode it, and pick the
//! offset/size pair out — including the `u64::MAX, u64::MAX` marker for a chunk never written.
//! All of that duplicated zarrs and existed only because the public API hung off `Array`,
//! which needs a node path `CodecPipeline.from_array_metadata_and_store` is never given.
//! zarrs 0.24 made `ShardingPartialDecoder` public with a `(storage, key)` handle instead.
//!
//! What is left is what is genuinely ours: the LEVELS of sharding an array has, and how an
//! element offset becomes a subchunk index at each one.
//!
//! Nested sharding puts a shard inside a shard. The innermost chunk stays the decode unit —
//! treating a subshard as the unit would decode many innermost chunks to keep the elements of
//! one, which is the amplification this whole path exists to avoid. So locating one chunk
//! walks one index per level: the outer index gives the subshard's extent, and the subshard's
//! own index, which lives inside that extent, gives the chunk's.

use std::sync::Arc;

use pyo3::PyResult;
use zarrs::array::codec::array_to_bytes::sharding::{
    ShardingCodecBound, ShardingCodecOptions, ShardingPartialDecoder,
};
use zarrs::array::{BytesPartialDecoderTraits, ChunkShape, CodecChainBound, CodecOptions};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;

use crate::utils::PyCodecErrExt as _;

/// One level of sharding: what it divides into, and how its offset/size table is stored.
struct Level {
    subchunk_shape: ChunkShape,
    index_codecs: Arc<CodecChainBound>,
    index_location: ShardingIndexLocation,
    sharding_options: ShardingCodecOptions,
}

/// What the pool needs to know about a sharded array to fetch one innermost chunk at a time.
pub(crate) struct ShardInfo {
    /// Outermost first. Exactly one entry for a singly sharded array, which is the case the
    /// hot path is tuned for: one iteration, one index, one cache lookup.
    levels: Vec<Level>,
    /// The INNERMOST chunk shape — the unit the codec chain decodes, and what the pool sizes
    /// its scratch buffer by.
    pub subchunk_shape: ChunkShape,
    /// The codecs that decode an innermost chunk, bound to the array's data type and fill
    /// value. This is the only chain that legitimately holds other codecs (blosc and so on);
    /// every chain above it must be exclusively sharded.
    pub inner_chain: Arc<CodecChainBound>,
}

impl ShardInfo {
    /// Read off the array's BOUND codec chain, or `None` if this array is not sharded, or is
    /// sharded in a way this path refuses.
    ///
    /// Descends while the chain's array-to-bytes codec is sharding, collecting a level each
    /// time. Refuses a chain that shards AND carries another codec beside it: the bytes on
    /// disk would then not be the shard its index describes — a bytes-to-bytes codec after
    /// sharding compresses the whole shard, so a byte range addresses compressed bytes, and
    /// an array-to-array codec before it reshapes what the inner chunks hold. Either way the
    /// index still parses and a read still returns something, which is why this is checked
    /// rather than assumed. zarrs calls the property `is_exclusively_sharded`, but that
    /// predicate hangs off `Array`.
    pub fn from_codec_chain(chain: &Arc<CodecChainBound>) -> Option<Self> {
        let mut levels: Vec<Level> = Vec::new();
        let mut current = chain.clone();
        loop {
            let step = {
                let sharding = current
                    .array_to_bytes_codec()
                    .as_any()
                    .downcast_ref::<ShardingCodecBound>();
                let Some(sharding) = sharding else { break };
                if !current.array_to_array_codecs().is_empty()
                    || !current.bytes_to_bytes_codecs().is_empty()
                {
                    return None;
                }
                (
                    Level {
                        subchunk_shape: sharding.subchunk_shape().clone(),
                        index_codecs: sharding.index_codecs().clone(),
                        index_location: sharding.index_location(),
                        sharding_options: sharding.options().clone(),
                    },
                    sharding.inner_codecs().clone(),
                )
            };
            levels.push(step.0);
            current = step.1;
        }
        let subchunk_shape = levels.last()?.subchunk_shape.clone();
        Some(Self {
            levels,
            subchunk_shape,
            inner_chain: current,
        })
    }

    /// How many levels of sharding this array has. One is the ordinary case.
    pub fn depth(&self) -> usize {
        self.levels.len()
    }

    /// What the level at `depth` divides into — the shard shape for the level below it.
    pub fn subchunk_shape_at(&self, depth: usize) -> &ChunkShape {
        &self.levels[depth].subchunk_shape
    }

    /// A partial decoder for one level of one shard.
    ///
    /// Constructing it READS that level's index, a full-latency round trip on the calling
    /// thread — the cost the two-phase arrangement here is built around, and the reason these
    /// are remembered for the life of a read-only array.
    ///
    /// `input` reads the bytes this level occupies: the store key itself at depth 0, or a byte
    /// interval of it below that. `shard_shape` is what this level divides — the array's chunk
    /// shape at depth 0, the level above's subchunk shape below that.
    pub fn level_decoder(
        &self,
        depth: usize,
        input: Arc<dyn BytesPartialDecoderTraits>,
        shard_shape: ChunkShape,
        options: &CodecOptions,
    ) -> PyResult<ShardingPartialDecoder> {
        let level = &self.levels[depth];
        ShardingPartialDecoder::new(
            input,
            shard_shape,
            level.subchunk_shape.clone(),
            self.inner_chain.clone(),
            &level.index_codecs,
            level.index_location,
            options,
            level.sharding_options.clone(),
        )
        .map_codec_err()
    }
}
