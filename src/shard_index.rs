//! Where an innermost chunk lives inside its shard, however many levels of sharding are
//! between them.

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

/// What the read path needs to know about a sharded array to fetch one innermost chunk at
/// a time.
pub(crate) struct ShardInfo {
    /// Outermost first. Exactly one entry for a singly sharded array, which is the case the
    /// hot path is tuned for: one iteration, one index, one cache lookup.
    levels: Vec<Level>,
    /// The innermost chunk shape, what a job sizes its scratch by. `None` when not sharded:
    /// the decode unit is the chunk, which only an item knows.
    pub subchunk_shape: Option<ChunkShape>,
    /// The codecs that decode an innermost chunk, bound to the data type and fill value. The
    /// only chain that may hold other codecs; every chain above it must be exclusively sharded.
    pub inner_chain: Arc<CodecChainBound>,
}

impl ShardInfo {
    /// Read off the array's bound codec chain, or `None` if this array is sharded in a way this
    /// path refuses.
    ///
    /// A non-sharded array is accepted with no levels: its chunk is its own decode unit.
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
        let subchunk_shape = levels.last().map(|level| level.subchunk_shape.clone());
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
