//! Where an innermost chunk lives inside its shard, through any number of sharding levels.

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

pub(crate) struct ShardInfo {
    /// Outermost first; empty when not sharded.
    levels: Vec<Level>,
    /// The innermost chunk shape, `None` when not sharded.
    pub subchunk_shape: Option<ChunkShape>,
    /// The codecs that decode an innermost chunk.
    pub inner_chain: Arc<CodecChainBound>,
}

impl ShardInfo {
    /// `None` if a sharding level has a codec beside it.
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

    pub fn depth(&self) -> usize {
        self.levels.len()
    }

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
