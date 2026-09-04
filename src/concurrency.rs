use pyo3::PyResult;
use zarrs::array::{
    ArrayCodecTraits, CodecOptions, RecommendedConcurrency,
    concurrency::calc_concurrency_outer_inner,
};

use crate::{CodecPipelineImpl, chunk_item::ChunkItem, utils::PyCodecErrExt as _};

/// The chunk concurrency and codec options for one WRITE batch.
///
/// A free function, not a trait. It was a trait with one implementation and one caller, which
/// is a name for an indirection nobody takes -- and the read path stopped calling it entirely
/// (see `lib.rs`, where a read uses the pipeline's own `CodecOptions`, because parallelism
/// there is the two pools rather than a chunk-at-a-time codec loop).
pub(crate) fn chunk_concurrency(
    items: &[ChunkItem],
    codec_pipeline_impl: &CodecPipelineImpl,
) -> PyResult<Option<(usize, CodecOptions)>> {
    let num_chunks = items.len();
    let Some(item) = items.first() else {
        return Ok(None);
    };

    let codec_concurrency = codec_pipeline_impl
        .codec_chain
        .recommended_concurrency(&item.shape)
        .map_codec_err()?;

    let min_concurrent_chunks =
        std::cmp::min(codec_pipeline_impl.chunk_concurrent_minimum, num_chunks);
    let max_concurrent_chunks =
        std::cmp::max(codec_pipeline_impl.chunk_concurrent_maximum, num_chunks);
    let (chunk_concurrent_limit, codec_concurrent_limit) = calc_concurrency_outer_inner(
        codec_pipeline_impl.num_threads,
        &RecommendedConcurrency::new(min_concurrent_chunks..max_concurrent_chunks),
        &codec_concurrency,
    );
    let codec_options = codec_pipeline_impl
        .codec_options
        .with_concurrent_target(codec_concurrent_limit);
    Ok(Some((chunk_concurrent_limit, codec_options)))
}
