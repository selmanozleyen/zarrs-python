use pyo3::PyResult;
use zarrs::array::{
    ArrayCodecTraits, CodecOptions, RecommendedConcurrency,
    concurrency::calc_concurrency_outer_inner,
};

use crate::{CodecPipelineImpl, chunk_item::ChunkItem, utils::PyCodecErrExt as _};

pub trait ChunkConcurrentLimitAndCodecOptions {
    fn get_chunk_concurrent_limit_and_codec_options(
        &self,
        codec_pipeline_impl: &CodecPipelineImpl,
    ) -> PyResult<Option<(usize, CodecOptions)>>;
}

// For the slice, not the Vec, which is the more general target and costs nothing -- `Vec` still
// reaches it by deref. The original reason was that the read path passed a slice out of the
// `ChunkItems` handle; that path no longer calls this at all (see `lib.rs`, where the read uses
// the pipeline's own `CodecOptions`), so the only caller left is the WRITE path, with a `Vec`.
impl ChunkConcurrentLimitAndCodecOptions for [ChunkItem] {
    fn get_chunk_concurrent_limit_and_codec_options(
        &self,
        codec_pipeline_impl: &CodecPipelineImpl,
    ) -> PyResult<Option<(usize, CodecOptions)>> {
        let num_chunks = self.len();
        let Some(item) = self.first() else {
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
}
