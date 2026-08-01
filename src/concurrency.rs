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

/// How many chunk items to process at once.
///
/// The usual recommendation is a wide range whose *minimum* is small
/// (`chunk_concurrent_minimum`, 4 by default). `calc_concurrency_outer_inner`
/// starts from that minimum and spends everything left over on the codec, which
/// is the right split for a subset that spans many inner chunks: the sharding
/// decoder parallelises across them and uses the budget.
///
/// A scattered item instead exposes explicit decode-granule groups (inner
/// chunks for sharding). Measured on a shuffled CSR minibatch (16 threads,
/// ~126 items), the old 4-item/4-codec split reached only 2.5 runnable threads
/// because most items had too few groups to spend four threads effectively.
///
/// So for scattered items ask for the items themselves to run concurrently and
/// pin the recommendation to that one value. When there are at least as many
/// items as threads this leaves the codec a target of 1, which is what a
/// single-inner-chunk decode wants. When there are fewer items than threads the
/// leftover is spent across the item's decode-granule groups first, then passed
/// into the codec if there are still too few groups. Thus two items with four
/// groups each can still expose eight independent decodes without hard-coding
/// either the shard or inner-chunk level as the only source of parallelism.
fn recommended_outer_concurrency(
    scattered: bool,
    num_chunks: usize,
    codec_pipeline_impl: &CodecPipelineImpl,
) -> RecommendedConcurrency {
    let floor = std::cmp::min(codec_pipeline_impl.chunk_concurrent_minimum, num_chunks);
    if scattered {
        let n = std::cmp::max(
            std::cmp::min(codec_pipeline_impl.num_threads, num_chunks),
            floor,
        );
        // Start and end are read directly by `calc_concurrency_outer_inner`;
        // pinning both to `n` fixes the outer limit at `n`.
        RecommendedConcurrency::new(n..n)
    } else {
        let ceiling = std::cmp::max(codec_pipeline_impl.chunk_concurrent_maximum, num_chunks);
        RecommendedConcurrency::new(floor..ceiling)
    }
}

impl ChunkConcurrentLimitAndCodecOptions for Vec<ChunkItem> {
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
            .recommended_concurrency(&item.shape, &codec_pipeline_impl.data_type)
            .map_codec_err()?;

        // A batch is built by a single zarr indexer, so it is either all
        // scattered or none of it is. `any` rather than `all` in case that ever
        // stops holding: the scattered split is the safe one for a mixed batch,
        // since it still gives every item a thread.
        let scattered = self.iter().any(|item| item.chunk_indices.is_some());
        let (chunk_concurrent_limit, codec_concurrent_limit) = calc_concurrency_outer_inner(
            codec_pipeline_impl.num_threads,
            &recommended_outer_concurrency(scattered, num_chunks, codec_pipeline_impl),
            &codec_concurrency,
        );
        let codec_options = codec_pipeline_impl
            .codec_options
            .with_concurrent_target(codec_concurrent_limit);
        Ok(Some((chunk_concurrent_limit, codec_options)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the sharding codec on the annbatch geometry: a shard holds
    /// ~1000 inner chunks, so the codec will happily take any target offered.
    fn sharded_codec_concurrency() -> RecommendedConcurrency {
        RecommendedConcurrency::new_maximum(1000)
    }

    fn split(scattered: bool, num_chunks: usize, num_threads: usize) -> (usize, usize) {
        let outer = {
            let floor = std::cmp::min(4, num_chunks); // chunk_concurrent_minimum default
            if scattered {
                let n = std::cmp::max(std::cmp::min(num_threads, num_chunks), floor);
                RecommendedConcurrency::new(n..n)
            } else {
                RecommendedConcurrency::new(floor..std::cmp::max(num_threads, num_chunks))
            }
        };
        calc_concurrency_outer_inner(num_threads, &outer, &sharded_codec_concurrency())
    }

    #[test]
    fn scattered_items_get_the_threads() {
        // The regression this exists for: 126 scattered items on 16 threads
        // used to run 4 at a time with a codec target of 4 that no
        // single-inner-chunk decode could spend.
        assert_eq!(split(false, 126, 16), (4, 4));
        assert_eq!(split(true, 126, 16), (16, 1));
    }

    #[test]
    fn few_scattered_items_hand_the_rest_back_to_the_codec() {
        // Fewer items than threads: the outer limit cannot absorb the budget,
        // so it flows to the codec rather than being lost.
        assert_eq!(split(true, 8, 16), (8, 2));
        assert_eq!(split(true, 2, 16), (2, 8));
        assert_eq!(split(true, 1, 16), (1, 16));
    }

    #[test]
    fn single_thread_stays_serial() {
        assert_eq!(split(true, 126, 1), (4, 1));
    }
}
