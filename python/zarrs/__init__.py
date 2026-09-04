import os

from ._internal import __version__, pool_sizes, release_pools_for_fork
from .pipeline import (
    UnsupportedDataTypeError,
    UnsupportedMetadataError,
    ZarrsCodecPipeline as _ZarrsCodecPipeline,
    read_stats,
)
from .utils import (
    DiscontiguousArrayError,
    FillValueNoneError,
    UnsupportedVIndexingError,
)


# Need to do this redirection so people can access the pipeline as `zarrs.ZarrsCodecPipeline` instead of `zarrs.pipeline.ZarrsCodecPipeline`
class ZarrsCodecPipeline(_ZarrsCodecPipeline):
    pass


# A FORK MUST NOT INHERIT A HELD LOCK. The two read pools are process-wide, and the check that
# rebuilds them in a child runs *inside* the mutex guarding them -- so a child forked while any
# thread was inside that critical section blocks on a lock whose owner does not exist, for
# ever. Emptying the slot before the fork means both sides rebuild on next use.
#
# `register_at_fork` covers `os.fork` and `multiprocessing`, which is what
# `torch.utils.data.DataLoader(num_workers > 0)` uses -- the workload this read path exists for.
# Guarded because it is POSIX-only; on Windows there is no fork to hook.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=release_pools_for_fork)


__all__ = [
    "ZarrsCodecPipeline",
    # Every member of `FALLBACK_TO_ZARR_PYTHON`, because under `codec_pipeline.strict` that is
    # the set that can reach user code -- and `except zarrs.UnsupportedDataTypeError` could not
    # name three of the five. `FillValueNoneError` is currently raised NOWHERE, so it reaches
    # nobody today; it is exported with the others because it is in that tuple, and a caller
    # writing `except` against the set should not have to know which members are live.
    "DiscontiguousArrayError",
    "FillValueNoneError",
    "UnsupportedDataTypeError",
    "UnsupportedMetadataError",
    "UnsupportedVIndexingError",
    # Ask what happened, rather than infer it from a throughput number. `read_stats` says
    # whether a read was served here or handed to zarr-python; `pool_sizes` says what the two
    # pools were actually built with, which is what the "was ignored" warning tells the caller
    # to check.
    "pool_sizes",
    "read_stats",
    "__version__",
]
