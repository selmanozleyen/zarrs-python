import os

from ._internal import __version__, release_pools_for_fork
from .pipeline import ZarrsCodecPipeline as _ZarrsCodecPipeline
from .utils import DiscontiguousArrayError, UnsupportedVIndexingError


# Need to do this redirection so people can access the pipeline as `zarrs.ZarrsCodecPipeline` instead of `zarrs.pipeline.ZarrsCodecPipeline`
class ZarrsCodecPipeline(_ZarrsCodecPipeline):
    pass


# A FORK MUST NOT INHERIT THE WORKER POOLS. They are process-wide, and `fork()` copies their
# memory but only the calling thread -- so a child's first read parks on workers that do not
# exist. Emptying the slot beforehand means both sides rebuild on next use.
#
# `register_at_fork` covers `os.fork` and `multiprocessing`, which is what
# `torch.utils.data.DataLoader(num_workers > 0)` uses. Guarded because it is POSIX-only.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=release_pools_for_fork)


__all__ = [
    "ZarrsCodecPipeline",
    "DiscontiguousArrayError",
    "UnsupportedVIndexingError",
    "__version__",
]
