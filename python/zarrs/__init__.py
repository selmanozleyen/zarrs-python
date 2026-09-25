from ._internal import __version__
from .pipeline import UnsupportedRangeReadError
from .pipeline import ZarrsCodecPipeline as _ZarrsCodecPipeline
from .ranges import aread_ranges
from .utils import DiscontiguousArrayError, UnsupportedVIndexingError


# Need to do this redirection so people can access the pipeline as `zarrs.ZarrsCodecPipeline` instead of `zarrs.pipeline.ZarrsCodecPipeline`
class ZarrsCodecPipeline(_ZarrsCodecPipeline):
    pass


__all__ = [
    "ZarrsCodecPipeline",
    "UnsupportedRangeReadError",
    "aread_ranges",
    "DiscontiguousArrayError",
    "UnsupportedVIndexingError",
    "__version__",
]
