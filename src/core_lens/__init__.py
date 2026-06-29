try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"

from core_lens.aoi import AoI, SeasonConfig
from core_lens.entities.tehsil import TehsilEntity

from loguru import logger

logger.disable("core_lens")

__all__ = ["__version__", "AoI", "SeasonConfig", "TehsilEntity"]
