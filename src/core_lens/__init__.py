try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"

from core_lens.aoi import AoI, SeasonConfig
from core_lens.entities.tehsil import TehsilEntity

__all__ = ["__version__", "AoI", "SeasonConfig", "TehsilEntity"]
