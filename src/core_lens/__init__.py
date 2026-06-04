try:
    from ._version import __version__  # type: ignore[import-not-found]
except ImportError:
    __version__ = "0.1.0"

from core_lens.aoi import AoI, SeasonConfig

__all__ = ["__version__", "AoI", "SeasonConfig"]
