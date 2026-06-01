try:
    from ._version import __version__  # type: ignore[import-not-found,  attr-defined]
except ImportError:
    __version__ = "0.1.0"

_all__ = ["__version__"]
