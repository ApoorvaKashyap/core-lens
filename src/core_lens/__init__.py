try:
    from ._version import __version__
except ImportError:
    __version__ = "0.1.0"


def hello() -> str:
    return "Hello from core-lens!"
