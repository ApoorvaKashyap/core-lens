from . import __version__


def hello() -> str:
    return "Hello from core-lens!"


def version() -> str:
    return __version__


if __name__ == "__main__":
    print(hello())
    print(f"Version: {version()}")
