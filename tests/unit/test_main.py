def test_version() -> None:
    from core_lens import __version__

    assert __version__ == "0.1.0"


def test_hello() -> None:
    from core_lens.__main__ import hello

    assert hello() == "Hello from core-lens!"


def test_version_type() -> None:
    from core_lens.__main__ import version

    assert isinstance(version(), str)


def test_main_execution() -> None:
    import subprocess
    import sys
    import pathlib

    pkg_dir = pathlib.Path(__file__).parent.parent.parent / "src"
    result = subprocess.run(
        [sys.executable, "-m", "core_lens"],
        cwd=str(pkg_dir),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Hello from core-lens!" in result.stdout
