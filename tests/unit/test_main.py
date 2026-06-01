def test_version():
    from core_lens import __version__

    assert __version__ == "0.1.0"


def test_hello():
    from core_lens.__main__ import hello

    assert hello() == "Hello from core-lens!"
