import polars as pl
from core_lens.utils.polars_utils import scan_with_key_filter
from typing import Any


def test_scan_with_key_filter_composite_key(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame({"id1": [1, 2], "id2": [3, 4], "val": ["A", "B"]}).write_parquet(p)

    key_values = pl.DataFrame({"id1": [2], "id2": [4]})
    lf = scan_with_key_filter(str(p), ["id1", "id2"], key_values)
    res = lf.collect()
    assert res.height == 1
    assert res["val"][0] == "B"


def test_collect_lf(tmp_path: Any, monkeypatch: Any) -> None:
    from core_lens.utils.polars_utils import collect_lf, _gpu_available
    import sys

    monkeypatch.setitem(sys.modules, "cudf_polars", None)

    # reset cache
    from core_lens.utils import polars_utils

    polars_utils._GPU_AVAILABLE = None

    p = tmp_path / "test.parquet"
    pl.DataFrame({"val": [1, 2, 3]}).write_parquet(p)
    lf = pl.scan_parquet(str(p))
    df = collect_lf(lf)
    assert df.height == 3
    assert not _gpu_available()


def test_gpu_available_caching(monkeypatch: Any) -> None:
    from core_lens.utils import polars_utils
    import sys

    monkeypatch.setitem(sys.modules, "cudf_polars", None)

    polars_utils._GPU_AVAILABLE = None
    assert not polars_utils._gpu_available()
    assert polars_utils._GPU_AVAILABLE is False


def test_gpu_compute_error(monkeypatch: Any) -> None:
    from core_lens.utils import polars_utils
    import polars as pl

    polars_utils._GPU_AVAILABLE = True

    original_collect = pl.LazyFrame.collect

    def mock_collect(self: Any, **kwargs: Any) -> Any:
        if kwargs.get("engine") != "streaming":
            raise pl.exceptions.ComputeError("CUDA out of memory")
        return original_collect(self, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", mock_collect)

    lf = pl.LazyFrame({"a": [1, 2, 3]})
    df = polars_utils.collect_lf(lf)
    assert df.height == 3
    assert polars_utils._GPU_AVAILABLE is False


def test_gpu_compute_error_other(monkeypatch: Any) -> None:
    from core_lens.utils import polars_utils
    import polars as pl

    polars_utils._GPU_AVAILABLE = True

    original_collect = pl.LazyFrame.collect

    def mock_collect(self: Any, **kwargs: Any) -> Any:
        if kwargs.get("engine") != "streaming":
            raise pl.exceptions.ComputeError("Other error")
        return original_collect(self, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", mock_collect)

    lf = pl.LazyFrame({"a": [1, 2, 3]})
    import pytest

    with pytest.raises(pl.exceptions.ComputeError, match="Other error"):
        polars_utils.collect_lf(lf)
