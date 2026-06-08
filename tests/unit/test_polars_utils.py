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
