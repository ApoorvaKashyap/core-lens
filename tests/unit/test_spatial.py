import pytest
from typing import Any
import pathlib
import polars as pl
import shapely.geometry as sgeom
from core_lens.utils.spatial import (
    resolve_path,
    build_bbox_index,
    exact_spatial_filter,
)


def test_resolve_path_relative(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    p = pathlib.Path("some_file.txt")
    p.write_text("hello")
    assert resolve_path("some_file.txt") == str(tmp_path / "some_file.txt")


def test_resolve_path_not_found(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_path(str(tmp_path / "missing.txt"))


def test_build_bbox_index_with_bbox_cols(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame(
        {"id": [1], "minx": [0.0], "miny": [0.0], "maxx": [1.0], "maxy": [1.0]}
    ).write_parquet(p)
    df = build_bbox_index(
        str(p), ["id"], ("minx", "miny", "maxx", "maxy"), "geom", "latlon"
    )
    assert df.columns == ["id", "minx", "miny", "maxx", "maxy"]


def test_build_bbox_index_latlon_error() -> None:
    with pytest.raises(ValueError, match="geometry_type='latlon' requires bbox_cols"):
        build_bbox_index("path", ["id"], None, "geom", "latlon")


def test_build_bbox_index_wkt(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame({"id": [1], "geom": ["POINT (0.5 0.5)"]}).write_parquet(p)
    df = build_bbox_index(str(p), ["id"], None, "geom", "wkt")
    assert df.columns == ["id", "minx", "miny", "maxx", "maxy"]


def test_exact_spatial_filter_empty(tmp_path: Any) -> None:
    candidates = pl.DataFrame({"id": []}, schema={"id": pl.Int64})
    res = exact_spatial_filter(
        candidates, "path", ["id"], "geom", "wkb", sgeom.Point(0, 0)
    )
    assert res.is_empty()


def test_exact_spatial_filter_composite_key(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame({"id1": [1], "id2": [2], "geom": ["POINT (0.5 0.5)"]}).write_parquet(p)
    candidates = pl.DataFrame({"id1": [1], "id2": [2]})
    res = exact_spatial_filter(
        candidates, str(p), ["id1", "id2"], "geom", "wkt", sgeom.box(0, 0, 1, 1)
    )
    assert res.height == 1


def test_exact_spatial_filter_area(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame(
        {
            "id": [1, 2],
            "geom": ["POINT (0.5 0.5)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    ).write_parquet(p)
    candidates = pl.DataFrame({"id": [1, 2]})
    res = exact_spatial_filter(
        candidates,
        str(p),
        ["id"],
        "geom",
        "wkt",
        sgeom.box(0, 0, 0.5, 1.0),
        relationship="area",
        threshold=0.1,
    )
    # The polygon area is 1.0. Intersection with box(0, 0, 0.5, 1.0) is area 0.5.
    # 0.5 / 1.0 = 0.5 >= 0.1 threshold, so it matches. Point doesn't have area.
    assert res.height == 1
    assert res["id"][0] == 2


def test_execute_spatial_join(tmp_path: Any) -> None:
    from core_lens.utils.spatial import execute_spatial_join
    from core_lens.schema.profile import SchemaProfile

    class DummyOther:
        static_path = str(tmp_path / "other.parquet")
        key_cols = ["oid"]
        schema_profile = SchemaProfile(
            key_cols=["oid"],
            geometry_col="geom",
            geometry_type="wkt",
            annual_time_col=None,
            fortnightly_time_col=None,
            bbox_cols=None,
        )

        def _resolve(self, path: str) -> str:
            return path

    pl.DataFrame(
        {
            "oid": [1, 2],
            "val": [10.0, 20.0],
            "geom": [
                "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
            ],
        }
    ).write_parquet(DummyOther.static_path)

    primary = pl.DataFrame(
        {"pid": [1], "geom": ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"]}
    )

    res = execute_spatial_join(
        primary,
        ["pid"],
        "geom",
        "wkt",
        DummyOther(),
        agg={"val": "mean", "count": "count", "area": "area"},
        other_entity_name="other",
    )
    assert res.height == 1
    assert res["other_val"][0] == 15.0
    assert res["other_count"][0] == 2
    assert res["other_area"][0] == 2.0
