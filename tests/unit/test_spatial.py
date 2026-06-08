import pytest
import pathlib
import polars as pl
import shapely.geometry as sgeom
from core_lens.utils.spatial import (
    resolve_path,
    build_bbox_index,
    exact_spatial_filter,
)


def test_resolve_path_relative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = pathlib.Path("some_file.txt")
    p.write_text("hello")
    assert resolve_path("some_file.txt") == str(tmp_path / "some_file.txt")


def test_resolve_path_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_path(str(tmp_path / "missing.txt"))


def test_build_bbox_index_with_bbox_cols(tmp_path):
    p = tmp_path / "test.parquet"
    pl.DataFrame(
        {"id": [1], "minx": [0.0], "miny": [0.0], "maxx": [1.0], "maxy": [1.0]}
    ).write_parquet(p)
    df = build_bbox_index(
        str(p), ["id"], ("minx", "miny", "maxx", "maxy"), "geom", "latlon"
    )
    assert df.columns == ["id", "minx", "miny", "maxx", "maxy"]


def test_build_bbox_index_latlon_error():
    with pytest.raises(ValueError, match="geometry_type='latlon' requires bbox_cols"):
        build_bbox_index("path", ["id"], None, "geom", "latlon")


def test_build_bbox_index_wkt(tmp_path):
    p = tmp_path / "test.parquet"
    pl.DataFrame({"id": [1], "geom": ["POINT (0.5 0.5)"]}).write_parquet(p)
    df = build_bbox_index(str(p), ["id"], None, "geom", "wkt")
    assert df.columns == ["id", "minx", "miny", "maxx", "maxy"]


def test_exact_spatial_filter_empty(tmp_path):
    candidates = pl.DataFrame({"id": []}, schema={"id": pl.Int64})
    res = exact_spatial_filter(
        candidates, "path", ["id"], "geom", "wkb", sgeom.Point(0, 0)
    )
    assert res.is_empty()


def test_exact_spatial_filter_composite_key(tmp_path):
    p = tmp_path / "test.parquet"
    pl.DataFrame({"id1": [1], "id2": [2], "geom": ["POINT (0.5 0.5)"]}).write_parquet(p)
    candidates = pl.DataFrame({"id1": [1], "id2": [2]})
    res = exact_spatial_filter(
        candidates, str(p), ["id1", "id2"], "geom", "wkt", sgeom.box(0, 0, 1, 1)
    )
    assert res.height == 1
