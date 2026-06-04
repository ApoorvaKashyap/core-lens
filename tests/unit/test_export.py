import pathlib
import pytest
import polars as pl
import geopandas as gpd

from core_lens.export import parquet, json, geoparquet, geojson
from core_lens.base.result import Result


def test_parquet(sample_result: Result, tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "out.parquet"
    parquet(sample_result, out_path)

    assert out_path.exists()
    df = pl.read_parquet(out_path)
    assert df.columns == ["mws_id", "ndvi_mean"]
    assert df.height == 2


def test_json(sample_result: Result, tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "out.json"
    json(sample_result, out_path)

    assert out_path.exists()
    df = pl.read_json(out_path)
    assert df.columns == ["mws_id", "ndvi_mean"]
    assert df.height == 2


def test_geoparquet_no_geometry(sample_result: Result, tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "out.geoparquet"
    with pytest.raises(TypeError, match="no geometry"):
        geoparquet(sample_result, out_path)


def test_geojson_no_geometry(sample_result: Result, tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "out.geojson"
    with pytest.raises(TypeError, match="no geometry"):
        geojson(sample_result, out_path)


def test_geoparquet_with_geometry(
    sample_result: Result, tmp_path: pathlib.Path
) -> None:
    result_with_geom = sample_result.with_geometry()

    out_path = tmp_path / "out.geoparquet"
    geoparquet(result_with_geom, out_path)

    assert out_path.exists()
    gdf = gpd.read_parquet(out_path)

    # Check that geometry column is appropriately mapped
    assert "geometry" in gdf.columns
    assert gdf.geometry.name == "geometry"
    assert len(gdf) == 2


def test_geojson_with_geometry(sample_result: Result, tmp_path: pathlib.Path) -> None:
    result_with_geom = sample_result.with_geometry()

    out_path = tmp_path / "out.json"
    geojson(result_with_geom, out_path)

    assert out_path.exists()
    gdf = gpd.read_file(out_path)

    assert "geometry" in gdf.columns
    assert gdf.geometry.name == "geometry"
    assert len(gdf) == 2
