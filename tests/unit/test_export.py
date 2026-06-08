import pathlib
import pytest
import polars as pl
import geopandas as gpd

from core_lens.export import parquet, json, geoparquet, geojson, csv
from core_lens.base.result import Result


def test_parquet(sample_result: Result, tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "out.parquet"
    parquet(sample_result, out_path)

    assert out_path.exists()
    df = pl.read_parquet(out_path)
    assert df.columns == ["mws_id", "ndvi_mean"]
    assert df.height == 2


def test_parquet_with_options(sample_result: Result, tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "out_options.parquet"
    parquet(sample_result, out_path, compression="uncompressed")

    assert out_path.exists()
    df = pl.read_parquet(out_path)
    assert df.columns == ["mws_id", "ndvi_mean"]
    assert df.height == 2


def test_parquet_with_geometry(sample_result: Result, tmp_path: pathlib.Path) -> None:
    result_with_geom = sample_result.with_geometry()
    out_path = tmp_path / "out.parquet"
    with pytest.raises(TypeError, match="has geometry"):
        parquet(result_with_geom, out_path)


def test_json(sample_result: Result, tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "out.json"
    json(sample_result, out_path)

    assert out_path.exists()
    df = pl.read_json(out_path)
    assert df.columns == ["mws_id", "ndvi_mean"]
    assert df.height == 2


def test_json_with_geometry(sample_result: Result, tmp_path: pathlib.Path) -> None:
    result_with_geom = sample_result.with_geometry()
    out_path = tmp_path / "out.json"
    with pytest.raises(TypeError, match="has geometry"):
        json(result_with_geom, out_path)


def test_csv(sample_result: Result, tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "out.csv"
    csv(sample_result, out_path)

    assert out_path.exists()
    df = pl.read_csv(out_path)
    assert df.columns == ["mws_id", "ndvi_mean"]
    assert df.height == 2


def test_csv_with_geometry(sample_result: Result, tmp_path: pathlib.Path) -> None:
    result_with_geom = sample_result.with_geometry()
    out_path = tmp_path / "out.csv"
    with pytest.raises(TypeError, match="has geometry"):
        csv(result_with_geom, out_path)


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


def test_geoparquet_with_options(sample_result: Result, tmp_path: pathlib.Path) -> None:
    result_with_geom = sample_result.with_geometry()

    out_path = tmp_path / "out_options.geoparquet"
    geoparquet(
        result_with_geom,
        out_path,
        compression="ZSTD",
        row_group_size=1000,
    )

    assert out_path.exists()
    gdf = gpd.read_parquet(out_path)

    assert "geometry" in gdf.columns
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


def test_geoparquet_options_partition(
    sample_result: Result, tmp_path: pathlib.Path
) -> None:
    result_with_geom = sample_result.with_geometry()
    import duckdb

    out_path = tmp_path / "out_partition1.geoparquet"
    geoparquet(result_with_geom, out_path, partition_by="mws_id")
    assert out_path.exists()

    out_path2 = tmp_path / "out_partition2.geoparquet"
    try:
        geoparquet(
            result_with_geom,
            out_path2,
            partition_by=["mws_id", "ndvi_mean"],
            unknown_bool=True,
        )
    except (duckdb.BinderException, duckdb.NotImplementedException):
        pass


def test_geojson_options(sample_result: Result, tmp_path: pathlib.Path) -> None:
    result_with_geom = sample_result.with_geometry()
    import duckdb

    out_path = tmp_path / "out_opts.json"
    try:
        geojson(
            result_with_geom,
            out_path,
            overwrite=True,
            some_option="test_val",
            flag=False,
            number=10,
        )
    except duckdb.BinderException:
        pass
