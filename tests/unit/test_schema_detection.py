"""Tests for ``core_lens.schema.detection``."""

from __future__ import annotations
from typing import Any

import datetime
import pathlib

import polars as pl
import pytest
import shapely.geometry as sgeom
import shapely.wkb as swkb

from core_lens.schema.detection import (
    SchemaDetectionError,
    _find_lon_companion,
    _infer_bbox_cols,
    _infer_geometry_type,
    _infer_time_col,
    _require_cols,
    detect,
)


def _wkb_bytes() -> bytes:
    return swkb.dumps(sgeom.box(73.0, 15.0, 74.0, 16.0))


def _write_static(path: pathlib.Path, geometry_col_dtype: str = "wkb") -> None:
    """Write a static Parquet to *path* using the requested geometry encoding.

    Args:
        path: Destination path.
        geometry_col_dtype: ``"wkb"`` for Binary, ``"wkt"`` for String,
            ``"latlon"`` for separate lat/lon float columns.
    """
    if geometry_col_dtype == "wkb":
        data = {"mws_id": ["1"], "geometry": [_wkb_bytes()]}
    elif geometry_col_dtype == "wkt":
        data = {"mws_id": ["1"], "geometry": ["POINT(73.5 15.5)"]}
    else:
        data = {"mws_id": ["1"], "lat": [15.5], "lon": [73.5]}
    pl.DataFrame(data).write_parquet(path)


def _write_annual(path: pathlib.Path, time_col: str = "year") -> None:
    pl.DataFrame({"mws_id": ["1"], time_col: [2021]}).write_parquet(path)


def _write_fortnightly(path: pathlib.Path) -> None:
    pl.DataFrame(
        {
            "mws_id": ["1"],
            "fortnightly_date": [datetime.date(2022, 1, 1)],
        }
    ).write_parquet(path)


class TestDetectFunction:
    def test_static_only_wkb(self, tmp_path: Any) -> None:
        static = tmp_path / "s.parquet"
        _write_static(static, "wkb")

        profile = detect(str(static), key_cols=["mws_id"], geometry_col="geometry")

        assert profile.geometry_type == "wkb"
        assert profile.key_cols == ["mws_id"]
        assert profile.annual_time_col is None
        assert profile.fortnightly_time_col is None

    def test_static_only_wkt(self, tmp_path: Any) -> None:
        static = tmp_path / "s.parquet"
        _write_static(static, "wkt")

        profile = detect(str(static), key_cols=["mws_id"], geometry_col="geometry")

        assert profile.geometry_type == "wkt"

    def test_static_with_annual(self, tmp_path: Any) -> None:
        static = tmp_path / "s.parquet"
        annual = tmp_path / "a.parquet"
        _write_static(static, "wkb")
        _write_annual(annual, "year")

        profile = detect(
            str(static),
            key_cols=["mws_id"],
            geometry_col="geometry",
            annual_path=str(annual),
        )

        assert profile.annual_time_col == "year"

    def test_static_with_fortnightly(self, tmp_path: Any) -> None:
        static = tmp_path / "s.parquet"
        fn = tmp_path / "fn.parquet"
        _write_static(static, "wkb")
        _write_fortnightly(fn)

        profile = detect(
            str(static),
            key_cols=["mws_id"],
            geometry_col="geometry",
            fortnightly_path=str(fn),
        )

        assert profile.fortnightly_time_col == "fortnightly_date"

    def test_missing_static_raises(self, tmp_path: Any) -> None:
        with pytest.raises(SchemaDetectionError, match="static"):
            detect(
                str(tmp_path / "nonexistent.parquet"),
                key_cols=["mws_id"],
                geometry_col="geometry",
            )

    def test_missing_key_col_raises(self, tmp_path: Any) -> None:
        static = tmp_path / "s.parquet"
        _write_static(static, "wkb")

        with pytest.raises(SchemaDetectionError, match="wrong_col"):
            detect(str(static), key_cols=["wrong_col"], geometry_col="geometry")

    def test_missing_geometry_col_raises(self, tmp_path: Any) -> None:
        static = tmp_path / "s.parquet"
        _write_static(static, "wkb")

        with pytest.raises(SchemaDetectionError, match="no_geom"):
            detect(str(static), key_cols=["mws_id"], geometry_col="no_geom")

    def test_extra_static_cols_captured(self, tmp_path: Any) -> None:
        static = tmp_path / "s.parquet"
        pl.DataFrame(
            {"mws_id": ["1"], "geometry": [_wkb_bytes()], "area_ha": [100.0]}
        ).write_parquet(static)

        profile = detect(str(static), key_cols=["mws_id"], geometry_col="geometry")

        assert "area_ha" in profile.extra_static_cols

    def test_latlon_geometry_inferred(self, tmp_path: Any) -> None:
        static = tmp_path / "s.parquet"
        pl.DataFrame(
            {
                "mws_id": ["1"],
                "lat": [15.5],
                "lon": [73.5],
                "minx": [73.0],
                "miny": [15.0],
                "maxx": [74.0],
                "maxy": [16.0],
            }
        ).write_parquet(static)

        profile = detect(str(static), key_cols=["mws_id"], geometry_col="lat")

        assert profile.geometry_type == "latlon"
        assert profile.bbox_cols is not None


class TestInferGeometryType:
    def test_binary_returns_wkb(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        _write_static(p, "wkb")
        schema = pl.scan_parquet(p).collect_schema()

        geo_type, lon = _infer_geometry_type(schema, "geometry", str(p))

        assert geo_type == "wkb"
        assert lon is None

    def test_string_returns_wkt(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        _write_static(p, "wkt")
        schema = pl.scan_parquet(p).collect_schema()

        geo_type, lon = _infer_geometry_type(schema, "geometry", str(p))

        assert geo_type == "wkt"
        assert lon is None

    def test_float_with_lon_returns_latlon(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame({"mws_id": ["1"], "lat": [15.5], "lon": [73.5]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        geo_type, lon = _infer_geometry_type(schema, "lat", str(p))

        assert geo_type == "latlon"
        assert lon == "lon"

    def test_float_without_lon_raises(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame({"mws_id": ["1"], "elevation": [450.0]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        with pytest.raises(SchemaDetectionError, match="longitude companion"):
            _infer_geometry_type(schema, "elevation", str(p))

    def test_unsupported_dtype_raises(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame({"mws_id": ["1"], "geom": [True]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        with pytest.raises(SchemaDetectionError, match="Cannot infer"):
            _infer_geometry_type(schema, "geom", str(p))


class TestInferBboxCols:
    def test_known_minx_pattern_detected(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame(
            {"minx": [0.0], "miny": [0.0], "maxx": [1.0], "maxy": [1.0]}
        ).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        result = _infer_bbox_cols(schema)

        assert result == ("minx", "miny", "maxx", "maxy")

    def test_xmin_pattern_detected(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame(
            {"xmin": [0.0], "ymin": [0.0], "xmax": [1.0], "ymax": [1.0]}
        ).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        result = _infer_bbox_cols(schema)

        assert result == ("xmin", "ymin", "xmax", "ymax")

    def test_no_bbox_pattern_returns_none(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame({"mws_id": ["1"], "geometry": [b""]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        assert _infer_bbox_cols(schema) is None


class TestInferTimeCol:
    def test_prefers_named_time_col_over_date_type(self, tmp_path: Any) -> None:
        p = tmp_path / "a.parquet"
        pl.DataFrame(
            {
                "mws_id": ["1"],
                "year": [2021],
                "obs_date": [datetime.date(2021, 1, 1)],
            }
        ).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        assert _infer_time_col(schema, str(p)) == "year"

    def test_falls_back_to_date_dtype(self, tmp_path: Any) -> None:
        p = tmp_path / "a.parquet"
        pl.DataFrame(
            {"mws_id": ["1"], "recorded_on": [datetime.date(2021, 1, 1)]}
        ).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        assert _infer_time_col(schema, str(p)) == "recorded_on"

    def test_falls_back_to_year_int_col(self, tmp_path: Any) -> None:
        p = tmp_path / "a.parquet"
        pl.DataFrame({"mws_id": ["1"], "crop_year": [2021]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        assert _infer_time_col(schema, str(p)) == "crop_year"

    def test_no_time_col_returns_none(self, tmp_path: Any) -> None:
        p = tmp_path / "a.parquet"
        pl.DataFrame({"mws_id": ["1"], "ndvi": [0.4]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        assert _infer_time_col(schema, str(p)) is None


class TestRequireCols:
    def test_passes_when_all_cols_present(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame({"a": [1], "b": [2]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        _require_cols(schema, ["a", "b"], str(p))

    def test_raises_when_col_missing(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame({"a": [1]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        with pytest.raises(SchemaDetectionError, match="missing_col"):
            _require_cols(schema, ["missing_col"], str(p))


class TestFindLonCompanion:
    def test_lat_finds_lon(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame({"lat": [15.0], "lon": [73.0]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        assert _find_lon_companion(schema, "lat") == "lon"

    def test_latitude_finds_longitude(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame({"latitude": [15.0], "longitude": [73.0]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        assert _find_lon_companion(schema, "latitude") == "longitude"

    def test_unrecognised_lat_col_returns_none(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        pl.DataFrame({"y": [15.0], "x": [73.0]}).write_parquet(p)
        schema = pl.scan_parquet(p).collect_schema()

        assert _find_lon_companion(schema, "y") is None
