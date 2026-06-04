"""Tests for ``core_lens.schema.profile.SchemaProfile``."""

from __future__ import annotations
from typing import Any

import pytest
from pydantic import ValidationError

from core_lens.schema.profile import SchemaProfile


def _valid_kwargs(**overrides: Any) -> dict[str, Any]:
    """Return a minimal set of valid SchemaProfile keyword arguments.

    Args:
        **overrides: Fields to override in the base set.

    Returns:
        A dict suitable for ``SchemaProfile(**kwargs)``.
    """
    base = {
        "key_cols": ["mws_id"],
        "geometry_col": "geometry",
        "geometry_type": "wkb",
        "annual_time_col": None,
        "fortnightly_time_col": None,
        "bbox_cols": None,
    }
    base.update(overrides)
    return base


class TestSchemaProfileConstruction:
    def test_minimal_valid_profile(self) -> None:
        profile = SchemaProfile(**_valid_kwargs())

        assert profile.key_cols == ["mws_id"]
        assert profile.geometry_col == "geometry"
        assert profile.geometry_type == "wkb"
        assert profile.annual_time_col is None
        assert profile.fortnightly_time_col is None
        assert profile.bbox_cols is None

    def test_defaults_for_extra_col_lists(self) -> None:
        profile = SchemaProfile(**_valid_kwargs())

        assert profile.extra_static_cols == []
        assert profile.extra_annual_cols == []
        assert profile.extra_fortnightly_cols == []

    def test_composite_key_cols(self) -> None:
        profile = SchemaProfile(
            **_valid_kwargs(key_cols=["state", "district", "tehsil"])
        )

        assert profile.key_cols == ["state", "district", "tehsil"]

    def test_all_geometry_types_accepted(self) -> None:
        for geo_type in ("wkb", "wkt"):
            profile = SchemaProfile(**_valid_kwargs(geometry_type=geo_type))
            assert profile.geometry_type == geo_type

    def test_latlon_geometry_type_with_bbox_cols(self) -> None:
        bbox = ("minx", "miny", "maxx", "maxy")
        profile = SchemaProfile(**_valid_kwargs(geometry_type="latlon", bbox_cols=bbox))

        assert profile.geometry_type == "latlon"
        assert profile.bbox_cols == bbox

    def test_extra_cols_stored(self) -> None:
        profile = SchemaProfile(
            **_valid_kwargs(
                extra_static_cols=["area_ha", "state"],
                extra_annual_cols=["ndvi_mean"],
                extra_fortnightly_cols=["ndvi"],
            )
        )

        assert "area_ha" in profile.extra_static_cols
        assert "ndvi_mean" in profile.extra_annual_cols
        assert "ndvi" in profile.extra_fortnightly_cols

    def test_temporal_cols_stored(self) -> None:
        profile = SchemaProfile(
            **_valid_kwargs(
                annual_time_col="year",
                fortnightly_time_col="fortnightly_date",
            )
        )

        assert profile.annual_time_col == "year"
        assert profile.fortnightly_time_col == "fortnightly_date"


class TestSchemaProfileValidators:
    def test_empty_key_cols_raises(self) -> None:
        with pytest.raises(ValidationError, match="key_cols"):
            SchemaProfile(**_valid_kwargs(key_cols=[]))

    def test_latlon_without_bbox_cols_raises(self) -> None:
        with pytest.raises(ValidationError, match="bbox_cols"):
            SchemaProfile(**_valid_kwargs(geometry_type="latlon", bbox_cols=None))

    def test_invalid_geometry_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            SchemaProfile(**_valid_kwargs(geometry_type="geojson"))


class TestSchemaProfileImmutability:
    def test_profile_is_frozen(self) -> None:
        profile = SchemaProfile(**_valid_kwargs())

        with pytest.raises(ValidationError):
            profile.geometry_col = "geom"
