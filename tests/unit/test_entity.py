"""Tests for ``core_lens.base.entity.BaseEntity``."""

from __future__ import annotations
from typing import Any
import pathlib

import shapely.geometry as sgeom

from core_lens.base.entity import BaseEntity
from core_lens.base.view import View


class TestEntityWhere:
    def test_where_returns_view(
        self, entity_cls: Any, static_parquet: pathlib.Path
    ) -> None:
        entity = entity_cls()
        view = entity.where(district="TestDistrict")
        assert isinstance(view, View)
        assert view.entity is entity
        assert view.entity_name == "minimalmws"
        assert len(view.keys) == 2


class TestEntitySpatialFilter:
    def test_spatial_filter_bbox(
        self, entity_cls: Any, static_parquet: pathlib.Path
    ) -> None:
        entity = entity_cls()
        view = entity.spatial_filter(bbox=(72.0, 14.0, 75.0, 17.0))
        assert isinstance(view, View)
        assert len(view.keys) == 2

    def test_spatial_filter_geometry(
        self, entity_cls: Any, static_parquet: pathlib.Path
    ) -> None:
        entity = entity_cls()
        view = entity.spatial_filter(geometry=sgeom.box(72.0, 14.0, 75.0, 17.0))
        assert isinstance(view, View)
        assert len(view.keys) == 2


class TestEntitySpatialJoin:
    def test_spatial_join(self, entity_cls: Any, static_parquet: pathlib.Path) -> None:
        entity = entity_cls()
        view = entity.spatial_join(entity_cls(), agg={"district": "first"})
        assert isinstance(view, View)
        # Assuming spatial_join creates an empty view for testing until fully implemented
        assert "mws_id" in view.keys.columns


class TestEntitySchemaProfile:
    def test_lazy_schema_profile(self, static_parquet: pathlib.Path) -> None:
        class RealEntity(BaseEntity):
            @property
            def key_cols(self) -> list[str]:
                return ["mws_id"]

            @property
            def geometry_col(self) -> str:
                return "geometry"

            @property
            def static_path(self) -> str:
                return str(static_parquet)

            @property
            def annual_path(self) -> str | None:
                return None

            @property
            def fortnightly_path(self) -> str | None:
                return None

        entity = RealEntity()
        profile = entity.schema_profile
        assert profile.key_cols == ["mws_id"]
        assert profile.geometry_col == "geometry"
        assert profile.geometry_type == "wkb"
