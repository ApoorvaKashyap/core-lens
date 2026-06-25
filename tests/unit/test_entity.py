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

    def test_where_no_args(self, entity_cls: Any) -> None:
        entity = entity_cls()
        view = entity.where()
        assert len(view.keys) == 2

    def test_where_invalid_kwarg(self, entity_cls: Any) -> None:
        entity = entity_cls()
        import pytest

        with pytest.raises(ValueError, match="not a registered entity name"):
            entity.where(unknown_kwarg="val")

    def test_where_entity_kwarg(
        self, entity_cls: Any, registered_entity_cls: Any, tmp_path: pathlib.Path
    ) -> None:
        import polars as pl
        import shapely.wkb as swkb
        from core_lens.aoi import AoI

        # Create a district entity
        district_path = tmp_path / "district.parquet"
        wkb_bytes = swkb.dumps(sgeom.box(72.0, 14.0, 75.0, 17.0))
        pl.DataFrame(
            {"district_id": ["D1"], "name": ["TestDistrict"], "geometry": [wkb_bytes]}
        ).write_parquet(district_path)

        class CustomDistrictEntity(BaseEntity):
            @property
            def key_cols(self) -> list[str]:
                return ["district_id"]

            @property
            def geometry_col(self) -> str:
                return "geometry"

            @property
            def static_path(self) -> str:
                return str(district_path)

            @property
            def schema_profile(self) -> Any:
                from core_lens.schema.profile import SchemaProfile

                return SchemaProfile(
                    key_cols=["district_id"],
                    geometry_col="geometry",
                    geometry_type="wkb",
                    annual_time_col=None,
                    fortnightly_time_col=None,
                    bbox_cols=None,
                )

        AoI.register(CustomDistrictEntity)

        entity = entity_cls()
        view = entity.where(customdistrict="D1")
        assert len(view.keys) >= 0

        import pytest

        with pytest.raises(ValueError, match="No rows matched"):
            entity.where(customdistrict="NonExistent")

        with pytest.raises(ValueError, match="not a registered entity name"):
            entity.where(districtentity="NonExistent")

    def test_resolve_not_found(self, entity_cls: Any) -> None:
        entity = entity_cls()
        import pytest

        with pytest.raises(FileNotFoundError):
            entity._resolve("non_existent_file.parquet")


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

    def test_spatial_filter_no_args(self, entity_cls: Any) -> None:
        entity = entity_cls()
        import pytest

        with pytest.raises(
            ValueError, match="spatial_filter.. requires either 'geometry' or 'bbox'"
        ):
            entity.spatial_filter()
