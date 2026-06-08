"""Tests for ``core_lens.aoi.AoI``."""

from __future__ import annotations
from typing import Any

import pathlib

import polars as pl
import pytest
import shapely.geometry as sgeom
import shapely.wkb as swkb

from core_lens.aoi import AoI, SeasonConfig, _REGISTRY, _entity_name
from core_lens.base.entity import BaseEntity, EntityValidationError
from core_lens.schema.profile import SchemaProfile


def _write_parquet_no_key_col(path: pathlib.Path) -> None:
    """Write a Parquet whose id column name differs from the entity key_cols.

    Args:
        path: Destination file path.
    """
    pl.DataFrame(
        {"wrong_id": ["1"], "geometry": [swkb.dumps(sgeom.box(0, 0, 1, 1))]}
    ).write_parquet(path)


def _write_parquet_no_geom_col(path: pathlib.Path) -> None:
    """Write a Parquet that has key col but not the declared geometry col.

    Args:
        path: Destination file path.
    """
    pl.DataFrame({"mws_id": ["1"], "geom": [b"x"]}).write_parquet(path)


def _make_bad_entity(
    static: pathlib.Path,
    key_cols: list[str],
    geometry_col: str,
) -> type[BaseEntity]:
    """Return a concrete entity class that deliberately mismatches the Parquet schema.

    Args:
        static: Path to an already-written Parquet file.
        key_cols: key_cols value to declare on the entity (may not exist in file).
        geometry_col: geometry_col value to declare (may not exist in file).

    Returns:
        A ``BaseEntity`` subclass whose declared columns differ from the file.
    """
    profile = SchemaProfile(
        key_cols=key_cols,
        geometry_col=geometry_col,
        geometry_type="wkb",
        annual_time_col=None,
        fortnightly_time_col=None,
        bbox_cols=None,
    )
    _static = str(static)
    _profile = profile
    _key_cols = key_cols
    _geometry_col = geometry_col

    class _BadEntity(BaseEntity):
        @property
        def key_cols(self) -> list[str]:
            return _key_cols

        @property
        def geometry_col(self) -> str:
            return _geometry_col

        @property
        def static_path(self) -> str:
            return _static

        @property
        def schema_profile(self) -> SchemaProfile:
            return _profile

    return _BadEntity


class TestAoIBoundaryModes:
    def test_bbox_sets_polygon_geometry(self, registered_entity_cls: Any) -> None:
        aoi = AoI(".", bbox=(73.0, 15.0, 74.0, 16.0))

        assert aoi.geometry is not None
        assert not aoi.geometry.is_empty  # type: ignore[attr-defined]

    def test_geometry_used_directly(self, registered_entity_cls: Any) -> None:
        geom = sgeom.Point(73.5, 15.5).buffer(0.5)
        aoi = AoI(".", geometry=geom)

        assert aoi.geometry is geom

    def test_no_boundary_raises(self, registered_entity_cls: Any) -> None:
        with pytest.raises(ValueError, match="exactly one boundary"):
            AoI(".")

    def test_bbox_and_geometry_raises(self, registered_entity_cls: Any) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            AoI(".", bbox=(0.0, 0.0, 1.0, 1.0), geometry=sgeom.Point(0.5, 0.5))

    def test_bbox_and_entity_kwarg_raises(self, registered_entity_cls: Any) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            AoI(".", bbox=(0.0, 0.0, 1.0, 1.0), mws_id="13_001")

    def test_geometry_and_entity_kwarg_raises(self, registered_entity_cls: Any) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            AoI(".", geometry=sgeom.Point(73.5, 15.5), mws_id="13_001")


class TestAoIDataRoot:
    def test_data_root_resolved_to_absolute(
        self, registered_entity_cls: Any, tmp_path: Any
    ) -> None:
        aoi = AoI(str(tmp_path), bbox=(73.0, 15.0, 74.0, 16.0))

        assert aoi.data_root == tmp_path.resolve()
        assert aoi.data_root.is_absolute()


class TestAoISeasons:
    def test_default_season_config_applied(self, registered_entity_cls: Any) -> None:
        aoi = AoI(".", bbox=(73.0, 15.0, 74.0, 16.0))

        assert isinstance(aoi.seasons, SeasonConfig)

    def test_custom_season_config_stored(self, registered_entity_cls: Any) -> None:
        custom = SeasonConfig(
            kharif=("06-01", "10-15"),
            rabi=("10-16", "02-28"),
            zaid=("03-01", "05-31"),
        )
        aoi = AoI(".", bbox=(73.0, 15.0, 74.0, 16.0), seasons=custom)

        assert aoi.seasons is custom

    def test_current_season_returns_string(self, bbox_aoi: Any) -> None:
        assert bbox_aoi.current_season in {"kharif", "rabi", "zaid"}

    def test_current_year_is_integer(self, bbox_aoi: Any) -> None:
        assert isinstance(bbox_aoi.current_year, int)
        assert bbox_aoi.current_year > 2000


class TestAoIRegister:
    def test_register_adds_entity_to_registry(self, entity_cls: Any) -> None:
        AoI.register(entity_cls)

        assert "minimalmws" in _REGISTRY

    def test_registered_entities_sorted(self, entity_cls: Any) -> None:
        AoI.register(entity_cls)

        names = AoI.registered_entities()

        assert names == sorted(names)

    def test_register_missing_static_raises(self, tmp_path: Any) -> None:
        from conftest import _make_entity_cls

        cls = _make_entity_cls(tmp_path / "nonexistent.parquet")

        with pytest.raises(EntityValidationError, match="does not exist"):
            AoI.register(cls)

    def test_register_wrong_key_col_raises(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        _write_parquet_no_key_col(p)
        cls = _make_bad_entity(p, key_cols=["mws_id"], geometry_col="geometry")

        with pytest.raises(EntityValidationError, match="key_cols"):
            AoI.register(cls)

    def test_register_missing_geometry_col_raises(self, tmp_path: Any) -> None:
        p = tmp_path / "s.parquet"
        _write_parquet_no_geom_col(p)
        cls = _make_bad_entity(p, key_cols=["mws_id"], geometry_col="geometry")

        with pytest.raises(EntityValidationError, match="geometry_col"):
            AoI.register(cls)

    def test_register_missing_annual_path_raises(
        self, tmp_path: Any, static_parquet: Any
    ) -> None:
        from conftest import _make_entity_cls

        cls = _make_entity_cls(
            static_parquet,
            annual=tmp_path / "nonexistent_annual.parquet",
        )

        with pytest.raises(EntityValidationError, match="annual"):
            AoI.register(cls)

    def test_register_missing_fortnightly_path_raises(
        self, tmp_path: Any, static_parquet: Any
    ) -> None:
        from conftest import _make_entity_cls

        cls = _make_entity_cls(
            static_parquet,
            fortnightly=tmp_path / "nonexistent_fn.parquet",
        )

        with pytest.raises(EntityValidationError, match="fortnightly"):
            AoI.register(cls)


class TestAoIDeregister:
    def test_deregister_removes_entity(self, entity_cls: Any) -> None:
        AoI.register(entity_cls)
        assert "minimalmws" in _REGISTRY

        AoI.deregister(entity_cls)

        assert "minimalmws" not in _REGISTRY

    def test_deregister_unknown_entity_is_silent(self, entity_cls: Any) -> None:
        AoI.deregister(entity_cls)


class TestAoIGetattr:
    def test_registered_entity_accessible_as_attribute(self, bbox_aoi: Any) -> None:
        view = bbox_aoi.minimalmws

        assert view is not None

    def test_unregistered_attribute_raises_attribute_error(self, bbox_aoi: Any) -> None:
        with pytest.raises(AttributeError, match="no attribute"):
            _ = bbox_aoi.nonexistent_entity

    def test_error_message_lists_registered_entities(self, bbox_aoi: Any) -> None:
        with pytest.raises(AttributeError, match="minimalmws"):
            _ = bbox_aoi.nonexistent_entity


class TestAoIPlot:
    def test_plot_returns_lonboard_map(self, bbox_aoi: Any) -> None:
        import lonboard

        plot_map = bbox_aoi.plot()
        assert isinstance(plot_map, lonboard.Map)
        assert len(plot_map.layers) == 1
        assert isinstance(plot_map.layers[0], lonboard.PolygonLayer)

    def test_plot_with_overlay(self, bbox_aoi: Any, tmp_path: Any) -> None:
        import lonboard
        import shapely.geometry as sgeom
        import shapely.wkb as swkb
        import polars as pl
        from core_lens.base.result import Result
        from core_lens.schema.profile import Resolution
        from core_lens.base.entity import BaseEntity

        # Need a dummy entity for Result
        class DummyEntity(BaseEntity):
            @property
            def key_cols(self) -> list[str]:
                return ["id"]

            @property
            def geometry_col(self) -> str:
                return "geometry"

            @property
            def static_path(self) -> str:
                return ""

            @property
            def schema_profile(self) -> Any:
                return None

        geom = sgeom.box(0, 0, 1, 1)
        df = pl.DataFrame({"id": ["1"], "geometry": [swkb.dumps(geom)]})

        result = Result(
            data=df,
            resolution=Resolution.STATIC,
            has_geometry=True,
            key_cols=["id"],
            entity_name="dummy",
            entity=DummyEntity(),
        )

        plot_map = bbox_aoi.plot(overlay=result)
        assert isinstance(plot_map, lonboard.Map)
        assert len(plot_map.layers) == 2
        assert isinstance(plot_map.layers[1], lonboard.PolygonLayer)


class TestEntityName:
    def test_strips_entity_suffix_and_lowercases(self) -> None:
        class MWSEntity(BaseEntity):
            @property
            def key_cols(self) -> list[str]:
                return []  # pragma: no cover

            @property
            def geometry_col(self) -> str:
                return ""  # pragma: no cover

            @property
            def static_path(self) -> str:
                return ""  # pragma: no cover

        assert _entity_name(MWSEntity) == "mws"

    def test_class_with_no_entity_suffix_lowercased(self) -> None:
        class Forest(BaseEntity):
            @property
            def key_cols(self) -> list[str]:
                return []  # pragma: no cover

            @property
            def geometry_col(self) -> str:
                return ""  # pragma: no cover

            @property
            def static_path(self) -> str:
                return ""  # pragma: no cover

        assert _entity_name(Forest) == "forest"


class TestAoIResolveGeometry:
    def test_resolve_geometry(
        self, registered_entity_cls: Any, tmp_path: pathlib.Path, monkeypatch: Any
    ) -> None:
        from core_lens.aoi import AoI
        import shapely.geometry as sgeom
        import shapely.wkb as swkb
        import polars as pl
        from core_lens.base.entity import BaseEntity

        static_file = tmp_path / "static.parquet"

        poly1 = sgeom.box(0, 0, 1, 1)
        poly2 = sgeom.box(1, 1, 2, 2)

        pl.DataFrame(
            {
                "mws_id": ["1", "2"],
                "district": ["Dharwad", "Dharwad"],
                "geometry": [swkb.dumps(poly1), swkb.dumps(poly2)],
            }
        ).write_parquet(static_file)

        class BoundEntity(BaseEntity):
            @property
            def key_cols(self) -> list[str]:
                return ["mws_id"]

            @property
            def geometry_col(self) -> str:
                return "geometry"

            @property
            def static_path(self) -> str:
                return str(static_file)

            @property
            def extra_static_cols(self) -> list[str]:
                return ["district"]

        from core_lens.aoi import _REGISTRY

        old_registry = dict(_REGISTRY)

        try:
            _REGISTRY.clear()
            _REGISTRY["mws"] = BoundEntity()

            # Test key match
            aoi = AoI(data_root=str(tmp_path), mws_id="1")
            assert getattr(aoi.geometry, "equals", lambda x: False)(poly1)

            # Test attribute match
            aoi = AoI(data_root=str(tmp_path), district="Dharwad")
            import shapely.ops as sops

            union = sops.unary_union([poly1, poly2])
            assert getattr(aoi.geometry, "equals", lambda x: False)(union)

            with pytest.raises(ValueError, match="No rows matched the filters"):
                AoI(data_root=str(tmp_path), mws_id="999")

            with pytest.raises(
                EntityValidationError,
                match="No registered entity can satisfy the filters",
            ):
                AoI(data_root=str(tmp_path), unknown_key="1")

            # Test fallback match by entity name
            aoi_fallback = AoI(data_root=str(tmp_path), mws="1")
            assert getattr(aoi_fallback.geometry, "equals", lambda x: False)(union)

        finally:
            _REGISTRY.clear()
            _REGISTRY.update(old_registry)

    def test_validate_entity_invalid_parquet(self, tmp_path: pathlib.Path) -> None:
        from core_lens.aoi import _validate_entity
        from core_lens.base.entity import BaseEntity

        p = tmp_path / "bad.parquet"
        p.write_text("not a parquet file")

        class BadEntity(BaseEntity):
            @property
            def key_cols(self) -> list[str]:
                return ["id"]

            @property
            def geometry_col(self) -> str:
                return "geom"

            @property
            def static_path(self) -> str:
                return str(p)

        with pytest.raises(EntityValidationError, match="could not read schema"):
            _validate_entity(BadEntity(), "bad_entity")
