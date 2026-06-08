"""Tests for ``core_lens.base.view.View``."""

from __future__ import annotations
from typing import Any

import polars as pl
import pytest

from core_lens.base.view import View


def _make_view(entity: Any, entity_name: str = "minimalmws") -> View:
    """Create a ``View`` with an empty keys frame for testing filter logic.

    Args:
        entity: A ``BaseEntity`` instance.
        entity_name: Name to assign to the view.

    Returns:
        A ``View`` with no keys and no filters.
    """
    keys = pl.DataFrame({"mws_id": []}, schema={"mws_id": pl.String})
    return View(keys=keys, entity=entity, entity_name=entity_name)


class TestViewConstruction:
    def test_stores_all_fields(self, entity_cls: Any) -> None:
        entity = entity_cls()
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        view = View(keys=keys, entity=entity, entity_name="mws")

        assert view.entity is entity
        assert view.entity_name == "mws"
        assert view.time_filter is None
        assert view.join_spec is None

    def test_optional_fields_default_to_none(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())

        assert view.time_filter is None
        assert view.join_spec is None


class TestViewBetweenDateRange:
    def test_date_range_stored_correctly(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())
        result = view.between("2010-01-01", "2023-12-31")

        assert result.time_filter == {"start": "2010-01-01", "end": "2023-12-31"}

    def test_returns_new_view(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())
        new_view = view.between("2010-01-01", "2023-12-31")

        assert new_view is not view

    def test_original_view_unchanged(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())
        view.between("2010-01-01", "2023-12-31")

        assert view.time_filter is None

    def test_missing_start_raises(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())

        with pytest.raises(ValueError, match="Both 'start' and 'end'"):
            view.between(end="2023-12-31")

    def test_missing_end_raises(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())

        with pytest.raises(ValueError, match="Both 'start' and 'end'"):
            view.between(start="2010-01-01")


class TestViewBetweenSeasonMode:
    def test_season_stored(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())
        result = view.between(season="kharif")

        assert result.time_filter == {"season": "kharif"}

    def test_season_with_year_int(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())
        result = view.between(season="kharif", year=2022)

        assert result.time_filter == {"season": "kharif", "year": 2022}

    def test_season_with_year_tuple(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())
        result = view.between(season="rabi", year=(2018, 2023))

        assert result.time_filter == {"season": "rabi", "year": (2018, 2023)}

    def test_current_season_accepted(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())
        result = view.between(season="current")

        assert result.time_filter == {"season": "current"}

    def test_invalid_season_name_raises(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())

        with pytest.raises(ValueError, match="Unknown season"):
            view.between(season="monsoon")

    def test_current_season_with_year_raises(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())

        with pytest.raises(ValueError, match="season='current'"):
            view.between(season="current", year=2022)

    def test_season_and_date_range_mixed_raises(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())

        with pytest.raises(ValueError, match="mutually exclusive"):
            view.between("2010-01-01", "2023-12-31", season="kharif")

    def test_year_without_season_raises(self, entity_cls: Any) -> None:
        view = _make_view(entity_cls())

        with pytest.raises(ValueError, match="'year' is only valid"):
            view.between(year=2022)


class TestViewAllSeasonsAccepted:
    @pytest.mark.parametrize("season", ["kharif", "rabi", "zaid", "current"])
    def test_valid_season_names(self, entity_cls: Any, season: Any) -> None:
        view = _make_view(entity_cls())
        result = view.between(season=season)

        assert result.time_filter is not None
        assert result.time_filter["season"] == season


class TestViewMaterialisation:
    def test_static_returns_result_with_geometry(
        self, entity_cls: Any, static_parquet: Any
    ) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        view = View(keys=keys, entity=entity_cls(), entity_name="minimalmws")

        result = view.static
        assert result.data.shape == (1, 3)

    def test_materialise_with_time_filter(
        self, entity_cls: Any, static_parquet: Any, monkeypatch: Any
    ) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        from core_lens.base.entity import BaseEntity
        from core_lens.schema.profile import SchemaProfile

        class TimeEntity(BaseEntity):
            @property
            def key_cols(self) -> list[str]:
                return ["mws_id"]

            @property
            def geometry_col(self) -> str:
                return "geometry"

            @property
            def annual_path(self) -> str:
                return str(static_parquet)

            @property
            def static_path(self) -> str:
                return str(static_parquet)

            @property
            def schema_profile(self) -> SchemaProfile:
                return SchemaProfile(
                    key_cols=["mws_id"],
                    geometry_col="geometry",
                    geometry_type="wkb",
                    annual_time_col="time",
                    fortnightly_time_col=None,
                    bbox_cols=None,
                    extra_static_cols=[],
                )

        view = View(keys=keys, entity=TimeEntity(), entity_name="minimalmws")
        view.time_filter = {"start": "2020-01-01", "end": "2020-12-31"}

        # Let's mock resolve_time_filter so we don't need a real time column in the static_parquet
        import core_lens.utils.season

        def mock_resolve(*args: Any, **kwargs: Any) -> pl.Expr:
            return pl.lit(True)

        monkeypatch.setattr(core_lens.utils.season, "resolve_time_filter", mock_resolve)

        from core_lens.schema.profile import Resolution

        result = view._materialise(Resolution.ANNUAL)
        assert result.data.shape == (1, 3)

    def test_annual_returns_result_without_geometry(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        view = View(keys=keys, entity=entity_cls_full(), entity_name="minimalmws")

        result = view.annual

        assert result.has_geometry is False

    def test_fortnightly_returns_result_without_geometry(
        self, entity_cls_full: Any
    ) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        view = View(keys=keys, entity=entity_cls_full(), entity_name="minimalmws")

        result = view.fortnightly

        assert result.has_geometry is False

    def test_annual_raises_when_no_annual_path(self, entity_cls: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        view = View(keys=keys, entity=entity_cls(), entity_name="minimalmws")

        with pytest.raises(AttributeError, match="no annual_path"):
            _ = view.annual

    def test_fortnightly_raises_when_no_fortnightly_path(self, entity_cls: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        view = View(keys=keys, entity=entity_cls(), entity_name="minimalmws")

        with pytest.raises(AttributeError, match="no fortnightly_path"):
            _ = view.fortnightly

    def test_join_spec_raises_not_implemented(self, entity_cls: Any) -> None:
        join_spec = {"other": "forest", "agg": {"cover": "area"}}
        keys = pl.DataFrame({"mws_id": []}, schema={"mws_id": pl.String})
        view = View(
            keys=keys,
            entity=entity_cls(),
            entity_name="minimalmws",
            join_spec=join_spec,
        )

        with pytest.raises(NotImplementedError, match="spatial_join"):
            _ = view.static

    def test_join_spec_preserved_across_between(self, entity_cls: Any) -> None:
        join_spec = {"other": "forest", "agg": {"cover": "area"}}
        keys = pl.DataFrame({"mws_id": []}, schema={"mws_id": pl.String})
        view = View(
            keys=keys,
            entity=entity_cls(),
            entity_name="minimalmws",
            join_spec=join_spec,
        )
        new_view = view.between("2020-01-01", "2022-12-31")

        assert new_view.join_spec == join_spec


class TestFortnightlyTemporalColumns:
    """Materialising a fortnightly View must inject all temporal grouping columns."""

    def test_year_column_added(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        assert "year" in result.columns

    def test_month_column_added(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        assert "month" in result.columns

    def test_year_month_column_added(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        assert "year_month" in result.columns

    def test_season_column_added(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        assert "season" in result.columns

    def test_season_year_column_added(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        assert "season_year" in result.columns

    def test_year_values_correct(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        # conftest writes fortnightly_date = [2022-01-01, 2022-01-15] for mws_id 13_001
        assert result.df()["year"].to_list() == [2022, 2022]

    def test_year_month_format(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        assert result.df()["year_month"].to_list() == ["2022-01", "2022-01"]

    def test_season_label_is_string(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        for label in result.df()["season"].to_list():
            assert isinstance(label, str)
            assert label in {"kharif", "rabi", "zaid"}

    def test_season_year_format(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        for label in result.df()["season_year"].to_list():
            # e.g. "rabi_2022"
            parts = label.rsplit("_", 1)
            assert len(parts) == 2
            assert parts[0] in {"kharif", "rabi", "zaid"}
            assert parts[1].isdigit()

    def test_existing_year_col_not_overwritten(self, tmp_path: Any) -> None:
        """If entity already supplies a 'year' column it must be preserved."""
        import datetime
        from core_lens.base.entity import BaseEntity
        from core_lens.schema.profile import SchemaProfile
        import shapely.geometry as sgeom
        import shapely.wkb as swkb

        wkb_bytes = swkb.dumps(sgeom.box(73.0, 15.0, 74.0, 16.0))
        static_path = tmp_path / "static.parquet"
        pl.DataFrame({"mws_id": ["x"], "geometry": [wkb_bytes]}).write_parquet(
            static_path
        )

        fn_path = tmp_path / "fortnightly.parquet"
        pl.DataFrame(
            {
                "mws_id": ["x"],
                "fortnightly_date": [datetime.date(2022, 1, 1)],
                "year": [9999],  # pre-existing — must survive unchanged
                "ndvi": [0.5],
            }
        ).write_parquet(fn_path)

        _static = str(static_path)
        _fn = str(fn_path)
        _profile = SchemaProfile(
            key_cols=["mws_id"],
            geometry_col="geometry",
            geometry_type="wkb",
            annual_time_col=None,
            fortnightly_time_col="fortnightly_date",
            bbox_cols=None,
        )

        class _TestEntity(BaseEntity):
            @property
            def key_cols(self) -> list[str]:
                return ["mws_id"]

            @property
            def geometry_col(self) -> str:
                return "geometry"

            @property
            def static_path(self) -> str:
                return _static

            @property
            def fortnightly_path(self) -> str | None:
                return _fn

            @property
            def schema_profile(self) -> SchemaProfile:
                return _profile

        keys = pl.DataFrame({"mws_id": ["x"]})
        result = View(keys=keys, entity=_TestEntity(), entity_name="test").fortnightly
        assert result.df()["year"].to_list() == [9999]


class TestAggregateBySeasonIntegration:
    """aggregate(by=...) must group correctly on fortnightly temporal columns."""

    def test_aggregate_by_year_groups_correctly(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        # both rows are 2022-01-xx → mean of [0.42, 0.47]
        agg = result.aggregate(pl.mean("ndvi"), by="year")
        df = agg.df()
        assert "year" in df.columns
        assert len(df) == 1
        assert df["year"].to_list() == [2022]

    def test_aggregate_by_month_groups_correctly(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        agg = result.aggregate(pl.mean("ndvi"), by="month")
        assert "month" in agg.df().columns
        assert len(agg.df()) == 1

    def test_aggregate_by_season_returns_valid_labels(
        self, entity_cls_full: Any
    ) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        agg = result.aggregate(pl.mean("ndvi"), by="season")
        df = agg.df()
        assert "season" in df.columns
        for s in df["season"].to_list():
            assert s in {"kharif", "rabi", "zaid"}

    def test_aggregate_by_season_year_returns_valid_labels(
        self, entity_cls_full: Any
    ) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        agg = result.aggregate(pl.mean("ndvi"), by="season_year")
        df = agg.df()
        assert "season_year" in df.columns
        for s in df["season_year"].to_list():
            parts = s.rsplit("_", 1)
            assert parts[0] in {"kharif", "rabi", "zaid"}

    def test_aggregate_by_year_month_groups_correctly(
        self, entity_cls_full: Any
    ) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        agg = result.aggregate(pl.mean("ndvi"), by="year_month")
        assert "year_month" in agg.df().columns
        assert agg.df()["year_month"].to_list() == ["2022-01"]

    def test_aggregate_ndvi_mean_value(self, entity_cls_full: Any) -> None:
        """Mean of 0.42 and 0.47 → 0.445."""
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(
            keys=keys, entity=entity_cls_full(), entity_name="mws"
        ).fortnightly
        agg = result.aggregate(pl.mean("ndvi"), by="year")
        import math

        assert math.isclose(agg.df()["ndvi"].to_list()[0], 0.445, rel_tol=1e-6)

    def test_derive_then_aggregate(self, entity_cls_full: Any) -> None:
        """derive() then aggregate() full pipeline."""
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = (
            View(keys=keys, entity=entity_cls_full(), entity_name="mws")
            .fortnightly.derive("ndvi_pct", pl.col("ndvi") * 100)
            .aggregate(pl.mean("ndvi_pct"), by="year")
        )
        assert "ndvi_pct" in result.df().columns
        assert "year" in result.df().columns


class TestWithGeometryJoin:
    """with_geometry() on a non-static Result must join the geometry column."""

    def test_with_geometry_sets_flag(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(keys=keys, entity=entity_cls_full(), entity_name="mws").annual
        assert result.has_geometry is False
        with_geom = result.with_geometry()
        assert with_geom.has_geometry is True

    def test_with_geometry_adds_geometry_col(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(keys=keys, entity=entity_cls_full(), entity_name="mws").annual
        with_geom = result.with_geometry()
        assert entity_cls_full().geometry_col in with_geom.columns

    def test_with_geometry_row_count_preserved(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(keys=keys, entity=entity_cls_full(), entity_name="mws").annual
        with_geom = result.with_geometry()
        assert len(with_geom.df()) == len(result.df())

    def test_with_geometry_original_unchanged(self, entity_cls_full: Any) -> None:
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = View(keys=keys, entity=entity_cls_full(), entity_name="mws").annual
        result.with_geometry()
        assert result.has_geometry is False

    def test_derive_with_geometry_chain(self, entity_cls_full: Any) -> None:
        """derive() → with_geometry() chain must work end-to-end."""
        keys = pl.DataFrame({"mws_id": ["13_001"]})
        result = (
            View(keys=keys, entity=entity_cls_full(), entity_name="mws")
            .annual.derive("ndvi_sq", pl.col("ndvi_mean") ** 2)
            .with_geometry()
        )
        assert "ndvi_sq" in result.columns
        assert result.has_geometry is True
