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
