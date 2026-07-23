"""Tests for ``core_lens.utils.season``."""

from __future__ import annotations
import datetime

import polars as pl
import pytest
from typing import Any

from core_lens.aoi import SeasonConfig
from core_lens.utils.season import resolve_time_filter, _parse_md, _year_bounds


@pytest.fixture()
def season_config() -> SeasonConfig:
    return SeasonConfig(
        kharif=("06-01", "10-31"),
        rabi=("11-01", "03-31"),
        zaid=("04-01", "05-31"),
    )


class TestResolveTimeFilter:
    def test_date_range(self, season_config: SeasonConfig) -> None:
        expr = resolve_time_filter(
            {"start": "2020-01-01", "end": "2020-12-31"},
            time_col="date",
            season_config=season_config,
        )
        assert isinstance(expr, pl.Expr)
        assert "date" in str(expr)
        # We can't strictly assert string contents, but we know it's built successfully

    def test_date_range_evaluation_on_types(self, season_config: SeasonConfig) -> None:
        df = pl.DataFrame(
            {
                "year_int": [2019, 2020, 2021],
                "date_col": [
                    datetime.date(2019, 6, 1),
                    datetime.date(2020, 1, 15),
                    datetime.date(2021, 1, 1),
                ],
            }
        )

        # Test integer year column
        expr_int = resolve_time_filter(
            {"start": "2020-01-01", "end": "2020-12-31"},
            time_col="year_int",
            season_config=season_config,
        )
        res_int = df.filter(expr_int)
        assert res_int["year_int"].to_list() == [2020]

        # Test Date column
        expr_date = resolve_time_filter(
            {"start": "2020-01-01", "end": "2020-12-31"},
            time_col="date_col",
            season_config=season_config,
        )
        res_date = df.filter(expr_date)
        assert res_date["year_int"].to_list() == [2020]

    def test_season_no_year(self, season_config: SeasonConfig) -> None:
        expr = resolve_time_filter(
            {"season": "kharif"},
            time_col="date",
            season_config=season_config,
        )
        assert isinstance(expr, pl.Expr)

    def test_season_single_year(self, season_config: SeasonConfig) -> None:
        expr = resolve_time_filter(
            {"season": "kharif", "year": 2021},
            time_col="date",
            season_config=season_config,
        )
        assert isinstance(expr, pl.Expr)

    def test_season_tuple_year(self, season_config: SeasonConfig) -> None:
        expr = resolve_time_filter(
            {"season": "kharif", "year": (2020, 2022)},
            time_col="date",
            season_config=season_config,
        )
        assert isinstance(expr, pl.Expr)

    def test_season_current(
        self, season_config: SeasonConfig, monkeypatch: Any
    ) -> None:
        # Patch date.today() to return a known date (July 15 -> Kharif)
        class MockDate(datetime.date):
            @classmethod
            def today(cls) -> "MockDate":
                return cls(2021, 7, 15)

        monkeypatch.setattr(datetime, "date", MockDate)

        expr = resolve_time_filter(
            {"season": "current"},
            time_col="date",
            season_config=season_config,
        )
        assert isinstance(expr, pl.Expr)

    def test_year_crossing_season(self, season_config: SeasonConfig) -> None:
        expr = resolve_time_filter(
            {"season": "rabi", "year": 2021},
            time_col="date",
            season_config=season_config,
        )
        assert isinstance(expr, pl.Expr)

    def test_invalid_structure_raises(self, season_config: SeasonConfig) -> None:
        with pytest.raises(ValueError, match="Unrecognised time_filter structure"):
            resolve_time_filter(
                {"invalid": "format"},
                time_col="date",
                season_config=season_config,
            )


def test_parse_md() -> None:
    assert _parse_md("06-15") == (6, 15)


def test_year_bounds() -> None:
    assert _year_bounds(None) == (1900, 2100)
    assert _year_bounds(2020) == (2020, 2020)
    assert _year_bounds((2020, 2025)) == (2020, 2025)


from core_lens.utils.season import add_temporal_columns  # noqa: E402


_DATES = [
    datetime.date(2022, 8, 1),  # kharif
    datetime.date(2022, 12, 15),  # rabi
    datetime.date(2023, 5, 10),  # zaid
]
_EXPECTED_SEASON_YEAR = ["kharif_2022", "rabi_2022", "zaid_2023"]


class TestAddTemporalColumns:
    """Tests for the ``add_temporal_columns`` guard logic.

    Regression suite for the ``or``→``and`` bug where ``season_year`` was
    added even when it already existed in the incoming DataFrame, causing a
    ``polars.exceptions.DuplicateError``.
    """

    def test_normal_path_all_columns_derived(self, season_config: SeasonConfig) -> None:
        """All five temporal columns are added when none are pre-existing."""
        df = pl.DataFrame({"date": _DATES, "val": [1, 2, 3]})
        result = add_temporal_columns(df, "date", season_config)
        assert isinstance(result, pl.DataFrame)

        assert "year" in result.columns
        assert "month" in result.columns
        assert "year_month" in result.columns
        assert "season" in result.columns
        assert "season_year" in result.columns
        assert result["season_year"].to_list() == _EXPECTED_SEASON_YEAR

    def test_season_year_preexisting_is_not_overwritten(
        self, season_config: SeasonConfig
    ) -> None:
        """``season_year`` already present in Parquet data must not be replaced.

        This is the primary regression case: the old ``or`` guard entered the
        block whenever ``season`` was absent and tried to add ``season_year``
        again, raising ``DuplicateError``.
        """
        df = pl.DataFrame(
            {"date": _DATES, "val": [1, 2, 3], "season_year": ["x", "y", "z"]}
        )
        result = add_temporal_columns(df, "date", season_config)
        assert isinstance(result, pl.DataFrame)

        # season_year must be left untouched.
        assert result["season_year"].to_list() == ["x", "y", "z"]
        # season should still be derived (it was absent).
        assert "season" in result.columns

    def test_season_preexisting_season_year_derived(
        self, season_config: SeasonConfig
    ) -> None:
        """``season`` present, ``season_year`` absent → derive ``season_year`` only."""
        df = pl.DataFrame(
            {
                "date": _DATES,
                "val": [1, 2, 3],
                "season": ["kharif", "rabi", "zaid"],
            }
        )
        result = add_temporal_columns(df, "date", season_config)
        assert isinstance(result, pl.DataFrame)

        assert result["season_year"].to_list() == _EXPECTED_SEASON_YEAR
        # Original season column should be unchanged.
        assert result["season"].to_list() == ["kharif", "rabi", "zaid"]

    def test_all_preexisting_noop(self, season_config: SeasonConfig) -> None:
        """When all five columns are pre-existing, nothing is overwritten."""
        df = pl.DataFrame(
            {
                "date": _DATES,
                "val": [1, 2, 3],
                "year": [2022, 2022, 2023],
                "month": [8, 12, 5],
                "year_month": ["2022-08", "2022-12", "2023-05"],
                "season": ["kharif", "rabi", "zaid"],
                "season_year": ["A", "B", "C"],
            }
        )
        result = add_temporal_columns(df, "date", season_config)
        assert isinstance(result, pl.DataFrame)

        # All pre-existing values must be preserved verbatim.
        assert result["season_year"].to_list() == ["A", "B", "C"]
        assert result["season"].to_list() == ["kharif", "rabi", "zaid"]
        assert result["year"].to_list() == [2022, 2022, 2023]
        assert result["month"].to_list() == [8, 12, 5]
        assert result["year_month"].to_list() == ["2022-08", "2022-12", "2023-05"]

    def test_lazy_frame_input(self, season_config: SeasonConfig) -> None:
        """Function accepts and returns a ``pl.LazyFrame`` transparently."""
        lf = pl.DataFrame({"date": _DATES, "val": [1, 2, 3]}).lazy()
        result_lf = add_temporal_columns(lf, "date", season_config)

        assert isinstance(result_lf, pl.LazyFrame)
        result = result_lf.collect()
        assert result["season_year"].to_list() == _EXPECTED_SEASON_YEAR

    def test_season_and_season_year_both_preexisting(
        self, season_config: SeasonConfig
    ) -> None:
        """Both season and season_year present → block skipped entirely, no duplicate."""
        df = pl.DataFrame(
            {
                "date": _DATES,
                "val": [1, 2, 3],
                "season": ["kharif", "rabi", "zaid"],
                "season_year": ["kharif_2022", "rabi_2022", "zaid_2023"],
            }
        )
        # Must not raise DuplicateError.
        result = add_temporal_columns(df, "date", season_config)
        assert isinstance(result, pl.DataFrame)
        assert result["season_year"].to_list() == _EXPECTED_SEASON_YEAR
        assert result["season"].to_list() == ["kharif", "rabi", "zaid"]
