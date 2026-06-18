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
