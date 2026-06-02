"""Tests for ``core_lens.aoi.SeasonConfig`` and the ``_md_in_range`` helper."""

from __future__ import annotations

import datetime

import pytest

from core_lens.aoi import SeasonConfig, _md_in_range


class TestMdInRange:
    def test_within_non_crossing_range(self):
        assert _md_in_range("08-15", "07-01", "10-30") is True

    def test_before_non_crossing_range(self):
        assert _md_in_range("06-30", "07-01", "10-30") is False

    def test_after_non_crossing_range(self):
        assert _md_in_range("11-01", "07-01", "10-30") is False

    def test_at_start_boundary(self):
        assert _md_in_range("07-01", "07-01", "10-30") is True

    def test_at_end_boundary(self):
        assert _md_in_range("10-30", "07-01", "10-30") is True

    def test_within_year_crossing_range(self):
        assert _md_in_range("01-15", "11-01", "03-31") is True

    def test_within_year_crossing_range_late_portion(self):
        assert _md_in_range("12-01", "11-01", "03-31") is True

    def test_outside_year_crossing_range(self):
        assert _md_in_range("07-01", "11-01", "03-31") is False


class TestSeasonConfigDefaults:
    def test_kharif_mid_season(self):
        config = SeasonConfig()

        assert config.season_for(datetime.date(2022, 8, 15)) == "kharif"

    def test_rabi_mid_season(self):
        config = SeasonConfig()

        assert config.season_for(datetime.date(2022, 12, 25)) == "rabi"

    def test_rabi_crosses_year_boundary(self):
        config = SeasonConfig()

        assert config.season_for(datetime.date(2023, 1, 31)) == "rabi"

    def test_zaid_mid_season(self):
        config = SeasonConfig()

        assert config.season_for(datetime.date(2022, 5, 10)) == "zaid"

    def test_season_for_kharif_start_boundary(self):
        config = SeasonConfig()

        assert config.season_for(datetime.date(2022, 7, 1)) == "kharif"

    def test_season_for_kharif_end_boundary(self):
        config = SeasonConfig()

        assert config.season_for(datetime.date(2022, 10, 30)) == "kharif"

    def test_season_for_rabi_start_boundary(self):
        config = SeasonConfig()

        assert config.season_for(datetime.date(2022, 11, 1)) == "rabi"

    def test_season_for_zaid_start_boundary(self):
        config = SeasonConfig()

        assert config.season_for(datetime.date(2022, 4, 1)) == "zaid"

    def test_season_for_zaid_end_boundary(self):
        config = SeasonConfig()

        assert config.season_for(datetime.date(2022, 6, 30)) == "zaid"


class TestSeasonConfigCustom:
    def test_custom_ranges_override_defaults(self):
        config = SeasonConfig(
            kharif=("06-01", "10-15"),
            rabi=("10-16", "02-28"),
            zaid=("03-01", "05-31"),
        )

        assert config.season_for(datetime.date(2022, 6, 15)) == "kharif"
        assert config.season_for(datetime.date(2022, 11, 30)) == "rabi"
        assert config.season_for(datetime.date(2022, 4, 20)) == "zaid"

    def test_ranges_with_gap_raises_on_gap_date(self):
        config = SeasonConfig(
            kharif=("06-01", "09-30"),
            rabi=("11-01", "02-28"),
            zaid=("03-01", "05-31"),
        )

        with pytest.raises(ValueError, match="does not fall within"):
            config.season_for(datetime.date(2022, 10, 15))
