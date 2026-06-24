"""Tests for ``core_lens.base.namespaces.stats.StatsNamespace``."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from core_lens.base.result import Result
from core_lens.base.namespaces.stats import (
    CorrelationError,
    CorrelateMethod,
    TestMethod,
    ChangeMethod,
    AnomalyCrossMethod,
    AnomalyTsMethod,
    SimilarityMethod,
    _sf,
)
from core_lens.schema.profile import Resolution


class TestStatsUtils:
    def test_sf_none(self) -> None:
        import math

        assert math.isnan(_sf(None))

    def test_year_col_from_profile(self, entity_cls_full: Any) -> None:
        r = _make(entity_cls_full(), pl.DataFrame({"mws_id": ["1"]}))
        assert r.stats._year_col() == "year"


def _make(
    entity: Any,
    data: pl.DataFrame,
    resolution: Resolution = Resolution.ANNUAL,
) -> Result:
    return Result(
        data=data,
        resolution=resolution,
        has_geometry=False,
        key_cols=["mws_id"],
        entity_name="mws",
        entity=entity,
    )


def _big_df(n: int = 20) -> pl.DataFrame:
    """20-row frame with two numeric cols for stats tests."""
    import numpy as np

    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "mws_id": [f"id_{i}" for i in range(n)],
            "ndvi": rng.uniform(0.2, 0.9, n).tolist(),
            "rainfall": rng.uniform(400, 1200, n).tolist(),
            "zone": (["high"] * (n // 2) + ["low"] * (n // 2)),
            "year": ([2021] * (n // 2) + [2022] * (n // 2)),
        }
    )


class TestDescribeByColumn:
    def test_returns_result(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.describe()
        assert isinstance(out, Result)

    def test_schema(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.describe().df()
        for col in (
            "column",
            "count",
            "mean",
            "std",
            "min",
            "p25",
            "median",
            "p75",
            "max",
        ):
            assert col in df.columns

    def test_row_per_numeric_col(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.describe().df()
        # ndvi, rainfall, year are numeric
        assert len(df) >= 2

    def test_column_subset(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.describe(columns=["ndvi"]).df()
        assert list(df["column"]) == ["ndvi"]

    def test_count_correct(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df(20))
        df = r.stats.describe(columns=["ndvi"]).df()
        assert df["count"][0] == 20

    def test_metadata_empty(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.describe()
        assert out.metadata == {}


class TestDescribeByEntity:
    def test_by_entity_returns_result(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.describe(columns=["ndvi", "rainfall"], by="entity")
        assert isinstance(out, Result)

    def test_by_entity_has_key_col(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.describe(columns=["ndvi"], by="entity").df()
        assert "mws_id" in df.columns

    def test_by_unknown_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(ValueError, match="Unknown by"):
            r.stats.describe(by="time")


class TestCorrelate:
    def test_pearson_returns_result(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.correlate(["ndvi", "rainfall"], method=CorrelateMethod.PEARSON)
        assert isinstance(out, Result)

    def test_output_schema(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.correlate(["ndvi", "rainfall"]).df()
        for col in ("column_a", "column_b", "correlation", "p_value"):
            assert col in df.columns

    def test_pair_count(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.correlate(["ndvi", "rainfall", "year"]).df()
        assert len(df) == 3  # C(3,2)

    def test_correlation_in_range(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.correlate(["ndvi", "rainfall"]).df()
        corr = df["correlation"][0]
        assert -1.0 <= corr <= 1.0

    def test_metadata_populated(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.correlate(
            ["ndvi", "rainfall"], method=CorrelateMethod.SPEARMAN, across="time"
        )
        assert out.metadata["method"] == "spearman"
        assert out.metadata["across"] == "time"
        assert "n_observations" in out.metadata

    def test_single_col_raises_correlation_error(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(CorrelationError):
            r.stats.correlate(["ndvi"])

    def test_invalid_method_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(ValueError, match="must be a "):
            r.stats.correlate(["ndvi", "rainfall"], method="bad")  # type: ignore[arg-type]

    def test_spearman_accepted(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.correlate(["ndvi", "rainfall"], method=CorrelateMethod.SPEARMAN)
        assert out.metadata["method"] == "spearman"

    def test_kendall_accepted(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.correlate(["ndvi", "rainfall"], method=CorrelateMethod.KENDALL)
        assert out.metadata["method"] == "kendall"


class TestStatisticalTest:
    def test_group_based_returns_result(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", groups="zone", method=TestMethod.MANN_WHITNEY)
        assert isinstance(out, Result)

    def test_group_data_schema(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.test(column="ndvi", groups="zone").df()
        for col in ("group", "n", "mean", "std", "median"):
            assert col in df.columns

    def test_group_metadata(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", groups="zone", method=TestMethod.T_TEST)
        for key in (
            "method",
            "statistic",
            "p_value",
            "significant",
            "significance_level",
        ):
            assert key in out.metadata

    def test_period_based(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(
            column="ndvi",
            periods=[(2021, 2021), (2022, 2022)],
            method=TestMethod.T_TEST,
        )
        df = out.df()
        assert len(df) == 2
        assert "group" in df.columns

    def test_single_sample(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", against=0.5, method=TestMethod.T_TEST)
        assert "p_value" in out.metadata
        assert out.df()["group"][0] == "all"

    def test_autoselect_method(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", groups="zone")
        assert out.metadata["method"] in {"t-test", "mann-whitney"}

    def test_autoselect_method_few_samples(self, entity_cls: Any) -> None:
        df = pl.DataFrame(
            {"mws_id": ["1", "2"], "ndvi": [0.5, 0.6], "zone": ["A", "B"]}
        )
        r = _make(entity_cls(), df)
        out = r.stats.test(column="ndvi", groups="zone")
        assert out.metadata["method"] == "mann-whitney"

    def test_wilcoxon_single_sample(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", against=0.5, method=TestMethod.WILCOXON)
        assert out.metadata["method"] == "wilcoxon"
        assert "p_value" in out.metadata

    def test_invalid_against_method(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(ValueError, match="not valid for single-sample"):
            r.stats.test(column="ndvi", against=0.5, method=TestMethod.CHI_SQUARE)

    def test_periods_no_year_col_raises(self, entity_cls: Any) -> None:
        df = pl.DataFrame({"mws_id": ["1"], "ndvi": [0.5]})
        r = _make(entity_cls(), df)
        with pytest.raises(ValueError, match="requires a year column"):
            r.stats.test(column="ndvi", periods=[(2021, 2021)])

    def test_run_test_kruskal(self, entity_cls: Any) -> None:
        df = pl.DataFrame(
            {
                "mws_id": ["1", "2", "3"],
                "ndvi": [0.5, 0.6, 0.7],
                "zone": ["A", "B", "C"],
            }
        )
        r = _make(entity_cls(), df)
        out = r.stats.test(column="ndvi", groups="zone", method=TestMethod.T_TEST)
        assert len(out.df()) == 3

    def test_run_test_wilcoxon_groups(self, entity_cls: Any) -> None:
        df = pl.DataFrame(
            {"mws_id": ["1", "2"], "ndvi": [0.5, 0.6], "zone": ["A", "B"]}
        )
        r = _make(entity_cls(), df)
        out = r.stats.test(column="ndvi", groups="zone", method=TestMethod.WILCOXON)
        assert out.metadata["method"] == "wilcoxon"

    def test_run_test_ks(self, entity_cls: Any) -> None:
        df = pl.DataFrame(
            {"mws_id": ["1", "2"], "ndvi": [0.5, 0.6], "zone": ["A", "B"]}
        )
        r = _make(entity_cls(), df)
        out = r.stats.test(column="ndvi", groups="zone", method=TestMethod.KS)
        assert out.metadata["method"] == "ks"

    def test_run_test_chisquare(self, entity_cls: Any) -> None:
        df = pl.DataFrame(
            {"mws_id": ["1", "2"], "ndvi": [0.5, 0.6], "zone": ["A", "B"]}
        )
        r = _make(entity_cls(), df)
        out = r.stats.test(column="ndvi", groups="zone", method=TestMethod.CHI_SQUARE)
        assert out.metadata["method"] == "chi-square"

    def test_no_mode_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        import pytest

        with pytest.raises(ValueError, match="Provide exactly one of"):
            r.stats.test(column="ndvi")


class TestChange:
    def test_absolute_returns_result(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.change("ndvi", from_period=2021, to_period=2022)
        assert isinstance(out, Result)

    def test_absolute_schema(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.change("ndvi", from_period=2021, to_period=2022).df()
        for col in ("mws_id", "value_from", "value_to", "change", "pct_change"):
            assert col in df.columns

    def test_percentage_metadata(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.change("ndvi", 2021, 2022, method=ChangeMethod.PERCENTAGE)
        assert out.metadata["method"] == "percentage"
        assert out.metadata["column"] == "ndvi"

    def test_trend_schema(self, entity_cls: Any) -> None:
        # need entities with multiple years — use a frame with 3 years per entity
        df = pl.DataFrame(
            {
                "mws_id": ["a", "a", "a", "b", "b", "b"],
                "year": [2020, 2021, 2022, 2020, 2021, 2022],
                "ndvi": [0.3, 0.4, 0.5, 0.6, 0.55, 0.5],
            }
        )
        r = _make(entity_cls(), df)
        out = r.stats.change("ndvi", 2020, 2022, method=ChangeMethod.TREND)
        result_df = out.df()
        for col in ("mws_id", "slope", "r_squared", "direction"):
            assert col in result_df.columns

    def test_trend_direction(self, entity_cls: Any) -> None:
        df = pl.DataFrame(
            {
                "mws_id": ["a", "a", "a"],
                "year": [2020, 2021, 2022],
                "ndvi": [0.3, 0.4, 0.5],
            }
        )
        r = _make(entity_cls(), df)
        out = r.stats.change("ndvi", 2020, 2022, method=ChangeMethod.TREND)
        assert out.df()["direction"][0] == "increasing"

    def test_trend_len_less_than_2(self, entity_cls: Any) -> None:
        df = pl.DataFrame(
            {
                "mws_id": ["a", "b", "b"],
                "year": [2020, 2021, 2022],
                "ndvi": [0.3, 0.4, 0.5],
            }
        )
        r = _make(entity_cls(), df)
        out = r.stats.change("ndvi", 2020, 2022, method=ChangeMethod.TREND)
        # 'a' has only 1 data point so it will be skipped
        assert len(out.df()) == 1

    def test_change_no_year_col_raises(self, entity_cls: Any) -> None:
        df = pl.DataFrame({"mws_id": ["1"], "ndvi": [0.5]})
        r = _make(entity_cls(), df)
        with pytest.raises(ValueError, match="Requires a year/time column"):
            r.stats.change("ndvi", 2020, 2022)

    def test_invalid_method_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(ValueError, match="must be a "):
            r.stats.change("ndvi", 2021, 2022, method="bad")  # type: ignore[arg-type]


class TestAnomalyCrossSectional:
    def _big_cs(self, entity: Any) -> Result:
        import numpy as np

        rng = np.random.default_rng(0)
        return _make(
            entity,
            pl.DataFrame(
                {
                    "mws_id": [f"id_{i}" for i in range(20)],
                    "ndvi": rng.normal(0.6, 0.1, 20).tolist(),
                }
            ),
        )

    def test_zscore_schema(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        df = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.ZSCORE
        ).df()
        assert "anomaly_score" in df.columns
        assert "is_anomaly" in df.columns

    def test_zscore_metadata(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        meta = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.ZSCORE
        ).metadata
        assert meta["mode"] == "cross_sectional"
        assert meta["method"] == "zscore"
        assert "baseline_mean" in meta

    def test_iqr_metadata(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        meta = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.IQR
        ).metadata
        assert "baseline_mean" in meta

    def test_percentile_metadata(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        meta = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.PERCENTILE
        ).metadata
        assert "baseline_mean" in meta

    def test_iqr_accepted(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        out = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.IQR
        )
        assert out.metadata["method"] == "iqr"

    def test_percentile_accepted(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        out = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.PERCENTILE
        )
        assert out.metadata["method"] == "percentile"

    def test_threshold_accepted(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        out = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.THRESHOLD
        )
        assert out.metadata["method"] == "threshold"

    def test_too_few_obs_raises(self, entity_cls: Any) -> None:
        r = _make(
            entity_cls(),
            pl.DataFrame(
                {
                    "mws_id": ["a", "b"],
                    "ndvi": [0.5, 0.6],
                }
            ),
        )
        with pytest.raises(ValueError, match="observations"):
            r.stats.anomaly(
                "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.ZSCORE
            )

    def test_invalid_method_raises(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        with pytest.raises(ValueError, match="does not support method"):
            r.stats.anomaly(
                column="ndvi", mode="cross_sectional", method=AnomalyTsMethod.STL
            )

    def test_is_anomaly_boolean(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        df = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.ZSCORE
        ).df()
        assert df["is_anomaly"].dtype == pl.Boolean

    def test_row_count_matches_input(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        df = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method=AnomalyCrossMethod.ZSCORE
        ).df()
        assert len(df) == 20

    def test_baseline_without_year_col(self, entity_cls: Any) -> None:
        df = pl.DataFrame(
            {"mws_id": [f"id_{i}" for i in range(20)], "ndvi": [0.5] * 20}
        )
        r = _make(entity_cls(), df)
        out = r.stats.anomaly(
            "ndvi",
            mode="cross_sectional",
            method=AnomalyCrossMethod.ZSCORE,
            baseline=(2020, 2022),
        )
        assert len(out.df()) == 20


class TestAnomalyTimeseries:
    def _ts_df(self) -> pl.DataFrame:
        """Two entities × 16 annual rows (baseline 8 + eval 8)."""
        import numpy as np

        rng = np.random.default_rng(7)
        years = list(range(2010, 2050))
        rows = []
        for eid in ["a", "b"]:
            vals = rng.normal(0.6, 0.05, len(years)).tolist()
            for yr, v in zip(years, vals):
                rows.append({"mws_id": eid, "year": yr, "ndvi": v})
        return pl.DataFrame(rows)

    def test_mad_schema(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        df = r.stats.anomaly(
            "ndvi", mode="timeseries", method=AnomalyTsMethod.MAD, baseline=(2010, 2017)
        ).df()
        for col in ("mws_id", "year", "anomaly_score", "is_anomaly"):
            assert col in df.columns

    def test_mad_baseline_excluded(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        df = r.stats.anomaly(
            "ndvi", mode="timeseries", method=AnomalyTsMethod.MAD, baseline=(2010, 2017)
        ).df()
        assert all(yr > 2017 for yr in df["year"].to_list())

    def test_cusum_accepted(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        out = r.stats.anomaly(
            "ndvi",
            mode="timeseries",
            method=AnomalyTsMethod.CUSUM,
            baseline=(2010, 2017),
        )
        assert out.metadata["method"] == "cusum"

    def test_stl_accepted(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        out = r.stats.anomaly(
            "ndvi", mode="timeseries", method=AnomalyTsMethod.STL, baseline=(2010, 2035)
        )
        assert out.metadata["method"] == "stl"

    def test_timeseries_metadata(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        meta = r.stats.anomaly(
            "ndvi", mode="timeseries", method=AnomalyTsMethod.MAD, baseline=(2010, 2017)
        ).metadata
        assert meta["mode"] == "timeseries"
        assert meta["baseline"] == (2010, 2017)
        assert meta["baseline_fitted"] is True

    def test_stl_exception_handling(self, entity_cls: Any, monkeypatch: Any) -> None:
        def mock_stl_init(*args: Any, **kwargs: Any) -> Any:
            raise Exception("Forced STL error")

        import statsmodels.tsa.seasonal  # type: ignore

        monkeypatch.setattr(statsmodels.tsa.seasonal, "STL", mock_stl_init)
        r = _make(entity_cls(), self._ts_df())
        out = r.stats.anomaly(
            "ndvi", mode="timeseries", method=AnomalyTsMethod.STL, baseline=(2010, 2035)
        )
        # Fallback will create NaN scores and False flags
        import math

        assert math.isnan(out.df()["anomaly_score"][0])
        assert not out.df()["is_anomaly"][0]

    def test_timeseries_no_year_col_raises(self, entity_cls: Any) -> None:
        df = pl.DataFrame({"mws_id": ["1", "1"], "ndvi": [0.5, 0.6]})
        r = _make(entity_cls(), df)
        with pytest.raises(ValueError, match="requires a year/time column"):
            r.stats.anomaly(
                "ndvi",
                mode="timeseries",
                method=AnomalyTsMethod.MAD,
                baseline=(2010, 2017),
            )

    def test_no_baseline_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        with pytest.raises(ValueError, match="baseline"):
            r.stats.anomaly("ndvi", mode="timeseries", method=AnomalyTsMethod.MAD)

    def test_invalid_mode_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        with pytest.raises(ValueError, match="Unknown mode"):
            r.stats.anomaly("ndvi", mode="spatial", method=AnomalyTsMethod.MAD)

    def test_invalid_ts_method_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        with pytest.raises(ValueError, match="does not support method"):
            r.stats.anomaly(
                column="ndvi",
                mode="timeseries",
                method=AnomalyCrossMethod.ZSCORE,
                baseline=(2010, 2017),
            )


class TestSimilarity:
    def _sim_df(self, n: int = 15) -> pl.DataFrame:
        import numpy as np

        rng = np.random.default_rng(99)
        return pl.DataFrame(
            {
                "mws_id": [f"id_{i}" for i in range(n)],
                "ndvi": rng.uniform(0.2, 0.9, n).tolist(),
                "rainfall": rng.uniform(400, 1200, n).tolist(),
            }
        )

    def test_euclidean_returns_result(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        out = r.stats.similarity(
            target="id_0",
            columns={"ndvi": None, "rainfall": None},
        )
        assert isinstance(out, Result)

    def test_output_schema(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        df = r.stats.similarity(
            target="id_0",
            columns={"ndvi": None, "rainfall": None},
        ).df()
        for col in ("mws_id", "similarity_score", "rank"):
            assert col in df.columns

    def test_option3_columns(self, entity_cls: Any) -> None:
        df = pl.DataFrame({"mws_id": ["MW_1", "MW_2", "MW_3"], "ndvi": [0.5, 0.6, 0.7]})
        r = _make(entity_cls(), df)
        out = r.stats.similarity(
            target="MW_1", columns={"ndvi": None, "state": ("static", None)}
        )
        assert isinstance(out, Result)
        assert "state" not in out.df().columns  # Only score and rank are returned
        assert "similarity_score" in out.df().columns

    def test_option3_exhaustive(self, entity_cls_full: Any) -> None:
        df = pl.DataFrame(
            {"mws_id": ["13_001", "13_002", "13_003"], "ndvi": [0.5, 0.6, 0.7]}
        )
        r = _make(entity_cls_full(), df)

        # 1. Valid resolution + filter_dict
        out = r.stats.similarity(
            target="13_001",
            columns={"ndvi": None, "ndvi_mean": ("annual", {"year": 2021})},
        )
        assert "similarity_score" in out.df().columns

        # 2. Invalid spec tuple
        with pytest.raises(
            ValueError, match="must be None or a \\(resolution, filter_dict\\)"
        ):
            r.stats.similarity(target="13_001", columns={"state": "static"})

        # 3. Invalid resolution
        with pytest.raises(ValueError, match="Unknown resolution"):
            r.stats.similarity(target="13_001", columns={"state": ("bad", None)})

        # 4. Missing column in schema (silently skipped, leading to no valid columns)
        with pytest.raises(ValueError, match="None of the specified columns"):
            r.stats.similarity(target="13_001", columns={"ghost_col": ("static", None)})

    def test_option3_fortnightly_and_filters(self, entity_cls_full: Any) -> None:
        import sys
        import os

        conftest_dir = os.path.dirname(__file__)
        if conftest_dir not in sys.path:
            sys.path.insert(0, conftest_dir)
        from conftest import (
            _make_static_parquet,
            _make_annual_parquet,
            _make_fortnightly_parquet,
            _make_entity_cls,
        )
        import tempfile
        import pathlib

        with tempfile.TemporaryDirectory() as td:
            sp = pathlib.Path(td) / "static.parquet"
            ap = pathlib.Path(td) / "annual.parquet"
            fp = pathlib.Path(td) / "fortnightly.parquet"
            _make_static_parquet(sp)
            _make_annual_parquet(ap)
            _make_fortnightly_parquet(fp)
            full_cls = _make_entity_cls(sp, annual=ap, fortnightly=fp)

            df = pl.DataFrame({"mws_id": ["13_001", "13_002"], "ndvi": [0.5, 0.6]})
            r = _make(full_cls(), df)

            out = r.stats.similarity(
                target="13_001",
                columns={
                    "ndvi_mean": ("annual", {"year": 2021, "agg": "max"}),
                    "ndvi": ("fortnightly", {"agg": "mean"}),
                },
            )
            assert "similarity_score" in out.df().columns

    def test_similarity_file_not_found(
        self, entity_cls_full: Any, monkeypatch: Any
    ) -> None:
        df = pl.DataFrame({"mws_id": ["13_001", "13_002"], "ndvi": [0.5, 0.6]})
        r = _make(entity_cls_full(), df)

        def mock_resolve(path: str) -> str:
            raise FileNotFoundError("Mocked file not found")

        monkeypatch.setattr(r.entity, "_resolve", mock_resolve)

        import pytest

        with pytest.raises(ValueError, match="None of the specified columns"):
            r.stats.similarity(
                target="13_001",
                columns={"ndvi": ("static", None)},
            )

    def test_similarity_mahalanobis(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df(20))
        out = r.stats.similarity(
            target="id_0",
            columns={"ndvi": None, "rainfall": None},
            method=SimilarityMethod.MAHALANOBIS,
        )
        assert "similarity_score" in out.df().columns

    def test_similarity_mahalanobis_exception(
        self, entity_cls: Any, monkeypatch: Any
    ) -> None:
        def mock_pinv(*args: Any, **kwargs: Any) -> Any:
            raise Exception("PINV error")

        import numpy.linalg

        monkeypatch.setattr(numpy.linalg, "pinv", mock_pinv)

        r = _make(entity_cls(), self._sim_df(20))
        out = r.stats.similarity(
            target="id_0",
            columns={"ndvi": None, "rainfall": None},
            method=SimilarityMethod.MAHALANOBIS,
        )
        assert "similarity_score" in out.df().columns

    def test_top_n_respected(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df(15))
        df = r.stats.similarity(
            target="id_0",
            columns={"ndvi": None, "rainfall": None},
            top_n=5,
        ).df()
        assert len(df) == 5

    def test_rank_starts_at_1(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        df = r.stats.similarity(
            target="id_0",
            columns={"ndvi": None, "rainfall": None},
            top_n=3,
        ).df()
        assert df["rank"].to_list() == [1, 2, 3]

    def test_target_excluded_from_output(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        df = r.stats.similarity(
            target="id_0",
            columns={"ndvi": None},
        ).df()
        assert "id_0" not in df["mws_id"].to_list()

    def test_similarity_score_positive(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        df = r.stats.similarity(
            target="id_0",
            columns={"ndvi": None, "rainfall": None},
        ).df()
        assert all(s > 0 for s in df["similarity_score"].to_list())

    def test_metadata_populated(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        out = r.stats.similarity(
            target="id_0",
            columns={"ndvi": None},
            method=SimilarityMethod.MANHATTAN,
            top_n=3,
        )
        assert out.metadata["method"] == "manhattan"
        assert out.metadata["target"] == "id_0"
        assert out.metadata["top_n"] == 3

    def test_target_not_found_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        with pytest.raises(ValueError, match="not found"):
            r.stats.similarity(target="ghost", columns={"ndvi": None})

    def test_no_matching_columns_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        with pytest.raises(ValueError, match="None of"):
            r.stats.similarity(target="id_0", columns={"missing_col": None})

    def test_cosine_accepted(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        out = r.stats.similarity(
            target="id_0", columns={"ndvi": None}, method=SimilarityMethod.COSINE
        )
        assert out.metadata["method"] == "cosine"

    def test_invalid_method_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        with pytest.raises(ValueError, match="must be a "):
            r.stats.similarity(target="id_0", columns={"ndvi": None}, method="bad")  # type: ignore[arg-type]
