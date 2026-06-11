"""Tests for ``core_lens.base.namespaces.stats.StatsNamespace``."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from core_lens.base.result import Result
from core_lens.base.namespaces.stats import CorrelationError
from core_lens.schema.profile import Resolution


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
        out = r.stats.correlate(["ndvi", "rainfall"], method="pearson")
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
        out = r.stats.correlate(["ndvi", "rainfall"], method="spearman", across="time")
        assert out.metadata["method"] == "spearman"
        assert out.metadata["across"] == "time"
        assert "n_observations" in out.metadata

    def test_single_col_raises_correlation_error(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(CorrelationError):
            r.stats.correlate(["ndvi"])

    def test_invalid_method_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(ValueError, match="Unknown method"):
            r.stats.correlate(["ndvi", "rainfall"], method="bad")

    def test_spearman_accepted(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.correlate(["ndvi", "rainfall"], method="spearman")
        assert out.metadata["method"] == "spearman"

    def test_kendall_accepted(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.correlate(["ndvi", "rainfall"], method="kendall")
        assert out.metadata["method"] == "kendall"


class TestStatisticalTest:
    def test_group_based_returns_result(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", groups="zone", method="mann-whitney")
        assert isinstance(out, Result)

    def test_group_data_schema(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        df = r.stats.test(column="ndvi", groups="zone").df()
        for col in ("group", "n", "mean", "std", "median"):
            assert col in df.columns

    def test_group_metadata(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", groups="zone", method="t-test")
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
            method="t-test",
        )
        df = out.df()
        assert len(df) == 2
        assert "group" in df.columns

    def test_single_sample(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", against=0.5, method="t-test")
        assert "p_value" in out.metadata
        assert out.df()["group"][0] == "all"

    def test_autoselect_method(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", groups="zone")
        assert out.metadata["method"] in {"t-test", "mann-whitney"}

    def test_no_mode_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(ValueError, match="Provide exactly one of"):
            r.stats.test(column="ndvi")

    def test_invalid_method_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(ValueError, match="Unknown method"):
            r.stats.test(column="ndvi", groups="zone", method="bad")

    def test_significant_flag_bool(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        out = r.stats.test(column="ndvi", groups="zone")
        assert isinstance(out.metadata["significant"], bool)


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
        out = r.stats.change("ndvi", 2021, 2022, method="percentage")
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
        out = r.stats.change("ndvi", 2020, 2022, method="trend")
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
        out = r.stats.change("ndvi", 2020, 2022, method="trend")
        assert out.df()["direction"][0] == "increasing"

    def test_invalid_method_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), _big_df())
        with pytest.raises(ValueError, match="Unknown method"):
            r.stats.change("ndvi", 2021, 2022, method="bad")


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
        df = r.stats.anomaly("ndvi", mode="cross_sectional", method="zscore").df()
        assert "anomaly_score" in df.columns
        assert "is_anomaly" in df.columns

    def test_zscore_metadata(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        meta = r.stats.anomaly("ndvi", mode="cross_sectional", method="zscore").metadata
        assert meta["mode"] == "cross_sectional"
        assert meta["method"] == "zscore"
        assert "baseline_mean" in meta

    def test_iqr_metadata(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        meta = r.stats.anomaly("ndvi", mode="cross_sectional", method="iqr").metadata
        assert "baseline_mean" in meta

    def test_percentile_metadata(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        meta = r.stats.anomaly(
            "ndvi", mode="cross_sectional", method="percentile"
        ).metadata
        assert "baseline_mean" in meta

    def test_iqr_accepted(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        out = r.stats.anomaly("ndvi", mode="cross_sectional", method="iqr")
        assert out.metadata["method"] == "iqr"

    def test_percentile_accepted(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        out = r.stats.anomaly("ndvi", mode="cross_sectional", method="percentile")
        assert out.metadata["method"] == "percentile"

    def test_threshold_accepted(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        out = r.stats.anomaly("ndvi", mode="cross_sectional", method="threshold")
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
            r.stats.anomaly("ndvi", mode="cross_sectional", method="zscore")

    def test_invalid_method_raises(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        with pytest.raises(ValueError, match="does not support method"):
            r.stats.anomaly(column="ndvi", mode="cross_sectional", method="stl")

    def test_is_anomaly_boolean(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        df = r.stats.anomaly("ndvi", mode="cross_sectional", method="zscore").df()
        assert df["is_anomaly"].dtype == pl.Boolean

    def test_row_count_matches_input(self, entity_cls: Any) -> None:
        r = self._big_cs(entity_cls())
        df = r.stats.anomaly("ndvi", mode="cross_sectional", method="zscore").df()
        assert len(df) == 20


class TestAnomalyTimeseries:
    def _ts_df(self) -> pl.DataFrame:
        """Two entities × 16 annual rows (baseline 8 + eval 8)."""
        import numpy as np

        rng = np.random.default_rng(7)
        years = list(range(2010, 2026))
        rows = []
        for eid in ["a", "b"]:
            vals = rng.normal(0.6, 0.05, len(years)).tolist()
            for yr, v in zip(years, vals):
                rows.append({"mws_id": eid, "year": yr, "ndvi": v})
        return pl.DataFrame(rows)

    def test_mad_schema(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        df = r.stats.anomaly(
            "ndvi", mode="timeseries", method="mad", baseline=(2010, 2017)
        ).df()
        for col in ("mws_id", "year", "anomaly_score", "is_anomaly"):
            assert col in df.columns

    def test_mad_baseline_excluded(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        df = r.stats.anomaly(
            "ndvi", mode="timeseries", method="mad", baseline=(2010, 2017)
        ).df()
        assert all(yr > 2017 for yr in df["year"].to_list())

    def test_cusum_accepted(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        out = r.stats.anomaly(
            "ndvi", mode="timeseries", method="cusum", baseline=(2010, 2017)
        )
        assert out.metadata["method"] == "cusum"

    def test_stl_accepted(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        out = r.stats.anomaly(
            "ndvi", mode="timeseries", method="stl", baseline=(2010, 2017)
        )
        assert out.metadata["method"] == "stl"

    def test_timeseries_metadata(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        meta = r.stats.anomaly(
            "ndvi", mode="timeseries", method="mad", baseline=(2010, 2017)
        ).metadata
        assert meta["mode"] == "timeseries"
        assert meta["baseline"] == (2010, 2017)
        assert meta["baseline_fitted"] is True

    def test_no_baseline_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        with pytest.raises(ValueError, match="baseline"):
            r.stats.anomaly("ndvi", mode="timeseries", method="mad")

    def test_invalid_mode_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        with pytest.raises(ValueError, match="Unknown mode"):
            r.stats.anomaly("ndvi", mode="spatial", method="mad")

    def test_invalid_ts_method_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._ts_df())
        with pytest.raises(ValueError, match="does not support method"):
            r.stats.anomaly(
                column="ndvi", mode="timeseries", method="zscore", baseline=(2010, 2017)
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
            method="manhattan",
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
        out = r.stats.similarity(target="id_0", columns={"ndvi": None}, method="cosine")
        assert out.metadata["method"] == "cosine"

    def test_invalid_method_raises(self, entity_cls: Any) -> None:
        r = _make(entity_cls(), self._sim_df())
        with pytest.raises(ValueError, match="Unknown method"):
            r.stats.similarity(target="id_0", columns={"ndvi": None}, method="bad")
