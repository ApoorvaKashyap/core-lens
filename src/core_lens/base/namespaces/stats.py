"""Statistical analysis namespace for core_lens Result objects."""

from __future__ import annotations

from enum import Enum
from itertools import combinations
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from core_lens.base.result import Result


class CorrelateMethod(Enum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"


class TestMethod(Enum):
    __test__ = False
    T_TEST = "t-test"
    MANN_WHITNEY = "mann-whitney"
    WILCOXON = "wilcoxon"
    KS = "ks"
    CHI_SQUARE = "chi-square"


class ChangeMethod(Enum):
    ABSOLUTE = "absolute"
    PERCENTAGE = "percentage"
    TREND = "trend"


class AnomalyCrossMethod(Enum):
    ZSCORE = "zscore"
    IQR = "iqr"
    PERCENTILE = "percentile"
    THRESHOLD = "threshold"


class AnomalyTsMethod(Enum):
    STL = "stl"
    CUSUM = "cusum"
    MAD = "mad"


class SimilarityMethod(Enum):
    EUCLIDEAN = "euclidean"
    COSINE = "cosine"
    MAHALANOBIS = "mahalanobis"
    MANHATTAN = "manhattan"


_MIN_OBS: dict[str, int] = {
    "stl": 24,
    "cusum": 8,
    "mad": 8,
    "zscore": 8,
    "iqr": 8,
    "percentile": 8,
    "threshold": 8,
}


class CorrelationError(ValueError):
    """Raised when correlation cannot be computed (e.g. fewer than 2 columns)."""


class StatsNamespace:
    """Statistical methods on ``result.stats.*``.

    All methods return a fresh :class:`~core_lens.base.result.Result`.
    Computed values always go in ``data``; method parameters go in ``metadata``.
    """

    def __init__(self, result: "Result") -> None:
        self._r = result

    def _numeric_cols(self, df: pl.DataFrame) -> list[str]:
        return [c for c in df.columns if df[c].dtype.is_numeric()]

    def _year_col(self) -> str | None:
        if "year" in self._r.data.columns:
            return "year"
        profile = self._r.entity.schema_profile
        return profile.annual_time_col

    def describe(
        self,
        columns: list[str] | None = None,
        by: str = "column",
    ) -> "Result":
        """Per-column or per-entity descriptive statistics.

        Args:
            columns: Numeric columns to describe. Defaults to all numeric cols.
            by: ``"column"`` (one row per column, design default) or
                ``"entity"`` (one row per entity with mean/std per column).

        Returns:
            New :class:`~core_lens.base.result.Result` with descriptive stats.

        Raises:
            ValueError: If ``by`` is not ``\"column\"`` or ``\"entity\"``.
        """
        df = self._r.data
        cols = columns if columns is not None else self._numeric_cols(df)

        if by == "column":
            rows = []
            for col in cols:
                s = df[col].drop_nulls()
                n = len(s)
                rows.append(
                    {
                        "column": col,
                        "count": n,
                        "mean": float(cast(float, s.mean())) if n > 0 else None,
                        "std": float(cast(float, s.std())) if n > 1 else None,
                        "min": float(cast(float, s.min())) if n > 0 else None,
                        "p25": float(cast(float, s.quantile(0.25))) if n > 0 else None,
                        "median": float(cast(float, s.median())) if n > 0 else None,
                        "p75": float(cast(float, s.quantile(0.75))) if n > 0 else None,
                        "max": float(cast(float, s.max())) if n > 0 else None,
                    }
                )
            data = pl.DataFrame(rows)

        elif by == "entity":
            key = self._r.key_cols[0]
            exprs = []
            for col in cols:
                exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
                exprs.append(pl.col(col).std().alias(f"{col}_std"))
            data = df.group_by(key).agg(exprs).sort(key)

        else:
            raise ValueError(
                f"StatsNamespace.describe: Unknown by={by!r}. Valid options: 'column', 'entity'."
            )

        return self._r._replace(data=data, has_geometry=False, metadata={})

    def correlate(
        self,
        columns: list[str],
        method: CorrelateMethod = CorrelateMethod.PEARSON,
        across: str = "entity",
    ) -> "Result":
        """Pairwise correlations between columns.

        Args:
            columns: At least 2 column names.
            method: A :class:`~core_lens.base.namespaces.stats.CorrelateMethod` enum value.
            across: ``"entity"`` or ``"time"`` — recorded in metadata only.

        Returns:
            Result whose data has columns
            ``column_a | column_b | correlation | p_value``.

        Raises:
            CorrelationError: If fewer than 2 columns supplied.
            ValueError: If ``method`` is not recognised.
        """
        if len(columns) < 2:
            raise CorrelationError(
                f"StatsNamespace.correlate: Requires at least 2 columns, got {len(columns)}."
            )
        if not isinstance(method, CorrelateMethod):
            raise ValueError(
                f"StatsNamespace.correlate: method must be a CorrelateMethod. Valid options: {[e.name for e in CorrelateMethod]}."
            )

        import scipy.stats as sp  # type: ignore[import-untyped]

        df = self._r.data
        n_obs = len(df)
        rows: list[dict[str, Any]] = []

        for col_a, col_b in combinations(columns, 2):
            sub = df.select([col_a, col_b]).drop_nulls()
            a = sub[col_a].to_numpy().astype(float)
            b = sub[col_b].to_numpy().astype(float)

            if method is CorrelateMethod.PEARSON:
                corr, pval = sp.pearsonr(a, b)
            elif method is CorrelateMethod.SPEARMAN:
                corr, pval = sp.spearmanr(a, b)
            else:
                corr, pval = sp.kendalltau(a, b)

            rows.append(
                {
                    "column_a": col_a,
                    "column_b": col_b,
                    "correlation": float(cast(float, corr)),
                    "p_value": float(cast(float, pval)),
                }
            )

        data = pl.DataFrame(rows)
        metadata: dict[str, Any] = {
            "method": method.value if method is not None else None,
            "columns": columns,
            "across": across,
            "n_observations": n_obs,
        }
        return self._r._replace(data=data, has_geometry=False, metadata=metadata)

    def test(
        self,
        column: str,
        groups: str | None = None,
        periods: list[tuple[int, int]] | None = None,
        against: float | None = None,
        method: TestMethod | None = None,
        significance_level: float = 0.05,
    ) -> "Result":
        """Hypothesis test in three modes: group-based, period-based, single-sample.

        Args:
            column: Numeric column to test.
            groups: Categorical column to split groups on.
            periods: List of ``(from_year, to_year)`` period tuples.
            against: Reference value for a one-sample test.
            method: A :class:`~core_lens.base.namespaces.stats.TestMethod` enum value.
                Auto-selected via Shapiro-Wilk if ``None``.
            significance_level: Alpha level for ``significant`` flag (default 0.05).

        Returns:
            Result whose data has per-group descriptive stats
            ``group | n | mean | std | median`` and
            ``metadata`` with ``statistic``, ``p_value``, ``significant``.

        Raises:
            ValueError: If none of ``groups``, ``periods``, ``against`` supplied,
                or if ``method`` is not recognised.
        """
        import scipy.stats as sp

        df = self._r.data
        all_vals = df[column].drop_nulls().to_numpy().astype(float)

        # auto-select
        if method is None:
            if len(all_vals) >= 3:
                _, p_sw = sp.shapiro(all_vals[:5000])
                method = TestMethod.T_TEST if p_sw > 0.05 else TestMethod.MANN_WHITNEY
            else:
                method = TestMethod.MANN_WHITNEY

        if not isinstance(method, TestMethod):
            raise ValueError(
                f"StatsNamespace.test: method must be a TestMethod. Valid options: {[e.name for e in TestMethod]}."
            )

        if against is not None:
            if method is TestMethod.T_TEST:
                stat, pval = sp.ttest_1samp(all_vals, against)
            elif method is TestMethod.WILCOXON:
                stat, pval = sp.wilcoxon(all_vals - against)
            else:
                raise ValueError(
                    f"StatsNamespace.test: method={method!r} is not valid for single-sample test against a reference value. "
                    "Valid options: 't-test', 'wilcoxon'."
                )
            data = pl.DataFrame(
                {
                    "group": ["all"],
                    "n": [len(all_vals)],
                    "mean": [float(np.mean(all_vals))],
                    "std": [float(np.std(all_vals, ddof=1))],
                    "median": [float(np.median(all_vals))],
                }
            )
            metadata: dict[str, Any] = {
                "method": method.value if method is not None else None,
                "statistic": float(cast(float, stat)),
                "p_value": float(cast(float, pval)),
                "significant": bool(cast(float, pval) < significance_level),
                "significance_level": significance_level,
                "against": against,
            }

        elif groups is not None:
            group_names = df[groups].unique().sort().to_list()
            arrays = [
                df.filter(pl.col(groups) == g)[column]
                .drop_nulls()
                .to_numpy()
                .astype(float)
                for g in group_names
            ]
            stat, pval = _run_test(method, arrays, sp)
            rows = [
                {
                    "group": str(g),
                    "n": len(a),
                    "mean": float(np.mean(a)) if len(a) else float("nan"),
                    "std": float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"),
                    "median": float(np.median(a)) if len(a) else float("nan"),
                }
                for g, a in zip(group_names, arrays)
            ]
            data = pl.DataFrame(rows)
            metadata = {
                "method": method.value if method is not None else None,
                "statistic": float(stat),
                "p_value": float(pval),
                "significant": bool(pval < significance_level),
                "significance_level": significance_level,
                "groups": groups,
            }

        elif periods is not None:
            year_col = self._year_col()
            if year_col is None:
                raise ValueError(
                    "StatsNamespace.test: periods mode requires a year column. Ensure data is at annual resolution."
                )
            arrays = [
                df.filter((pl.col(year_col) >= p[0]) & (pl.col(year_col) <= p[1]))[
                    column
                ]
                .drop_nulls()
                .to_numpy()
                .astype(float)
                for p in periods
            ]
            stat, pval = _run_test(method, arrays, sp)
            rows = [
                {
                    "group": f"{p[0]}-{p[1]}",
                    "n": len(a),
                    "mean": float(np.mean(a)) if len(a) else float("nan"),
                    "std": float(np.std(a, ddof=1)) if len(a) > 1 else float("nan"),
                    "median": float(np.median(a)) if len(a) else float("nan"),
                }
                for p, a in zip(periods, arrays)
            ]
            data = pl.DataFrame(rows)
            metadata = {
                "method": method.value if method is not None else None,
                "statistic": float(stat),
                "p_value": float(pval),
                "significant": bool(pval < significance_level),
                "significance_level": significance_level,
                "periods": [list(p) for p in periods],
            }

        else:
            raise ValueError(
                "StatsNamespace.test: Missing test mode. Provide exactly one of: 'groups', 'periods', or 'against'."
            )

        return self._r._replace(data=data, has_geometry=False, metadata=metadata)

    def change(
        self,
        column: str,
        from_period: int,
        to_period: int,
        method: ChangeMethod = ChangeMethod.ABSOLUTE,
    ) -> "Result":
        """Change between two time periods per entity.

        Args:
            column: Value column to compute change for.
            from_period: Start year/period integer.
            to_period: End year/period integer.
            method: A :class:`~core_lens.base.namespaces.stats.ChangeMethod` enum value.

        Returns:
            For *absolute* / *percentage*: data has
            ``key_col | value_from | value_to | change | pct_change``.
            For *trend*: ``key_col | slope | r_squared | direction``.

        Raises:
            ValueError: If ``method`` is not recognised or year column absent.
        """
        if not isinstance(method, ChangeMethod):
            raise ValueError(
                f"StatsNamespace.change: method must be a ChangeMethod. Valid options: {[e.name for e in ChangeMethod]}."
            )

        import scipy.stats as sp

        df = self._r.data
        key = self._r.key_cols[0]
        year_col = self._year_col()
        if year_col is None:
            raise ValueError(
                "StatsNamespace.change: Requires a year/time column. Ensure data is at annual resolution."
            )

        if method in (ChangeMethod.ABSOLUTE, ChangeMethod.PERCENTAGE):
            from_df = (
                df.filter(pl.col(year_col) == from_period)
                .select([key, column])
                .rename({column: "value_from"})
            )
            to_df = (
                df.filter(pl.col(year_col) == to_period)
                .select([key, column])
                .rename({column: "value_to"})
            )
            joined = from_df.join(to_df, on=key, how="inner").with_columns(
                [
                    (pl.col("value_to") - pl.col("value_from")).alias("change"),
                    (
                        (pl.col("value_to") - pl.col("value_from"))
                        / pl.col("value_from")
                        * 100
                    ).alias("pct_change"),
                ]
            )
            data = joined

        else:  # trend
            period_df = df.filter(
                (pl.col(year_col) >= from_period) & (pl.col(year_col) <= to_period)
            )
            rows: list[dict[str, Any]] = []
            for eid in period_df[key].unique().sort().to_list():
                sub = (
                    period_df.filter(pl.col(key) == eid)
                    .sort(year_col)
                    .select([year_col, column])
                    .drop_nulls()
                )
                x = sub[year_col].to_numpy().astype(float)
                y = sub[column].to_numpy().astype(float)
                if len(y) < 2:
                    continue
                slope, _, r_val, _, _ = sp.linregress(x, y)
                rows.append(
                    {
                        key: eid,
                        "slope": float(cast(float, slope)),
                        "r_squared": float(cast(float, r_val) ** 2),
                        "direction": (
                            "increasing"
                            if cast(float, slope) > 0
                            else "decreasing"
                            if cast(float, slope) < 0
                            else "stable"
                        ),
                    }
                )
            data = pl.DataFrame(rows)

        metadata: dict[str, Any] = {
            "method": method.value if method is not None else None,
            "column": column,
            "from_period": from_period,
            "to_period": to_period,
        }
        return self._r._replace(data=data, has_geometry=False, metadata=metadata)

    def anomaly(
        self,
        column: str,
        mode: str,
        method: AnomalyCrossMethod | AnomalyTsMethod,
        baseline: tuple[int, int] | None = None,
        threshold: float = 2.0,
    ) -> "Result":
        """Anomaly detection in cross-sectional or timeseries mode.

        Args:
            column: Value column to analyse.
            mode: ``"cross_sectional"`` or ``"timeseries"``.
            method: An :class:`~core_lens.base.namespaces.stats.AnomalyCrossMethod` or :class:`~core_lens.base.namespaces.stats.AnomalyTsMethod` enum value.
            baseline: ``(from_year, to_year)`` inclusive.  Required for
                timeseries; optional for cross-sectional.
            threshold: Sigma / score threshold for anomaly flag (default 2.0).

        Returns:
            Result whose data has ``key_col | anomaly_score | is_anomaly``
            (cross-sectional) or ``key_col | year | anomaly_score | is_anomaly``
            (timeseries, baseline period excluded).

        Raises:
            ValueError: If ``mode``, ``method``, or observation count invalid.
        """
        df = self._r.data
        key = self._r.key_cols[0]

        if mode == "cross_sectional":
            if not isinstance(method, AnomalyCrossMethod):
                raise ValueError(
                    f"StatsNamespace.anomaly: mode 'cross_sectional' does not support method {method!r}. "
                    f"Valid options: {[e.name for e in AnomalyCrossMethod]}."
                )

            # baseline subset for computing reference stats
            if baseline is not None and "year" in df.columns:
                ref_vals = (
                    df.filter(
                        (pl.col("year") >= baseline[0])
                        & (pl.col("year") <= baseline[1])
                    )[column]
                    .drop_nulls()
                    .to_numpy()
                    .astype(float)
                )
            else:
                ref_vals = df[column].drop_nulls().to_numpy().astype(float)

            min_obs = _MIN_OBS[method.value]
            if len(ref_vals) < min_obs:
                raise ValueError(
                    f"StatsNamespace.anomaly: method {method!r} requires at least {min_obs} baseline observations, "
                    f"but only got {len(ref_vals)}."
                )

            all_vals = df[column].to_numpy().astype(float)

            if method is AnomalyCrossMethod.ZSCORE:
                mean = float(np.nanmean(ref_vals))
                std = float(np.nanstd(ref_vals, ddof=1))
                scores = (all_vals - mean) / (std or 1.0)
                flags = np.abs(scores) > threshold
                meta: dict[str, Any] = {
                    "mode": "cross_sectional",
                    "method": "zscore",
                    "baseline": baseline,
                    "baseline_mean": mean,
                    "baseline_std": std,
                }

            elif method is AnomalyCrossMethod.IQR:
                q1 = float(np.nanpercentile(ref_vals, 25))
                q3 = float(np.nanpercentile(ref_vals, 75))
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                med = float(np.nanmedian(ref_vals))
                scores = (all_vals - med) / (iqr or 1.0)
                flags = (all_vals < lo) | (all_vals > hi)
                meta = {
                    "mode": "cross_sectional",
                    "method": "iqr",
                    "baseline": baseline,
                    "baseline_mean": float(np.nanmean(ref_vals)),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                }

            elif method is AnomalyCrossMethod.PERCENTILE:
                lo = float(np.nanpercentile(ref_vals, 5))
                hi = float(np.nanpercentile(ref_vals, 95))
                med = float(np.nanmedian(ref_vals))
                std = float(np.nanstd(ref_vals)) or 1.0
                scores = (all_vals - med) / std
                flags = (all_vals < lo) | (all_vals > hi)
                meta = {
                    "mode": "cross_sectional",
                    "method": "percentile",
                    "baseline": baseline,
                    "baseline_mean": float(np.nanmean(ref_vals)),
                    "lower_pct": lo,
                    "upper_pct": hi,
                }

            else:  # threshold
                mean = float(np.nanmean(ref_vals))
                scores = all_vals - mean
                flags = np.abs(scores) > threshold
                meta = {
                    "mode": "cross_sectional",
                    "method": "threshold",
                    "baseline": baseline,
                    "baseline_mean": mean,
                    "threshold": threshold,
                }

            data = df.select(pl.col(key)).with_columns(
                [
                    pl.Series("anomaly_score", scores.tolist(), dtype=pl.Float64),
                    pl.Series("is_anomaly", flags.tolist(), dtype=pl.Boolean),
                ]
            )

        elif mode == "timeseries":
            if not isinstance(method, AnomalyTsMethod):
                raise ValueError(
                    f"StatsNamespace.anomaly: mode 'timeseries' does not support method {method!r}. "
                    f"Valid options: {[e.name for e in AnomalyTsMethod]}."
                )
            if baseline is None:
                raise ValueError(
                    "StatsNamespace.anomaly: mode 'timeseries' requires a defined baseline=(from_year, to_year)."
                )

            year_col = self._year_col()
            if year_col is None:
                raise ValueError(
                    "StatsNamespace.anomaly: mode 'timeseries' requires a year/time column. Ensure data is at annual or fortnightly resolution."
                )

            min_obs = _MIN_OBS[method.value]
            rows: list[dict[str, Any]] = []

            for eid in df[key].unique().sort().to_list():
                sub = df.filter(pl.col(key) == eid).sort(year_col)
                base_vals = (
                    sub.filter(
                        (pl.col(year_col) >= baseline[0])
                        & (pl.col(year_col) <= baseline[1])
                    )[column]
                    .drop_nulls()
                    .to_numpy()
                    .astype(float)
                )
                if len(base_vals) < min_obs:
                    continue

                eval_sub = sub.filter(pl.col(year_col) > baseline[1])
                eval_vals = eval_sub[column].drop_nulls().to_numpy().astype(float)
                eval_years = eval_sub[year_col].to_list()

                ts_scores: list[float] = []
                ts_flags: list[bool] = []

                if method is AnomalyTsMethod.MAD:
                    med = float(np.median(base_vals))
                    mad = float(np.median(np.abs(base_vals - med)))
                    scale = (mad * 1.4826) or 1.0
                    ts_scores = [(v - med) / scale for v in eval_vals]
                    ts_flags = [abs(s) > threshold for s in ts_scores]

                elif method is AnomalyTsMethod.CUSUM:
                    mean = float(np.mean(base_vals))
                    std = float(np.std(base_vals, ddof=1)) or 1.0
                    k, h = 0.5 * std, threshold * std
                    cp, cn = 0.0, 0.0
                    for v in eval_vals:
                        cp = max(0.0, cp + v - mean - k)
                        cn = max(0.0, cn - v + mean - k)
                        s = max(cp, cn)
                        ts_scores.append(s)
                        ts_flags.append(s > h)

                else:  # stl
                    try:
                        from statsmodels.tsa.seasonal import STL  # type: ignore[import-untyped]

                        full_vals = sub[column].to_numpy().astype(float)
                        if len(full_vals) < min_obs:
                            continue
                        period = (
                            24 if len(full_vals) >= 24 else max(2, len(full_vals) // 2)
                        )
                        res = STL(full_vals, period=period).fit()
                        resid = res.resid
                        base_len = len(base_vals)
                        base_resid = resid[:base_len]
                        eval_resid = resid[base_len : base_len + len(eval_years)]
                        std = float(np.std(base_resid, ddof=1)) or 1.0
                        ts_scores = [float(r / std) for r in eval_resid]
                        ts_flags = [abs(s) > threshold for s in ts_scores]
                    except Exception:
                        ts_scores = [float("nan")] * len(eval_years)
                        ts_flags = [False] * len(eval_years)

                for yr, sc, fl in zip(eval_years, ts_scores, ts_flags):
                    rows.append(
                        {
                            key: eid,
                            year_col: yr,
                            "anomaly_score": float(sc),
                            "is_anomaly": bool(fl),
                        }
                    )

            if rows:
                data = pl.DataFrame(rows)
            else:
                data = pl.DataFrame(
                    {
                        key: pl.Series([], dtype=pl.String),
                        year_col: pl.Series([], dtype=pl.Int32),
                        "anomaly_score": pl.Series([], dtype=pl.Float64),
                        "is_anomaly": pl.Series([], dtype=pl.Boolean),
                    }
                )
            global_base_vals = (
                df.filter(
                    (pl.col(year_col) >= baseline[0])
                    & (pl.col(year_col) <= baseline[1])
                )[column]
                .drop_nulls()
                .to_numpy()
                .astype(float)
            )

            meta = {
                "mode": "timeseries",
                "method": method.value if method is not None else None,
                "baseline": baseline,
                "baseline_mean": float(np.mean(global_base_vals))
                if len(global_base_vals) > 0
                else float("nan"),
                "baseline_fitted": True,
            }

        else:
            raise ValueError(
                f"StatsNamespace.anomaly: Unknown mode {mode!r}. Valid options: 'cross_sectional', 'timeseries'."
            )

        return self._r._replace(data=data, has_geometry=False, metadata=meta)

    def similarity(
        self,
        target: str,
        columns: dict[str, Any],
        method: SimilarityMethod = SimilarityMethod.EUCLIDEAN,
        top_n: int = 10,
    ) -> "Result":
        """Find entities most similar to ``target`` across ``columns``.

        Args:
            target: Key value of the reference entity.
            columns: Mapping ``{column_name: None | (resolution, filter_dict)}``.

                * ``None`` — use the column directly from ``result.data``.
                * ``(resolution, filter_dict)`` — fetch from the entity's file at
                  the given resolution (``"static"``, ``"annual"``,
                  ``"fortnightly"``).  Supported ``filter_dict`` keys:

                  - ``"year": int | (int, int)`` — year equality or range.
                  - ``"season": str`` — season label (``"kharif"`` etc.).
                  - ``"agg": str`` — time-aggregation before joining
                    (``"mean"`` default, ``"sum"``, ``"min"``, ``"max"``).

            method: Distance metric. A :class:`~core_lens.base.namespaces.stats.SimilarityMethod` enum value.
            top_n: Number of most-similar entities to return (default 10).

        Returns:
            Result with ``key_col | similarity_score | rank``.

        Raises:
            ValueError: If ``method`` is invalid, no columns can be resolved,
                or ``target`` is not found.
        """
        if not isinstance(method, SimilarityMethod):
            raise ValueError(
                f"StatsNamespace.similarity: method must be a SimilarityMethod. "
                f"Valid options: {[e.name for e in SimilarityMethod]}."
            )

        df = self._r.data
        key = self._r.key_cols[0]

        # --- Build feature DataFrame ------------------------------------------
        # Universe of entity keys from result.data.
        feat = df.select([key])

        for col_name, spec in columns.items():
            if spec is None:
                # Pull from result.data directly.
                if col_name in df.columns:
                    feat = feat.join(df.select([key, col_name]), on=key, how="left")
                # else silently skip — not in data and no spec provided
                continue

            # spec = (resolution_str, filter_dict)
            if not (isinstance(spec, tuple) and len(spec) == 2):
                raise ValueError(
                    f"StatsNamespace.similarity: Column spec for {col_name!r} must be "
                    "None or a (resolution, filter_dict) tuple."
                )

            resolution_str, filter_dict = spec
            filter_dict = dict(filter_dict) if filter_dict else {}

            entity = self._r.entity
            path: str | None
            if resolution_str == "static":
                path = entity.static_path
            elif resolution_str == "annual":
                path = entity.annual_path
            elif resolution_str == "fortnightly":
                path = entity.fortnightly_path
            else:
                raise ValueError(
                    f"StatsNamespace.similarity: Unknown resolution {resolution_str!r} "
                    "in column spec. Valid: 'static', 'annual', 'fortnightly'."
                )

            if path is None:
                continue  # entity doesn't have this resolution

            try:
                abs_path = entity._resolve(path)
            except FileNotFoundError:
                continue

            schema = pl.read_parquet_schema(abs_path)
            if col_name not in schema:
                continue  # column doesn't exist in this file

            col_lf = pl.scan_parquet(abs_path)

            # Apply year filter.
            if "year" in filter_dict:
                yr = filter_dict["year"]
                profile = entity.schema_profile
                yr_col: str | None = None
                if resolution_str == "annual":
                    yr_col = profile.annual_time_col
                elif resolution_str == "fortnightly":
                    yr_col = profile.fortnightly_time_col
                # Fallback: look for a literal "year" integer column.
                if yr_col is None and "year" in schema:
                    yr_col = "year"
                if yr_col is not None:
                    if isinstance(yr, int):
                        col_lf = col_lf.filter(pl.col(yr_col) == yr)
                    else:
                        col_lf = col_lf.filter(pl.col(yr_col).is_between(yr[0], yr[1]))

            # Apply season filter.
            if "season" in filter_dict and "season" in schema:
                col_lf = col_lf.filter(pl.col("season") == filter_dict["season"])

            # Aggregate over time per entity.
            agg_fn = filter_dict.get("agg", "mean")
            if agg_fn == "sum":
                agg_expr = pl.col(col_name).sum().alias(col_name)
            elif agg_fn == "min":
                agg_expr = pl.col(col_name).min().alias(col_name)
            elif agg_fn == "max":
                agg_expr = pl.col(col_name).max().alias(col_name)
            else:  # mean (default)
                agg_expr = pl.col(col_name).mean().alias(col_name)

            if resolution_str == "static":
                # Static: no grouping needed — one row per entity.
                fetched = col_lf.select([key, col_name]).collect()
            else:
                fetched = col_lf.group_by(key).agg(agg_expr).collect()

            feat = feat.join(fetched.select([key, col_name]), on=key, how="left")

        # ------------------------------------------------------------------
        feature_cols = [c for c in feat.columns if c != key]

        if not feature_cols:
            raise ValueError(
                f"StatsNamespace.similarity: None of the specified columns "
                f"{list(columns)} could be resolved. "
                f"Available columns in result: {df.columns}."
            )

        feat = feat.drop_nulls(subset=feature_cols)
        ids = feat[key].to_list()

        if target not in ids:
            raise ValueError(
                f"StatsNamespace.similarity: Target entity {target!r} not found in "
                f"{key} column after resolving columns and dropping null rows."
            )

        mat = feat.select(feature_cols).to_numpy().astype(float)
        tidx = ids.index(target)

        # z-score normalise before computing distances.
        means = np.nanmean(mat, axis=0)
        stds = np.nanstd(mat, axis=0, ddof=1)
        stds[stds == 0] = 1.0
        norm = (mat - means) / stds
        tvec = norm[tidx]

        if method is SimilarityMethod.EUCLIDEAN:
            dists = np.sqrt(np.sum((norm - tvec) ** 2, axis=1))
        elif method is SimilarityMethod.MANHATTAN:
            dists = np.sum(np.abs(norm - tvec), axis=1)
        elif method is SimilarityMethod.COSINE:
            row_norms = np.linalg.norm(norm, axis=1)
            row_norms[row_norms == 0] = 1.0
            tnorm = np.linalg.norm(tvec) or 1.0
            dists = 1.0 - (norm @ tvec) / (row_norms * tnorm)
        else:  # mahalanobis
            try:
                from scipy.spatial.distance import mahalanobis  # type: ignore[import-untyped]

                cov = np.cov(norm.T)
                vi = np.linalg.pinv(cov)
                dists = np.array([mahalanobis(row, tvec, vi) for row in norm])
            except Exception:
                dists = np.sqrt(np.sum((norm - tvec) ** 2, axis=1))

        dists[tidx] = float("inf")  # exclude target itself
        top_idx = np.argsort(dists)[:top_n]

        data = pl.DataFrame(
            {
                key: [ids[i] for i in top_idx],
                "similarity_score": [float(1.0 / (1.0 + dists[i])) for i in top_idx],
                "rank": list(range(1, len(top_idx) + 1)),
            }
        )
        metadata: dict[str, Any] = {
            "method": method.value if method is not None else None,
            "target": target,
            "columns": {k: v for k, v in columns.items()},
            "top_n": top_n,
        }
        return self._r._replace(data=data, has_geometry=False, metadata=metadata)


def _run_test(
    method: TestMethod,
    arrays: list[np.ndarray],
    sp: Any,
) -> tuple[float, float]:
    """Dispatch the appropriate scipy test for 2+ groups."""
    if len(arrays) > 2:
        stat, pval = sp.kruskal(*arrays)
        return float(stat), float(pval)

    a, b = arrays[0], arrays[1]
    if method is TestMethod.T_TEST:
        stat, pval = sp.ttest_ind(a, b)
    elif method is TestMethod.MANN_WHITNEY:
        stat, pval = sp.mannwhitneyu(a, b, alternative="two-sided")
    elif method is TestMethod.WILCOXON:
        n = min(len(a), len(b))
        stat, pval = sp.wilcoxon(a[:n], b[:n])
    elif method is TestMethod.KS:
        stat, pval = sp.ks_2samp(a, b)
    else:  # chi-square
        stat, pval = sp.chisquare([len(a), len(b)])
    return float(stat), float(pval)
