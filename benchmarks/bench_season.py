"""bench_season.py — Benchmark for core_lens.utils.season
=======================================================
Targets:
  - SeasonConfig.__post_init__      validation cost (datetime.strptime × 6)
  - SeasonConfig.season_for()       per-date season classification
  - resolve_time_filter()           dict → Polars Expr build (date-range + season modes)
  - _date_range_expr()              expression build for date ranges
  - _season_expr()                  expression build for seasons (year-crossing vs not)
  - add_temporal_columns()          sub_annual vectorised temporal derivation

Scalene focuses on:
  - Python datetime.strptime calls in __post_init__
  - string comparison loop in season_for()
  - Polars expression assembly overhead in resolve_time_filter
  - add_temporal_columns when/then vectorised season assignment

Usage:
    uv run scalene run --html --outfile benchmarks/profiles/bench_season.html benchmarks/bench_season.py
    uv run python benchmarks/bench_season.py
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import polars as pl
from typing import cast

from core_lens.aoi import SeasonConfig
from core_lens.utils.season import add_temporal_columns, resolve_time_filter

# ── Config ────────────────────────────────────────────────────────────────────
# Representative date range for temporal column derivation benchmark.
START_DATE = date(2010, 1, 1)
END_DATE = date(2023, 12, 31)

# Number of sub_annual periods between START_DATE and END_DATE (roughly).
N_SUB_ANNUAL = (END_DATE - START_DATE).days // 14

# Representative synthetic sub_annual dataframe for add_temporal_columns.
N_MWS = 500  # number of entity instances (e.g. MWS)
DATES = [START_DATE + timedelta(days=14 * i) for i in range(N_SUB_ANNUAL)]
# Long format: N_MWS × N_SUB_ANNUAL rows
_all_dates = DATES * N_MWS
SUB_ANNUAL_DF = pl.DataFrame(
    {
        "mws_id": [str(mid) for mid in range(N_MWS) for _ in DATES],
        "date": _all_dates,
        "ndvi": [0.5] * (N_MWS * N_SUB_ANNUAL),
    }
)
print(f"Synthetic sub_annual frame: {SUB_ANNUAL_DF.shape}")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ── 1. SeasonConfig construction — default ────────────────────────────────────
_section("1. SeasonConfig()  [default — 6× strptime validation]")
REPS = 10_000
t0 = time.perf_counter()
for _ in range(REPS):
    SeasonConfig()
t1 = time.perf_counter()
print(
    f"SeasonConfig() ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 2. SeasonConfig construction — custom ─────────────────────────────────────
_section("2. SeasonConfig(custom)  [custom ranges — same 6× strptime]")
t0 = time.perf_counter()
for _ in range(REPS):
    SeasonConfig(
        kharif=("06-15", "10-15"),
        rabi=("10-16", "02-28"),
        zaid=("03-01", "06-14"),
    )
t1 = time.perf_counter()
print(
    f"SeasonConfig(custom) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 3. SeasonConfig.season_for() — hot path ──────────────────────────────────
_section("3. SeasonConfig.season_for(date)  [string comparison loop]")
cfg = SeasonConfig()
sample_dates = [START_DATE + timedelta(days=i * 30) for i in range(12)]

REPS_SF = 1_000_000
t0 = time.perf_counter()
for i in range(REPS_SF):
    cfg.season_for(sample_dates[i % 12])
t1 = time.perf_counter()
print(
    f"season_for ×{REPS_SF}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_SF * 1e6:.3f} µs/call)"
)


# ── 4. resolve_time_filter — date range mode ──────────────────────────────────
_section("4. resolve_time_filter() — date range mode")
TimeFilter = dict[str, str | int | tuple[int, int] | None]
tf_daterange: TimeFilter = {
    "start": "2010-01-01",
    "end": "2023-12-31",
    "_is_year_col": False,
}
REPS_TF = 100_000
t0 = time.perf_counter()
for _ in range(REPS_TF):
    resolve_time_filter(tf_daterange, "date", cfg)
t1 = time.perf_counter()
print(
    f"resolve (date range) ×{REPS_TF}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_TF * 1e6:.2f} µs/call)"
)


# ── 5. resolve_time_filter — season mode (non-year-crossing) ─────────────────
_section("5. resolve_time_filter() — season=kharif (non-year-crossing)")
tf_kharif: TimeFilter = {"season": "kharif", "year": 2022}
t0 = time.perf_counter()
for _ in range(REPS_TF):
    resolve_time_filter(tf_kharif, "date", cfg)
t1 = time.perf_counter()
print(
    f"resolve (kharif) ×{REPS_TF}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_TF * 1e6:.2f} µs/call)"
)


# ── 6. resolve_time_filter — season mode (year-crossing rabi) ────────────────
_section("6. resolve_time_filter() — season=rabi (year-crossing OR expr)")
tf_rabi: TimeFilter = {"season": "rabi", "year": 2022}
t0 = time.perf_counter()
for _ in range(REPS_TF):
    resolve_time_filter(tf_rabi, "date", cfg)
t1 = time.perf_counter()
print(
    f"resolve (rabi) ×{REPS_TF}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_TF * 1e6:.2f} µs/call)"
)


# ── 7. resolve_time_filter — year range tuple ─────────────────────────────────
_section("7. resolve_time_filter() — season=kharif, year=(2018, 2023)")
tf_range: TimeFilter = {"season": "kharif", "year": (2018, 2023)}
t0 = time.perf_counter()
for _ in range(REPS_TF):
    resolve_time_filter(tf_range, "date", cfg)
t1 = time.perf_counter()
print(
    f"resolve (year range) ×{REPS_TF}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_TF * 1e6:.2f} µs/call)"
)


# ── 8. add_temporal_columns — large sub_annual DataFrame ─────────────────────
_section(f"8. add_temporal_columns()  [{SUB_ANNUAL_DF.shape[0]:,} rows]")
t0 = time.perf_counter()
result_df = add_temporal_columns(SUB_ANNUAL_DF, "date", cfg)
t1 = time.perf_counter()
print(f"add_temporal_columns: {(t1 - t0) * 1000:.2f} ms")
print(f"Output shape        : {cast(pl.DataFrame, result_df).shape}")
print(
    f"New columns added   : {[c for c in cast(pl.DataFrame, result_df).columns if c not in SUB_ANNUAL_DF.columns]}"
)

# Second call — all columns already present, should be near-zero.
t0 = time.perf_counter()
add_temporal_columns(result_df, "date", cfg)
t1 = time.perf_counter()
print(f"2nd call (noop)     : {(t1 - t0) * 1000:.2f} ms  (columns already present)")


# ── 9. Polars expression execution — resolve + filter on frame ────────────────
_section("9. Polars expression execution: filter(resolve_time_filter())")
expr = resolve_time_filter(tf_kharif, "date", cfg)
t0 = time.perf_counter()
filtered = result_df.filter(expr)
t1 = time.perf_counter()
print(f"filter(kharif 2022) : {(t1 - t0) * 1000:.2f} ms")
print(
    f"Rows matched        : {len(cast(pl.DataFrame, filtered)):,} / {len(cast(pl.DataFrame, result_df)):,}"
)

print("\n✓ bench_season.py complete")
