"""bench_result.py — Benchmark for core_lens.base.result.Result
=============================================================
Targets:
  - Result.df()                 zero-copy accessor
  - Result.gdf()                WKB decode + GeoPandas construction
  - Result.lazy()               LazyFrame wrapping
  - Result.materialise()        eager evaluate + re-wrap as in-memory lazy
  - Result.with_geometry()      static-join to attach geometry to non-static result
  - Result.derive()             with_columns expression append
  - Result.aggregate()          group_by + agg (no temporal grouping)
  - Result.aggregate(by=...)    temporal grouping (annual / sub_annual)
  - Result.aggregate errors     STATIC resolution guard, invalid by, sub_annual-only by
  - Result.gdf errors           TypeError when has_geometry=False
  - Result.stats / Result.plot  namespace accessor overhead

Scalene focuses on:
  - shapely.wkb.loads() in gdf() — Python loop over WKB bytes
  - GeoPandas construction overhead
  - collect_lf() in with_geometry()
  - group_by / agg in aggregate()

Usage:
    uv run scalene run --html --outfile benchmarks/profiles/bench_result.html benchmarks/bench_result.py
    uv run python benchmarks/bench_result.py
"""

from __future__ import annotations

import time

import polars as pl

from core_lens import AoI
from core_lens.entities.mws import MWSEntity

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"
INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 38.0)
SMALL_BBOX: tuple[float, float, float, float] = (86.0, 21.0, 87.5, 22.0)

AoI.register(MWSEntity)
aoi = AoI(DATA_ROOT, bbox=INDIA_BBOX)
aoi_small = AoI(DATA_ROOT, bbox=SMALL_BBOX)

# Materialise once — all benchmarks below operate on this result.
result = aoi.mws.static.materialise()
result_small = aoi_small.mws.static.materialise()
print(f"Dataset: {result.df().shape[0]} MWS, {result.df().shape[1]} columns")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ── 1. Result.df() ────────────────────────────────────────────────────────────
_section("1. Result.df()  [zero-copy — should be ~0 ms]")
REPS = 100_000
t0 = time.perf_counter()
for _ in range(REPS):
    result.df()
t1 = time.perf_counter()
print(
    f"df() ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 2. Result.lazy() ─────────────────────────────────────────────────────────
_section("2. Result.lazy()  [LazyFrame wrapping — should be ~0 ms]")
t0 = time.perf_counter()
for _ in range(REPS):
    result.lazy()
t1 = time.perf_counter()
print(
    f"lazy() ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 3. Result.gdf() — WKB decode + GeoPandas ─────────────────────────────────
_section("3. Result.gdf()  [WKB decode + GeoPandas — Python loop (small)]")
t0 = time.perf_counter()
gdf = result_small.gdf()
t1 = time.perf_counter()
print(f"gdf()            : {(t1 - t0) * 1000:.2f} ms")
print(f"GeoDataFrame     : {gdf.shape}, CRS={gdf.crs}")

# Second call (data already in memory, no re-read from Parquet).
t0 = time.perf_counter()
result_small.gdf()
t1 = time.perf_counter()
print(
    f"gdf() (2nd call) : {(t1 - t0) * 1000:.2f} ms  "
    "(WKB decode happens each time — no internal cache)"
)


# ── 4. Result.derive() — single expression ────────────────────────────────────
_section("4. Result.derive('area_km2', expr)  [with_columns wrapper]")
t0 = time.perf_counter()
result_derived = result.derive(
    "area_km2",
    (pl.col("area_in_ha") / 100).round(2),  # ha → km²
)
t1 = time.perf_counter()
print(f"derive(1 col)    : {(t1 - t0) * 1000:.2f} ms")
print(
    f"New columns      : {[c for c in result_derived.columns if c not in result.columns]}"
)


# ── 5. Result.derive() — chained (2 exprs) ────────────────────────────────────
_section("5. Result.derive chain (2 columns)")
t0 = time.perf_counter()
result_rich = result.derive("area_km2", (pl.col("area_in_ha") / 100).round(2)).derive(
    "plain_area_km2", (pl.col("te_plain_area") / 100).round(3)
)
t1 = time.perf_counter()
print(f"derive ×2        : {(t1 - t0) * 1000:.2f} ms")


# ── 6. Result.with_geometry() — no-op on static ───────────────────────────────
_section("6. Result.with_geometry()  [no-op on static — should be ~0 ms]")
REPS = 10_000
t0 = time.perf_counter()
for _ in range(REPS):
    result.with_geometry()
t1 = time.perf_counter()
print(
    f"with_geometry (noop) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── Note: meaningful with_geometry() test requires annual/sub_annual result ──
# Uncomment with an MWS entity that has annual_path set:

result_annual = aoi.mws.annual  # no geometry
t0 = time.perf_counter()
result_with_geom = result_annual.with_geometry()
print(f"with_geometry (annual): {(time.perf_counter() - t0) * 1000:.2f} ms")


# ── 7. Result.aggregate() — no grouping (collapse all) ───────────────────────
_section("7. Result.aggregate(pl.mean('area'), by=None)  [collapse all]")
# MWS has annual data — use a real annual result instead of monkey-patching.
result_annual = aoi_small.mws.between("2018-01-01", "2023-12-31").annual
t0 = time.perf_counter()
agg_result = result_annual.aggregate(
    pl.mean("ndvi")
    if "ndvi" in result_annual.columns
    else pl.first(result_annual.columns[-1])
)
t1 = time.perf_counter()
print(f"aggregate(no by) : {(t1 - t0) * 1000:.2f} ms")
print(f"Shape            : {agg_result.df().shape}")


# ── 8. Result.aggregate(by='year') ──────────────────────────────────────────────
_section("8. Result.aggregate(by='year')  [group by mws_id + year]")
t0 = time.perf_counter()
agg_year = result_annual.aggregate(
    pl.mean("ndvi")
    if "ndvi" in result_annual.columns
    else pl.first(result_annual.columns[-1]),
    by="year",
)
t1 = time.perf_counter()
print(f"aggregate(year)  : {(t1 - t0) * 1000:.2f} ms")
print(f"Shape            : {agg_year.df().shape}")


# ── 9. Result.materialise() — eager evaluate (≈ df + wrap) ────────────────────
_section("9. Result.materialise()  [collect + re-wrap in-memory lazy]")
# Use result (all-India static) — data already in memory so measures wrap cost.
REPS_M = 100
t0 = time.perf_counter()
for _ in range(REPS_M):
    result.materialise()
t1 = time.perf_counter()
print(
    f"materialise (— mem) ×{REPS_M}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_M * 1000:.2f} ms/call)"
)


# ── 10. Result.aggregate(by='season') / by='month' / by='year_month' ─────────
_section("10. Result.aggregate() sub_annual groupings")
from core_lens.base.view import Season  # noqa: E402

result_fn = aoi_small.mws.between(season=Season.KHARIF, year=(2018, 2022)).sub_annual

data_col = result_fn.columns[-1]  # last column (data column)

for by_key in ("month", "season", "year_month", "season_year", "year"):
    t0 = time.perf_counter()
    agg_r = result_fn.aggregate(pl.first(data_col), by=by_key)
    t1 = time.perf_counter()
    print(
        f"aggregate(by={by_key!r:12}) : {(t1 - t0) * 1000:.2f} ms  shape={agg_r.df().shape}"
    )


# ── 11. Result.aggregate() error paths ──────────────────────────────────
_section("11. Result.aggregate() error paths  [guard branches — no I/O]")
errors_caught = 0

# 11a. aggregate on STATIC resolution is forbidden
try:
    result.aggregate(pl.mean(result.columns[-1]))
except ValueError:
    errors_caught += 1

# 11b. unknown by key
try:
    result_annual.aggregate(pl.mean(result_annual.columns[-1]), by="quarter")
except ValueError:
    errors_caught += 1

# 11c. sub_annual-only grouping on annual resolution
try:
    result_annual.aggregate(pl.mean(result_annual.columns[-1]), by="season")
except ValueError:
    errors_caught += 1

print(f"Error guards raised  : {errors_caught}/3 (expect 3)")


# ── 12. Result.gdf() TypeError when has_geometry=False ───────────────────────
_section("12. Result.gdf() TypeError  [no geometry guard]")
gdf_err = False
try:
    result_annual.gdf()  # annual result has no geometry
except TypeError:
    gdf_err = True
print(f"gdf() TypeError guard: {gdf_err} (expect True)")


# ── 13. Result.stats / Result.plot namespace access ────────────────────────
_section("13. Result.stats / Result.plot  [namespace accessor cost]")
REPS_NS = 50_000
t0 = time.perf_counter()
for _ in range(REPS_NS):
    result_annual.stats
t1 = time.perf_counter()
print(
    f"result.stats ×{REPS_NS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_NS * 1e6:.2f} µs/call)"
)

t0 = time.perf_counter()
for _ in range(REPS_NS):
    result_annual.plot
t1 = time.perf_counter()
print(
    f"result.plot  ×{REPS_NS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_NS * 1e6:.2f} µs/call)"
)


# ── Cleanup ───────────────────────────────────────────────────────────────────
AoI.deregister(MWSEntity)
print("\n✓ bench_result.py complete")
