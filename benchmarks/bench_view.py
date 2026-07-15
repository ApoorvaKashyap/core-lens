"""bench_view.py — Benchmark for core_lens.base.view.View
=======================================================
Targets:
  - View.where()                attribute filter (Polars semi-join on static file)
  - View.spatial_filter()       STRtree filter narrowing existing keys
  - View.between()              time-filter dict construction (no I/O)
  - View._materialise()         static / annual / fortnightly Parquet reads
  - View chaining               where → spatial_filter → between → static

Scalene focuses on:
  - scan_with_key_filter()      predicate-pushdown LazyFrame
  - collect_lf()                streaming collect (CPU or GPU)
  - resolve_time_filter()       expression build for time predicates
  - add_temporal_columns()      fortnightly temporal column derivation

Usage:
    uv run scalene run --html --outfile benchmarks/profiles/bench_view.html benchmarks/bench_view.py
    uv run python benchmarks/bench_view.py
"""

from __future__ import annotations

import time

import shapely.geometry as sgeom

from core_lens import AoI
from core_lens.base.view import Season
from core_lens.entities.mws import MWSEntity

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"
INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 38.0)
SMALL_BBOX: tuple[float, float, float, float] = (86.0, 21.0, 87.5, 22.0)

AoI.register(MWSEntity)
aoi = AoI(DATA_ROOT, bbox=INDIA_BBOX)
aoi_small = AoI(DATA_ROOT, bbox=SMALL_BBOX)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ── 1. View creation via AoI.__getattr__ ─────────────────────────────────────
_section("1. aoi.mws  [creates scoped View via spatial_filter]")
t0 = time.perf_counter()
view_all = aoi.mws
t1 = time.perf_counter()
print(f"aoi.mws          : {(t1 - t0) * 1000:.2f} ms")
print(f"Keys             : {view_all.keys.collect().shape}")


# ── 2. View.where() — single filter ──────────────────────────────────────────
_section("2. View.where(state='Bihar')")
t0 = time.perf_counter()
view_bihar = view_all.where(state="Bihar")
t1 = time.perf_counter()
print(f"where(state)     : {(t1 - t0) * 1000:.2f} ms")
print(f"Bihar keys       : {view_bihar.keys.collect().shape}")


# ── 3. View.where() — composite AND filter ────────────────────────────────────
_section("3. View.where(state='Bihar', district='Banka', tehsil='Banka')")
t0 = time.perf_counter()
view_banka = view_all.where(state="Bihar", district="Banka", tehsil="Banka")
t1 = time.perf_counter()
print(f"where(3 cols)    : {(t1 - t0) * 1000:.2f} ms")
print(f"Banka keys       : {view_banka.keys.collect().shape}")


# ── 4. View.spatial_filter() — further narrow ─────────────────────────────────
_section("4. view_all.spatial_filter(bbox=SMALL_BBOX)")
small_poly = sgeom.box(*SMALL_BBOX)
t0 = time.perf_counter()
view_spatial = view_all.spatial_filter(bbox=SMALL_BBOX)
t1 = time.perf_counter()
print(f"spatial_filter   : {(t1 - t0) * 1000:.2f} ms")
print(f"Spatial keys     : {view_spatial.keys.collect().shape}")


# ── 5. View.between() — date range (no I/O) ───────────────────────────────────
_section("5. View.between(start, end)  [pure dict construction — no I/O]")
REPS = 100_000
t0 = time.perf_counter()
for _ in range(REPS):
    view_bihar.between("2010-01-01", "2023-12-31")
t1 = time.perf_counter()
print(
    f"between(dates) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 6. View.between() — season mode ──────────────────────────────────────────
_section("6. View.between(season=Season.KHARIF, year=2022)")
t0 = time.perf_counter()
for _ in range(REPS):
    view_bihar.between(season=Season.KHARIF, year=2022)
t1 = time.perf_counter()
print(
    f"between(season) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 7. View.static — full materialise (all rows) ─────────────────────────────
_section("7. aoi.mws.static  [full static materialisation]")
t0 = time.perf_counter()
result_static = aoi.mws.static
t1 = time.perf_counter()
print(f"static (all)     : {(t1 - t0) * 1000:.2f} ms")
print(f"Shape            : {result_static.df().shape}")

_section("8. aoi_small.mws.static  [small-bbox materialisation]")
t0 = time.perf_counter()
result_small = aoi_small.mws.static
t1 = time.perf_counter()
print(f"static (small)   : {(t1 - t0) * 1000:.2f} ms")
print(f"Shape            : {result_small.df().shape}")


# ── 9. Chained pipeline: where → spatial_filter → static ─────────────────────
_section("9. Full chain: where → spatial_filter → static")
t0 = time.perf_counter()
result_chain = aoi.mws.spatial_filter(bbox=SMALL_BBOX).static
t1 = time.perf_counter()
print(f"Chain            : {(t1 - t0) * 1000:.2f} ms")
print(f"Shape            : {result_chain.df().shape}")


# ── 10. annual materialisation ──────────────────────────────────────────────────
_section("10. aoi_small.mws.annual  [annual materialisation]")
t0 = time.perf_counter()
result_annual = aoi_small.mws.between("2018-01-01", "2023-12-31").annual
t1 = time.perf_counter()
print(f"annual           : {(t1 - t0) * 1000:.2f} ms")
print(f"Shape            : {result_annual.df().shape}")


# ── 11. fortnightly materialisation ───────────────────────────────────────────
_section("11. aoi_small.mws.fortnightly  [fortnightly materialisation]")
t0 = time.perf_counter()
result_fn = aoi_small.mws.between(season=Season.KHARIF, year=2022).fortnightly
t1 = time.perf_counter()
print(f"fortnightly      : {(t1 - t0) * 1000:.2f} ms")
print(f"Shape            : {result_fn.df().shape}")


# ── Cleanup ───────────────────────────────────────────────────────────────────
AoI.deregister(MWSEntity)
print("\n✓ bench_view.py complete")
