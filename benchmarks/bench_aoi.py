"""bench_aoi.py — Benchmark for core_lens.aoi
===========================================
Targets:
  - AoI.register()              entity registration + path probe
  - AoI.__init__(bbox=)         bbox boundary mode
  - AoI.__init__(geometry=)     raw Shapely geometry mode
  - AoI.__init__(**kwargs)      named-boundary resolution (Parquet scan + unary_union)
  - AoI.__getattr__             lazy View creation + first spatial_filter call
  - AoI.deregister()            cleanup

Scalene focuses on:
  - _validate_entity()          schema read at init time
  - _resolve_named_boundary()   scan_parquet + shapely.from_wkb + unary_union

Usage:
    uv run scalene run --html --outfile benchmarks/profiles/bench_aoi.html benchmarks/bench_aoi.py
    uv run python benchmarks/bench_aoi.py   # timing only
"""

from __future__ import annotations

import time

import shapely.geometry as sgeom

from core_lens import AoI, SeasonConfig
from core_lens.entities.mws import MWSEntity

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"

# A mws_id known to exist in your dataset.
MWS_ID = "12_322881"  # update to a valid mws_id in your data

# Bounding box covering all of India (WGS-84).
INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 38.0)

# A small bbox for the named-boundary AoI sanity check.
SMALL_BBOX: tuple[float, float, float, float] = (86.0, 21.0, 87.5, 22.0)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ── 1. Registration ───────────────────────────────────────────────────────────
_section("1. AoI.register(MWSEntity)")
t0 = time.perf_counter()

AoI.register(MWSEntity)

t1 = time.perf_counter()
print(f"register()       : {(t1 - t0) * 1000:.2f} ms")
print(f"Registered       : {AoI.registered_entities()}")


# ── 2. AoI init — bbox mode ───────────────────────────────────────────────────
_section("2. AoI.__init__(bbox=INDIA_BBOX)  [validates all entities]")
t0 = time.perf_counter()

aoi_bbox = AoI(DATA_ROOT, bbox=INDIA_BBOX)

t1 = time.perf_counter()
print(f"AoI(bbox)        : {(t1 - t0) * 1000:.2f} ms")
print(f"Geometry type    : {type(aoi_bbox.geometry).__name__}")


# ── 3. AoI init — raw geometry mode ──────────────────────────────────────────
_section("3. AoI.__init__(geometry=shapely_poly)")
poly = sgeom.box(*SMALL_BBOX)
t0 = time.perf_counter()

aoi_geom = AoI(DATA_ROOT, geometry=poly)

t1 = time.perf_counter()
print(f"AoI(geometry)    : {(t1 - t0) * 1000:.2f} ms")


# ── 4. AoI init — named boundary (most expensive) ────────────────────────────
_section(f"4. AoI.__init__(mws_id={MWS_ID!r})")
t0 = time.perf_counter()

aoi_named = AoI(DATA_ROOT, mws_id=MWS_ID)

t1 = time.perf_counter()
print(f"AoI(named)       : {(t1 - t0) * 1000:.2f} ms")
print(f"Geometry type    : {type(aoi_named.geometry).__name__}")


# ── 5. AoI init — custom SeasonConfig ────────────────────────────────────────
_section("5. AoI.__init__ with custom SeasonConfig")
t0 = time.perf_counter()

aoi_season = AoI(
    DATA_ROOT,
    bbox=INDIA_BBOX,
    seasons=SeasonConfig(
        kharif=("06-15", "10-15"),
        rabi=("10-16", "02-28"),
        zaid=("03-01", "06-14"),
    ),
)

t1 = time.perf_counter()
print(f"AoI(seasons)     : {(t1 - t0) * 1000:.2f} ms")
print(f"Current season   : {aoi_season.current_season}")


# ── 6. View access via __getattr__ ───────────────────────────────────────────
_section("6. aoi.mws  [first __getattr__ call — runs spatial_filter]")
t0 = time.perf_counter()

view = aoi_bbox.mws

t1 = time.perf_counter()
print(f"First access     : {(t1 - t0) * 1000:.2f} ms")
print(f"Keys shape       : {view.keys.collect().shape}")

# Second access — should hit _scoped cache (near-zero cost).
t0 = time.perf_counter()
view2 = aoi_bbox.mws
t1 = time.perf_counter()
print(f"Cached access    : {(t1 - t0) * 1000:.4f} ms  (expect ~0 ms)")


# ── 7. current_season / current_year (property overhead) ─────────────────────
_section("7. AoI.current_season / current_year")
REPS = 10_000
t0 = time.perf_counter()
for _ in range(REPS):
    aoi_bbox.current_season
t1 = time.perf_counter()
print(
    f"current_season ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)

t0 = time.perf_counter()
for _ in range(REPS):
    aoi_bbox.current_year
t1 = time.perf_counter()
print(
    f"current_year   ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 8. registered_entities() ─────────────────────────────────────────────────
_section("8. AoI.registered_entities()")
REPS = 50_000
t0 = time.perf_counter()
for _ in range(REPS):
    AoI.registered_entities()
t1 = time.perf_counter()
print(
    f"registered_entities ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 9. deregister ─────────────────────────────────────────────────────────────
_section("9. AoI.deregister(MWSEntity)")
t0 = time.perf_counter()
AoI.deregister(MWSEntity)
t1 = time.perf_counter()
print(f"deregister()     : {(t1 - t0) * 1000:.4f} ms")
print(f"Remaining        : {AoI.registered_entities()}")

print("\n✓ bench_aoi.py complete")
