"""bench_spatial.py — Benchmark for core_lens.utils.spatial
=========================================================
Targets:
  - build_bbox_index()          cold build (Shapely decode path) vs pre-computed bbox path
  - bbox_intersects_geometry()  pure-Polars rectangular pre-filter
  - exact_spatial_filter()      STRtree centroid + area modes
  - execute_spatial_join()      cross-entity aggregated spatial join

Scalene focuses on:
  - shapely.from_wkb()          WKB decode per-row
  - STRtree construction + query
  - Polars .join() for key semi-join in exact_spatial_filter
  - Python loop in execute_spatial_join aggregation

Usage:
    uv run scalene run --html --outfile benchmarks/profiles/bench_spatial.html benchmarks/bench_spatial.py
    uv run python benchmarks/bench_spatial.py
"""

from __future__ import annotations

import time

import shapely.geometry as sgeom

from core_lens import AoI
from core_lens.entities.mws import MWSEntity
from core_lens.utils.spatial import (
    bbox_intersects_geometry,
    build_bbox_index,
    exact_spatial_filter,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"
INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 38.0)
SMALL_BBOX: tuple[float, float, float, float] = (86.0, 21.0, 87.5, 22.0)

AoI.register(MWSEntity)
entity = MWSEntity(data_root=DATA_ROOT)
static_path = entity._resolve(entity.static_path)
profile = entity.schema_profile


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ── 1. build_bbox_index — cold (Shapely decode path) ─────────────────────────
_section("1. build_bbox_index() — cold start (Shapely decode)")
t0 = time.perf_counter()

index = build_bbox_index(
    static_path=static_path,
    key_cols=entity.key_cols,
    bbox_cols=profile.bbox_cols,  # None if no pre-computed cols
    geometry_col=profile.geometry_col,
    geometry_type=profile.geometry_type,
)

t1 = time.perf_counter()
print(f"build_bbox_index : {(t1 - t0) * 1000:.2f} ms")
print(f"Index shape      : {index.shape}")
print(f"Bbox cols used   : {profile.bbox_cols}")


# ── 2. build_bbox_index — warm (cached on entity._index) ─────────────────────
_section("2. entity._index  [cached access]")
t0 = time.perf_counter()

cached_index = entity._index  # first call triggers build; subsequent are free

t1 = time.perf_counter()
print(f"First _index     : {(t1 - t0) * 1000:.2f} ms")

t0 = time.perf_counter()
entity._index
t1 = time.perf_counter()
print(f"Cached _index    : {(t1 - t0) * 1000:.4f} ms  (expect ~0 ms)")


# ── 3. bbox_intersects_geometry — large geometry (all India) ─────────────────
_section("3. bbox_intersects_geometry() — all-India bbox")
india_poly = sgeom.box(*INDIA_BBOX)
REPS = 1_000
candidates = bbox_intersects_geometry(index, india_poly)  # keep last result for later
t0 = time.perf_counter()
for _ in range(REPS):
    candidates = bbox_intersects_geometry(index, india_poly)
t1 = time.perf_counter()
print(
    f"bbox_intersects ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)
print(f"Candidates       : {len(candidates)} / {len(index)}")


# ── 4. bbox_intersects_geometry — small geometry (Banka patch) ───────────────
_section("4. bbox_intersects_geometry() — small bbox")
small_poly = sgeom.box(*SMALL_BBOX)
candidates_small = bbox_intersects_geometry(
    index, small_poly
)  # keep last result for later
t0 = time.perf_counter()
for _ in range(REPS):
    candidates_small = bbox_intersects_geometry(index, small_poly)
t1 = time.perf_counter()
print(
    f"bbox_intersects ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)
print(f"Candidates       : {len(candidates_small)} / {len(index)}")


# ── 5. exact_spatial_filter — centroid mode ───────────────────────────────────
_section("5. exact_spatial_filter() — centroid mode (small bbox)")
t0 = time.perf_counter()

keys_centroid = exact_spatial_filter(
    candidates=candidates_small,
    static_path=static_path,
    key_cols=entity.key_cols,
    geometry_col=profile.geometry_col,
    geometry_type=profile.geometry_type,
    aoi_geometry=small_poly,
    relationship="centroid",
)

t1 = time.perf_counter()
print(f"exact_spatial (centroid): {(t1 - t0) * 1000:.2f} ms")
print(f"Matched (centroid)      : {len(keys_centroid)}")


# ── 6. exact_spatial_filter — area mode ──────────────────────────────────────
_section("6. exact_spatial_filter() — area mode (small bbox)")
t0 = time.perf_counter()

keys_area = exact_spatial_filter(
    candidates=candidates_small,
    static_path=static_path,
    key_cols=entity.key_cols,
    geometry_col=profile.geometry_col,
    geometry_type=profile.geometry_type,
    aoi_geometry=small_poly,
    relationship="area",
    threshold=0.5,
)

t1 = time.perf_counter()
print(f"exact_spatial (area)    : {(t1 - t0) * 1000:.2f} ms")
print(f"Matched (area ≥ 50%)    : {len(keys_area)}")


# ── 7. exact_spatial_filter — large candidate set (all India) ────────────────
_section("7. exact_spatial_filter() — centroid mode (all India)")
t0 = time.perf_counter()

keys_all = exact_spatial_filter(
    candidates=candidates,
    static_path=static_path,
    key_cols=entity.key_cols,
    geometry_col=profile.geometry_col,
    geometry_type=profile.geometry_type,
    aoi_geometry=india_poly,
    relationship="centroid",
)

t1 = time.perf_counter()
print(f"exact_spatial (all India): {(t1 - t0) * 1000:.2f} ms")
print(f"Matched                  : {len(keys_all)}")


# ── 8. Full spatial_filter pipeline (entity method) ──────────────────────────
_section("8. entity.spatial_filter(bbox=) — full pipeline")
t0 = time.perf_counter()

view = entity.spatial_filter(bbox=SMALL_BBOX)

t1 = time.perf_counter()
print(f"spatial_filter (bbox) : {(t1 - t0) * 1000:.2f} ms")
print(f"View keys             : {view.keys.collect().shape}")

t0 = time.perf_counter()
view2 = entity.spatial_filter(geometry=small_poly)
t1 = time.perf_counter()
print(f"spatial_filter (geom) : {(t1 - t0) * 1000:.2f} ms")


# ── Cleanup ───────────────────────────────────────────────────────────────────
AoI.deregister(MWSEntity)
print("\n✓ bench_spatial.py complete")
