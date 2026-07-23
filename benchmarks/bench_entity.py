"""bench_entity.py — Benchmark for core_lens.base.entity.BaseEntity
===================================================================
Targets:
  - BaseEntity._resolve()        absolute / relative / cloud URI resolution
  - BaseEntity.where()           attribute filter (single col, composite, list val)
  - BaseEntity.where()           entity-kwarg spatial lookup path
  - BaseEntity.spatial_filter()  full pipeline (bbox + geometry mode)
  - BaseEntity.spatial_join()    deferred join spec construction
  - BaseEntity.schema_profile    instance-level cache (hasattr guard)
  - BaseEntity._index            process-level cache (_cached_build_index)
  - BaseEntity.geometry_lazy     lazy geometry scan construction
  - _so_key()                    storage_options → hashable tuple
  - _entity_name()               entity class name → string
  - _cached_detect()             process-level schema cache (functools.cache)
  - _cached_build_index()        process-level bbox index cache
  - EntityValidationError        exception class (import overhead)

Scalene focuses on:
  - _resolve() pathlib.Path ops vs cloud URI fast-path
  - cached_read_schema() in where() — hot path schema lookup
  - bbox_intersects_geometry() → exact_spatial_filter() inside where()
  - functools.cache lookup overhead in _cached_detect / _cached_build_index

Usage:
    uv run scalene run --html --outfile benchmarks/profiles/bench_entity.html benchmarks/bench_entity.py
    uv run python benchmarks/bench_entity.py
"""

from __future__ import annotations

import time

import shapely.geometry as sgeom

from core_lens import AoI
from core_lens.entities.mws import MWSEntity
from core_lens.base.entity import (
    _so_key,
    _entity_name,
    _cached_detect,
    EntityValidationError,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"
INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 38.0)
SMALL_BBOX: tuple[float, float, float, float] = (86.0, 21.0, 87.5, 22.0)
MWS_ID = "12_322881"  # update to a valid mws_id in your dataset

AoI.register(MWSEntity)
entity = MWSEntity(data_root=DATA_ROOT)
static_path = entity._resolve(entity.static_path)


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ── 1. _so_key() — storage_options → hashable tuple ──────────────────────────
_section("1. _so_key()  [dict → sorted tuple — O(n log n)]")
REPS = 1_000_000

so_empty: dict[str, str] = {}
so_aws: dict[str, str] = {"region": "ap-south-1", "access_key": "AKIAIOSFODNN7EXAMPLE"}

t0 = time.perf_counter()
for _ in range(REPS):
    _so_key(so_empty)
t1 = time.perf_counter()
print(
    f"_so_key (empty) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e9:.1f} ns/call)"
)

t0 = time.perf_counter()
for _ in range(REPS):
    _so_key(so_aws)
t1 = time.perf_counter()
print(
    f"_so_key (2 keys) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e9:.1f} ns/call)"
)
print(f"  _so_key(aws): {_so_key(so_aws)}")


# ── 2. _entity_name() — class → string ───────────────────────────────────────
_section("2. _entity_name()  [class → name string, 'Entity' suffix strip]")
REPS_EN = 1_000_000
t0 = time.perf_counter()
for _ in range(REPS_EN):
    _entity_name(MWSEntity)
t1 = time.perf_counter()
print(
    f"_entity_name ×{REPS_EN}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_EN * 1e9:.1f} ns/call)"
)
print(f"  _entity_name(MWSEntity): {_entity_name(MWSEntity)!r}  (expect 'mws')")


# ── 3. BaseEntity._resolve() — local relative path ───────────────────────────
_section("3. BaseEntity._resolve()  [relative → absolute local path]")
REPS_R = 100_000
t0 = time.perf_counter()
for _ in range(REPS_R):
    entity._resolve(entity.static_path)
t1 = time.perf_counter()
print(
    f"_resolve (relative) ×{REPS_R}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_R * 1e6:.2f} µs/call)"
)
print(f"  resolved: {entity._resolve(entity.static_path)!r}")

# Cloud URI fast-path (no pathlib.Path, no existence check).
entity_cloud_sim = MWSEntity.__new__(MWSEntity)  # bypass __init__ for testing
entity_cloud_sim._data_root = "s3://my-bucket/data"
entity_cloud_sim._storage_options = {}
REPS_CR = 100_000
t0 = time.perf_counter()
for _ in range(REPS_CR):
    entity_cloud_sim._resolve("s3://my-bucket/data/mws/static/")
t1 = time.perf_counter()
print(
    f"_resolve (cloud abs) ×{REPS_CR}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_CR * 1e6:.2f} µs/call)"
)


# ── 4. BaseEntity.schema_profile — instance cache ────────────────────────────
_section("4. BaseEntity.schema_profile  [hasattr guard — instance-level cache]")
entity2 = MWSEntity(data_root=DATA_ROOT)
t0 = time.perf_counter()
entity2.schema_profile  # cold — triggers _cached_detect
t1 = time.perf_counter()
print(f"schema_profile (cold) : {(t1 - t0) * 1000:.2f} ms")

REPS_SP = 100_000
t0 = time.perf_counter()
for _ in range(REPS_SP):
    entity2.schema_profile  # warm — hasattr guard short-circuits
t1 = time.perf_counter()
print(
    f"schema_profile (warm) ×{REPS_SP}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_SP * 1e6:.3f} µs/call)"
)


# ── 5. _cached_detect() — process-level functools.cache ──────────────────────
_section("5. _cached_detect()  [process-level functools.cache lookup]")
profile = entity.schema_profile
key = (
    static_path,
    tuple(entity.key_cols),
    entity.geometry_col,
    entity._resolve(entity.annual_path) if entity.annual_path else None,
    entity._resolve(entity.sub_annual_path) if entity.sub_annual_path else None,
    _so_key(entity._storage_options),
)
# Prime the cache.
_cached_detect(*key)

REPS_CD = 100_000
t0 = time.perf_counter()
for _ in range(REPS_CD):
    _cached_detect(*key)
t1 = time.perf_counter()
print(
    f"_cached_detect (warm) ×{REPS_CD}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_CD * 1e6:.3f} µs/call)"
)
print(f"Cache info: {_cached_detect.cache_info()}")


# ── 6. BaseEntity._index — process-level bbox cache ──────────────────────────
_section("6. BaseEntity._index  [process-level bbox index cache]")
t0 = time.perf_counter()
idx = entity._index  # may be warm from earlier modules
t1 = time.perf_counter()
print(f"_index (first access) : {(t1 - t0) * 1000:.2f} ms")
print(f"Index shape           : {idx.shape}")

REPS_IX = 100_000
t0 = time.perf_counter()
for _ in range(REPS_IX):
    entity._index  # warm — hasattr guard short-circuits
t1 = time.perf_counter()
print(
    f"_index (cached) ×{REPS_IX}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_IX * 1e6:.3f} µs/call)"
)


# ── 7. BaseEntity.geometry_lazy — lazy geometry LazyFrame ────────────────────
_section("7. BaseEntity.geometry_lazy  [lazy scan construction + hasattr cache]")
entity3 = MWSEntity(data_root=DATA_ROOT)
t0 = time.perf_counter()
lf = entity3.geometry_lazy  # cold — builds pl.scan_parquet plan
t1 = time.perf_counter()
print(f"geometry_lazy (cold)  : {(t1 - t0) * 1000:.2f} ms")

REPS_GL = 100_000
t0 = time.perf_counter()
for _ in range(REPS_GL):
    entity3.geometry_lazy  # warm — hasattr guard short-circuits
t1 = time.perf_counter()
print(
    f"geometry_lazy (warm) ×{REPS_GL}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_GL * 1e6:.3f} µs/call)"
)


# ── 8. BaseEntity.where() — single attribute filter ──────────────────────────
_section("8. BaseEntity.where(state='Bihar')  [single attribute filter]")
t0 = time.perf_counter()
view_attr = entity.where(state="Bihar")
t1 = time.perf_counter()
print(f"where(state)         : {(t1 - t0) * 1000:.2f} ms")
print(f"Bihar keys           : {view_attr.keys.collect().shape}")


# ── 9. BaseEntity.where() — composite attribute filter ───────────────────────
_section("9. BaseEntity.where(state=..., district=..., tehsil=...)  [composite AND]")
t0 = time.perf_counter()
view_composite = entity.where(state="Bihar", district="Banka", tehsil="Banka")
t1 = time.perf_counter()
print(f"where(3 cols)        : {(t1 - t0) * 1000:.2f} ms")
print(f"Banka keys           : {view_composite.keys.collect().shape}")


# ── 10. BaseEntity.where() — list value (is_in) ──────────────────────────────
_section("10. BaseEntity.where(state=['Bihar', 'Jharkhand'])  [list value → is_in]")
t0 = time.perf_counter()
view_list = entity.where(state=["Bihar", "Jharkhand"])
t1 = time.perf_counter()
print(f"where(list)          : {(t1 - t0) * 1000:.2f} ms")
print(f"Bihar+Jharkhand keys : {view_list.keys.collect().shape}")


# ── 11. BaseEntity.where() — unknown kwarg error ─────────────────────────────
_section("11. BaseEntity.where()  [unknown kwarg → ValueError guard]")
err_raised = False
try:
    entity.where(totally_unknown_column="xyz")
except ValueError:
    err_raised = True
print(f"ValueError guard     : {err_raised} (expect True)")


# ── 12. BaseEntity.spatial_filter() — bbox mode ──────────────────────────────
_section("12. BaseEntity.spatial_filter(bbox=SMALL_BBOX)  [full pipeline]")
t0 = time.perf_counter()
view_sf = entity.spatial_filter(bbox=SMALL_BBOX)
t1 = time.perf_counter()
print(f"spatial_filter (bbox): {(t1 - t0) * 1000:.2f} ms")
print(f"Keys                 : {view_sf.keys.collect().shape}")

# geometry mode
small_poly = sgeom.box(*SMALL_BBOX)
t0 = time.perf_counter()
view_sf2 = entity.spatial_filter(geometry=small_poly)
t1 = time.perf_counter()
print(f"spatial_filter (geom): {(t1 - t0) * 1000:.2f} ms")

# area relationship
t0 = time.perf_counter()
view_sf_area = entity.spatial_filter(
    bbox=SMALL_BBOX, relationship="area", threshold=0.3
)
t1 = time.perf_counter()
print(
    f"spatial_filter (area): {(t1 - t0) * 1000:.2f} ms  keys={view_sf_area.keys.collect().shape}"
)


# ── 13. BaseEntity.spatial_join() — deferred spec ────────────────────────────
_section("13. BaseEntity.spatial_join()  [deferred spec construction — no I/O]")
REPS_SJ = 10_000
t0 = time.perf_counter()
for _ in range(REPS_SJ):
    entity.spatial_join(other=entity, agg={"area_in_ha": "sum"})
t1 = time.perf_counter()
print(
    f"spatial_join (spec) ×{REPS_SJ}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_SJ * 1e6:.2f} µs/call)"
)


# ── 14. EntityValidationError — import / raise overhead ──────────────────────
_section("14. EntityValidationError  [raise + catch overhead]")
REPS_EV = 100_000
t0 = time.perf_counter()
for _ in range(REPS_EV):
    try:
        raise EntityValidationError("bench test")
    except EntityValidationError:
        pass
t1 = time.perf_counter()
print(
    f"raise+catch ×{REPS_EV}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_EV * 1e6:.2f} µs/call)"
)


# ── Cleanup ───────────────────────────────────────────────────────────────────
AoI.deregister(MWSEntity)
print("\n✓ bench_entity.py complete")
