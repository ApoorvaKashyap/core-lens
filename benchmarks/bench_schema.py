"""
bench_schema.py — Benchmark for core_lens.schema.detection
============================================================
Targets:
  - detect()                    full schema auto-detection from Parquet metadata
  - _read_schema()              pl.scan_parquet(...).collect_schema() overhead
  - _infer_geometry_type()      dtype-based geometry type inference
  - _infer_bbox_cols()          pattern-match against known bbox column sets
  - _infer_time_col()           name + dtype heuristic time column detection
  - entity.schema_profile       cached vs cold access on BaseEntity

Scalene focuses on:
  - pl.scan_parquet() + collect_schema() — Parquet footer read cost
  - Python list comprehensions in _require_cols, extra_*_cols derivation

Usage:
    uv run scalene run --html --outfile benchmarks/profiles/bench_schema.html benchmarks/bench_schema.py
    uv run python benchmarks/bench_schema.py
"""

from __future__ import annotations

import time

from core_lens.entities.mws import MWSEntity
from core_lens.schema.detection import (
    detect,
    _infer_bbox_cols,
    _infer_geometry_type,
    _infer_time_col,
    _read_schema,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"

entity = MWSEntity(data_root=DATA_ROOT)
static_path = entity._resolve(entity.static_path)
annual_path = entity._resolve(entity.annual_path) if entity.annual_path else None
fn_path = entity._resolve(entity.fortnightly_path) if entity.fortnightly_path else None


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ── 1. _read_schema() — static file ───────────────────────────────────────────
_section("1. _read_schema(static_path)  [Parquet footer read]")
REPS = 20
schema = _read_schema(static_path, label="static")  # ensure bound before loop
t0 = time.perf_counter()
for _ in range(REPS):
    schema = _read_schema(static_path, label="static")
t1 = time.perf_counter()
print(
    f"_read_schema ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1000:.2f} ms/call)"
)
print(f"Columns          : {list(schema.names())}")


# ── 2. _infer_geometry_type() ─────────────────────────────────────────────────
_section("2. _infer_geometry_type()  [dtype check — O(1)]")
REPS = 500_000
t0 = time.perf_counter()
for _ in range(REPS):
    _infer_geometry_type(schema, entity.geometry_col, static_path)
t1 = time.perf_counter()
print(
    f"_infer_geom_type ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.3f} µs/call)"
)
geom_type, lon_col = _infer_geometry_type(schema, entity.geometry_col, static_path)
print(f"Detected type    : {geom_type!r}, lon_col={lon_col!r}")


# ── 3. _infer_bbox_cols() ─────────────────────────────────────────────────────
_section("3. _infer_bbox_cols()  [4-pattern scan — O(patterns × 4)]")
REPS = 500_000
t0 = time.perf_counter()
for _ in range(REPS):
    _infer_bbox_cols(schema)
t1 = time.perf_counter()
print(
    f"_infer_bbox_cols ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.3f} µs/call)"
)
bbox_cols = _infer_bbox_cols(schema)
print(f"Detected bbox    : {bbox_cols!r}")


# ── 4. _infer_time_col() — static schema (no time col expected) ───────────────
_section("4. _infer_time_col(static_schema)  [name + dtype heuristic]")
REPS = 500_000
t0 = time.perf_counter()
for _ in range(REPS):
    _infer_time_col(schema, static_path)
t1 = time.perf_counter()
print(
    f"_infer_time_col ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.3f} µs/call)"
)
print(f"Detected time col: {_infer_time_col(schema, static_path)!r}")


# ── 5. detect() — cold (no annual/fortnightly) ────────────────────────────────
_section("5. detect(static only)  [full detection — cold]")
REPS = 10
profile = detect(  # ensure bound before loop
    static_path=static_path,
    key_cols=entity.key_cols,
    geometry_col=entity.geometry_col,
    annual_path=None,
    fortnightly_path=None,
)
t0 = time.perf_counter()
for _ in range(REPS):
    profile = detect(
        static_path=static_path,
        key_cols=entity.key_cols,
        geometry_col=entity.geometry_col,
        annual_path=None,
        fortnightly_path=None,
    )
t1 = time.perf_counter()
print(
    f"detect(static) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1000:.2f} ms/call)"
)
print(f"Profile geometry_type : {profile.geometry_type!r}")
print(f"Profile bbox_cols     : {profile.bbox_cols!r}")
print(f"Profile extra_static  : {profile.extra_static_cols}")


# ── 6. detect() — with annual + fortnightly ───────────────────────────────────
if annual_path or fn_path:
    _section("6. detect(static + annual + fortnightly)  [3× schema reads]")
    profile_full = detect(  # ensure bound before loop
        static_path=static_path,
        key_cols=entity.key_cols,
        geometry_col=entity.geometry_col,
        annual_path=annual_path,
        fortnightly_path=fn_path,
    )
    t0 = time.perf_counter()
    for _ in range(REPS):
        profile_full = detect(
            static_path=static_path,
            key_cols=entity.key_cols,
            geometry_col=entity.geometry_col,
            annual_path=annual_path,
            fortnightly_path=fn_path,
        )
    t1 = time.perf_counter()
    print(
        f"detect(full) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS * 1000:.2f} ms/call)"
    )
    print(f"annual_time_col      : {profile_full.annual_time_col!r}")
    print(f"fortnightly_time_col : {profile_full.fortnightly_time_col!r}")
else:
    print(
        "\n[skip] No annual/fortnightly paths on TehsilEntity — add an entity with those paths"
    )


# ── 7. entity.schema_profile — cold vs cached ─────────────────────────────────
_section("7. entity.schema_profile  [cold vs cached via hasattr guard]")

# Fresh entity (no cache).
entity2 = MWSEntity(data_root=DATA_ROOT)
t0 = time.perf_counter()
entity2.schema_profile  # cold
t1 = time.perf_counter()
print(f"schema_profile (cold)  : {(t1 - t0) * 1000:.2f} ms")

REPS = 100_000
t0 = time.perf_counter()
for _ in range(REPS):
    entity2.schema_profile  # warm — hasattr guard short-circuits
t1 = time.perf_counter()
print(
    f"schema_profile (warm) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.3f} µs/call)"
)

print("\n✓ bench_schema.py complete")
