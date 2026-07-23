"""bench_schema.py — Benchmark for core_lens.schema.detection
============================================================
Targets:
  - detect()                    full schema auto-detection from Parquet metadata
  - _read_schema()              pl.scan_parquet(...).collect_schema() overhead
  - _infer_geometry_type()      dtype-based geometry type inference
  - _infer_bbox_cols()          pattern-match against known bbox column sets
  - _infer_time_col()           name + dtype heuristic time column detection
  - _is_year_col_from_schema()  integer-year vs Date/Datetime column detection
  - _find_lon_companion()       lat-column → lon companion heuristic
  - _require_cols()             missing-column guard (error path)
  - SchemaProfile.from_file()   JSON round-trip construction
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
    _is_year_col_from_schema,
    _find_lon_companion,
    _require_cols,
    SchemaDetectionError,
)
from core_lens.schema.profile import SchemaProfile

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"

entity = MWSEntity(data_root=DATA_ROOT)
static_path = entity._resolve(entity.static_path)
annual_path = entity._resolve(entity.annual_path) if entity.annual_path else None
fn_path = entity._resolve(entity.sub_annual_path) if entity.sub_annual_path else None


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


# ── 5. detect() — cold (no annual/sub_annual) ────────────────────────────────
_section("5. detect(static only)  [full detection — cold]")
REPS = 10
profile = detect(  # ensure bound before loop
    static_path=static_path,
    key_cols=entity.key_cols,
    geometry_col=entity.geometry_col,
    annual_path=None,
    sub_annual_path=None,
)
t0 = time.perf_counter()
for _ in range(REPS):
    profile = detect(
        static_path=static_path,
        key_cols=entity.key_cols,
        geometry_col=entity.geometry_col,
        annual_path=None,
        sub_annual_path=None,
    )
t1 = time.perf_counter()
print(
    f"detect(static) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1000:.2f} ms/call)"
)
print(f"Profile geometry_type : {profile.geometry_type!r}")
print(f"Profile bbox_cols     : {profile.bbox_cols!r}")
print(f"Profile extra_static  : {profile.extra_static_cols}")


# ── 6. detect() — with annual + sub_annual ───────────────────────────────────
if annual_path or fn_path:
    _section("6. detect(static + annual + sub_annual)  [3× schema reads]")
    profile_full = detect(  # ensure bound before loop
        static_path=static_path,
        key_cols=entity.key_cols,
        geometry_col=entity.geometry_col,
        annual_path=annual_path,
        sub_annual_path=fn_path,
    )
    t0 = time.perf_counter()
    for _ in range(REPS):
        profile_full = detect(
            static_path=static_path,
            key_cols=entity.key_cols,
            geometry_col=entity.geometry_col,
            annual_path=annual_path,
            sub_annual_path=fn_path,
        )
    t1 = time.perf_counter()
    print(
        f"detect(full) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS * 1000:.2f} ms/call)"
    )
    print(f"annual_time_col      : {profile_full.annual_time_col!r}")
    print(f"sub_annual_time_col : {profile_full.sub_annual_time_col!r}")
else:
    print(
        "\n[skip] No annual/sub_annual paths on TehsilEntity — add an entity with those paths"
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


# ── 8. _is_year_col_from_schema() ────────────────────────────────────────────
_section("8. _is_year_col_from_schema()  [dtype check — O(1)]")
# annual schema — has an integer year column
if annual_path:
    annual_schema_for_test = _read_schema(annual_path, label="annual")
    REPS = 500_000
    t0 = time.perf_counter()
    for _ in range(REPS):
        _is_year_col_from_schema(annual_schema_for_test, "year")
    t1 = time.perf_counter()
    print(
        f"_is_year_col ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS * 1e6:.3f} µs/call)"
    )
    print(
        f"is_year_col (annual 'year'): {_is_year_col_from_schema(annual_schema_for_test, 'year')!r}"
    )
    print(
        f"is_year_col (None):          {_is_year_col_from_schema(annual_schema_for_test, None)!r}"
    )
else:
    print("[skip] No annual path on entity")


# ── 9. _find_lon_companion() ─────────────────────────────────────────────────
_section("9. _find_lon_companion()  [lat→lon heuristic — O(candidates)]")
import polars as pl  # noqa: E401, E402  (already imported; re-stated for Scalene trace clarity)

# Build a synthetic schema with lat + lon columns to exercise the positive branch.
synthetic_schema = pl.Schema(
    {
        "id": pl.Int64,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "ndvi": pl.Float32,
    }
)
REPS_FC = 500_000
t0 = time.perf_counter()
for _ in range(REPS_FC):
    _find_lon_companion(synthetic_schema, "lat")
t1 = time.perf_counter()
print(
    f"_find_lon_companion ×{REPS_FC}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_FC * 1e6:.3f} µs/call)"
)
print(
    f"Found companion      : {_find_lon_companion(synthetic_schema, 'lat')!r}  (expect 'lon')"
)
print(
    f"No companion         : {_find_lon_companion(synthetic_schema, 'ndvi')!r}  (expect None)"
)


# ── 10. _require_cols() — error path ─────────────────────────────────────────
_section("10. _require_cols() error path  [missing column raises SchemaDetectionError]")
err_raised = False
try:
    _require_cols(schema, ["nonexistent_col_xyz"], static_path)
except SchemaDetectionError:
    err_raised = True
print(f"SchemaDetectionError  : {err_raised} (expect True)")

# Happy path — should not raise.
_require_cols(schema, list(schema.names()[:2]), static_path)
print("_require_cols (ok)    : no error (expect no error)")


# ── 11. SchemaProfile.from_file() — JSON round-trip ──────────────────────────
_section("11. SchemaProfile.from_file()  [JSON round-trip]")
import json  # noqa: E402
import tempfile  # noqa: E402
import pathlib  # noqa: E402

profile2 = entity2.schema_profile
profile_dict = profile2.model_dump()

with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
    json.dump(profile_dict, tmp)
    tmp_path = tmp.name

REPS_FF = 1_000
t0 = time.perf_counter()
for _ in range(REPS_FF):
    reloaded = SchemaProfile.from_file(tmp_path)
t1 = time.perf_counter()
print(
    f"from_file ×{REPS_FF}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_FF * 1000:.2f} ms/call)"
)
print(f"Round-trip geometry_type: {reloaded.geometry_type!r}")
pathlib.Path(tmp_path).unlink(missing_ok=True)


print("\n✓ bench_schema.py complete")
