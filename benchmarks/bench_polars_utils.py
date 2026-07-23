"""bench_polars_utils.py — Benchmark for core_lens.utils.polars_utils
===================================================================
Targets:
  - _gpu_available()            module-level probe + caching
  - collect_lf()                CPU streaming collect vs standard collect
  - scan_with_key_filter()      LazyFrame with key semi-join predicate build

Scalene focuses on:
  - collect(engine="streaming") vs collect() — streaming overhead
  - pl.scan_parquet() + join(lazy) predicate pushdown efficiency
  - large key set (many IDs) vs small key set

Usage:
    uv run scalene run --html --outfile benchmarks/profiles/bench_polars_utils.html benchmarks/bench_polars_utils.py
    uv run python benchmarks/bench_polars_utils.py
"""

from __future__ import annotations

import time

import polars as pl

from core_lens import AoI
from core_lens.aoi import SeasonConfig
from core_lens.entities.mws import MWSEntity
from core_lens.utils.polars_utils import (
    _gpu_available,
    collect_lf,
    scan_with_key_filter,
)
from core_lens.utils.season import resolve_time_filter

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"
INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 38.0)
SMALL_BBOX: tuple[float, float, float, float] = (86.0, 21.0, 87.5, 22.0)

AoI.register(MWSEntity)
entity = MWSEntity(data_root=DATA_ROOT)
static_path = entity._resolve(entity.static_path)

# Build key sets: all keys vs a small subset.
all_keys = pl.read_parquet(static_path, columns=entity.key_cols)
small_keys = all_keys.head(50)
single_key = all_keys.head(1)

print(f"All keys   : {len(all_keys)}")
print(f"Small keys : {len(small_keys)}")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ── 1. _gpu_available() — probe + cache ──────────────────────────────────────
_section("1. _gpu_available()  [first call = import probe; subsequent = cached]")
REPS = 1_000_000
t0 = time.perf_counter()
for _ in range(REPS):
    _gpu_available()
t1 = time.perf_counter()
print(
    f"_gpu_available ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e9:.1f} ns/call)"
)
print(f"GPU available  : {_gpu_available()}")


# ── 2. scan_with_key_filter() — LazyFrame construction (no collect) ───────────
_section("2. scan_with_key_filter()  [LazyFrame construction — no I/O]")
REPS = 10_000
t0 = time.perf_counter()
for _ in range(REPS):
    lf = scan_with_key_filter(
        path=static_path,
        key_cols=entity.key_cols,
        key_values=small_keys,
    )
t1 = time.perf_counter()
print(
    f"scan_with_key (build) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 3. collect_lf() — single key ─────────────────────────────────────────────
_section("3. collect_lf()  — single key (1 entity row)")
REPS = 10
lf_single = scan_with_key_filter(
    path=static_path,
    key_cols=entity.key_cols,
    key_values=single_key,
)
# Materialise once outside loop to warm filesystem cache.
collect_lf(lf_single)

t0 = time.perf_counter()
for _ in range(REPS):
    lf = scan_with_key_filter(
        path=static_path,
        key_cols=entity.key_cols,
        key_values=single_key,
    )
    collect_lf(lf)
t1 = time.perf_counter()
print(
    f"collect(1 key) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1000:.2f} ms/call)"
)


# ── 4. collect_lf() — small key set ──────────────────────────────────────────
_section(f"4. collect_lf()  — small key set ({len(small_keys)} entities)")
collect_lf(scan_with_key_filter(static_path, entity.key_cols, small_keys))  # warm

df = collect_lf(
    scan_with_key_filter(  # ensure bound before loop
        path=static_path, key_cols=entity.key_cols, key_values=small_keys
    )
)
t0 = time.perf_counter()
for _ in range(REPS):
    lf = scan_with_key_filter(
        path=static_path,
        key_cols=entity.key_cols,
        key_values=small_keys,
    )
    df = collect_lf(lf)
t1 = time.perf_counter()
print(
    f"collect(small) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1000:.2f} ms/call)"
)
print(f"Rows returned  : {len(df)}")


# ── 5. collect_lf() — all keys ────────────────────────────────────────────────
_section(f"5. collect_lf()  — all keys ({len(all_keys)} entities)")
REPS_ALL = 5
collect_lf(scan_with_key_filter(static_path, entity.key_cols, all_keys))  # warm

df_all = collect_lf(
    scan_with_key_filter(  # ensure bound before loop
        path=static_path, key_cols=entity.key_cols, key_values=all_keys
    )
)
t0 = time.perf_counter()
for _ in range(REPS_ALL):
    lf = scan_with_key_filter(
        path=static_path,
        key_cols=entity.key_cols,
        key_values=all_keys,
    )
    df_all = collect_lf(lf)
t1 = time.perf_counter()
print(
    f"collect(all) ×{REPS_ALL}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_ALL * 1000:.2f} ms/call)"
)
print(f"Rows returned  : {len(df_all)}")


# ── 6. scan_with_key_filter() + time_expr ────────────────────────────────────
_section("6. scan_with_key_filter() + time_expr  [with temporal predicate]")
# Only meaningful when the file has a time column; TehsilEntity is static-only
# so we demonstrate expression wiring even if the column doesn't exist.
cfg = SeasonConfig()
TimeFilter = dict[str, str | int | tuple[int, int] | None]
tf: TimeFilter = {"start": "2018-01-01", "end": "2023-12-31", "_is_year_col": False}
# Build a fake time expr; filter will reduce to empty if column absent.
time_expr = resolve_time_filter(tf, "obs_date", cfg)

t0 = time.perf_counter()
for _ in range(REPS):
    lf = scan_with_key_filter(
        path=static_path,
        key_cols=entity.key_cols,
        key_values=small_keys,
        time_expr=time_expr,
    )
t1 = time.perf_counter()
print(
    f"scan + time_expr (build) ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS * 1e6:.2f} µs/call)"
)


# ── 7. cached_read_schema() — cold vs cached ──────────────────────────────────
_section("7. cached_read_schema()  [cold vs functools.cache warm]")
from core_lens.utils.polars_utils import cached_read_schema  # noqa: E402
from core_lens.utils.polars_utils import _cached_schema_internal  # noqa: E402

# Warm the cache first (may already be warm from module import chain).
cached_read_schema(static_path)

REPS_CS = 100_000
t0 = time.perf_counter()
for _ in range(REPS_CS):
    cached_read_schema(static_path)
t1 = time.perf_counter()
print(
    f"cached_read_schema (warm) ×{REPS_CS}: {(t1 - t0) * 1000:.2f} ms total  "
    f"({(t1 - t0) / REPS_CS * 1e6:.3f} µs/call)"
)
print(f"Schema columns: {list(cached_read_schema(static_path).names())[:5]} ...")

# Cold path: clear the underlying cache, measure one miss.
_cached_schema_internal.cache_clear()
t0 = time.perf_counter()
cached_read_schema(static_path)
t1 = time.perf_counter()
print(f"cached_read_schema (cold) : {(t1 - t0) * 1000:.2f} ms  (Parquet footer read)")
print(f"Cache info after: {_cached_schema_internal.cache_info()}")


# ── Cleanup ───────────────────────────────────────────────────────────────────
AoI.deregister(MWSEntity)
print("\n✓ bench_polars_utils.py complete")
