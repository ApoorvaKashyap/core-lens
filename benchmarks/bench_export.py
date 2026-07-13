"""bench_export.py — Benchmark for core_lens.export.formats
=========================================================
Targets:
  - export.parquet()            Polars write_parquet (no geometry)
  - export.json()               Polars write_json (no geometry)
  - export.csv()                Polars write_csv (no geometry)
  - export.geoparquet()         DuckDB spatial COPY → Parquet (WKB → geometry)
  - export.geojson()            DuckDB spatial COPY → GeoJSON (WKB → GeoJSON)
  - _get_duckdb_conn()          singleton connection + INSTALL/LOAD spatial

Scalene focuses on:
  - DuckDB spatial extension load overhead (first call)
  - write_parquet / write_json / write_csv serialisation speed
  - DuckDB ST_GeomFromWKB + COPY overhead vs Polars native writers
  - Memory copies between Polars DataFrame → DuckDB

Usage:
    uv run scalene run --html --outfile benchmarks/profiles/bench_export.html benchmarks/bench_export.py
    uv run python benchmarks/bench_export.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path


from core_lens import AoI
from core_lens.entities.mws import MWSEntity
from core_lens.export import csv, geoparquet, geojson, json, parquet
from core_lens.export.formats import _get_duckdb_conn

# ── Config ────────────────────────────────────────────────────────────────────
DATA_ROOT = "data/"
INDIA_BBOX: tuple[float, float, float, float] = (68.0, 6.0, 98.0, 38.0)
SMALL_BBOX: tuple[float, float, float, float] = (86.0, 21.0, 87.5, 22.0)

AoI.register(MWSEntity)

# Materialise two result sizes for comparison.
aoi_all = AoI(DATA_ROOT, bbox=INDIA_BBOX)
aoi_small = AoI(DATA_ROOT, bbox=SMALL_BBOX)

result_all = aoi_all.mws.static  # full dataset — geometry present
result_small = aoi_small.mws.static  # small subset

print(f"All   result : {result_all.data.shape}")
print(f"Small result : {result_small.data.shape}")


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)

    # ── 1. DuckDB singleton init (first call installs spatial extension) ──────
    _section("1. _get_duckdb_conn()  [INSTALL spatial; LOAD spatial — cold]")
    t0 = time.perf_counter()
    conn = _get_duckdb_conn()
    t1 = time.perf_counter()
    print(f"DuckDB init (cold)  : {(t1 - t0) * 1000:.2f} ms")

    REPS = 10_000
    t0 = time.perf_counter()
    for _ in range(REPS):
        _ = _get_duckdb_conn()
    t1 = time.perf_counter()
    print(
        f"_get_duckdb_conn ×{REPS}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS * 1e6:.3f} µs/call)  (cached)"
    )

    # ── 2. export.parquet() — no geometry (strip geom col first)
    _section("2. export.parquet()  [Polars write_parquet, no geometry]")

    # Strip geometry (and nested bbox) to pass has_geometry=False and allow CSV export.
    geom_col = result_small.entity.geometry_col
    drop_cols = [geom_col]
    if "bbox" in result_small.data.columns:
        drop_cols.append("bbox")
    df_no_geom = result_small.data.drop(drop_cols)
    result_no_geom = result_small._replace(data=df_no_geom, has_geometry=False)

    REPS_IO = 5
    out = tmp / "bench.parquet"
    t0 = time.perf_counter()
    for _ in range(REPS_IO):
        parquet(result_no_geom, out)
    t1 = time.perf_counter()
    sz = out.stat().st_size
    print(
        f"parquet (small) ×{REPS_IO}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS_IO * 1000:.2f} ms/call)  → {sz // 1024} KB"
    )

    # All dataset.
    df_all_no_geom = result_all.data.drop(geom_col)
    result_all_no_geom = result_all._replace(data=df_all_no_geom, has_geometry=False)
    out_all = tmp / "bench_all.parquet"
    t0 = time.perf_counter()
    for _ in range(REPS_IO):
        parquet(result_all_no_geom, out_all)
    t1 = time.perf_counter()
    sz_all = out_all.stat().st_size
    print(
        f"parquet (all)   ×{REPS_IO}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS_IO * 1000:.2f} ms/call)  → {sz_all // 1024} KB"
    )

    # ── 3. export.json() ──────────────────────────────────────────────────────
    _section("3. export.json()  [Polars write_json]")
    out_json = tmp / "bench.json"
    t0 = time.perf_counter()
    for _ in range(REPS_IO):
        json(result_no_geom, out_json)
    t1 = time.perf_counter()
    sz = out_json.stat().st_size
    print(
        f"json (small) ×{REPS_IO}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS_IO * 1000:.2f} ms/call)  → {sz // 1024} KB"
    )

    # ── 4. export.csv() ───────────────────────────────────────────────────────
    _section("4. export.csv()  [Polars write_csv]")
    out_csv = tmp / "bench.csv"
    t0 = time.perf_counter()
    for _ in range(REPS_IO):
        csv(result_no_geom, out_csv)
    t1 = time.perf_counter()
    sz = out_csv.stat().st_size
    print(
        f"csv (small)  ×{REPS_IO}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS_IO * 1000:.2f} ms/call)  → {sz // 1024} KB"
    )

    # ── 5. export.geoparquet() — DuckDB spatial (small) ──────────────────────
    _section("5. export.geoparquet()  [DuckDB ST_GeomFromWKB + COPY]")
    out_gpq = tmp / "bench.geoparquet"
    t0 = time.perf_counter()
    for _ in range(REPS_IO):
        geoparquet(result_small, out_gpq)
    t1 = time.perf_counter()
    sz = out_gpq.stat().st_size
    print(
        f"geoparquet (small) ×{REPS_IO}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS_IO * 1000:.2f} ms/call)  → {sz // 1024} KB"
    )

    out_gpq_all = tmp / "bench_all.geoparquet"
    t0 = time.perf_counter()
    geoparquet(result_all, out_gpq_all)
    t1 = time.perf_counter()
    sz_all = out_gpq_all.stat().st_size
    print(
        f"geoparquet (all)   ×1: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) * 1000:.2f} ms/call)  → {sz_all // 1024} KB"
    )

    # ── 6. export.geojson() — DuckDB spatial ──────────────────────────────────
    _section("6. export.geojson()  [DuckDB ST_GeomFromWKB + COPY GDAL GeoJSON]")
    out_gj = tmp / "bench.geojson"
    t0 = time.perf_counter()
    for _ in range(REPS_IO):
        geojson(result_small, out_gj)
    t1 = time.perf_counter()
    sz = out_gj.stat().st_size
    print(
        f"geojson (small) ×{REPS_IO}: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) / REPS_IO * 1000:.2f} ms/call)  → {sz // 1024} KB"
    )

    out_gj_all = tmp / "bench_all.geojson"
    t0 = time.perf_counter()
    geojson(result_all, out_gj_all)
    t1 = time.perf_counter()
    sz_all = out_gj_all.stat().st_size
    print(
        f"geojson (all)   ×1: {(t1 - t0) * 1000:.2f} ms total  "
        f"({(t1 - t0) * 1000:.2f} ms/call)  → {sz_all // 1024} KB"
    )

    # ── 7. Head-to-head: geoparquet vs geojson ────────────────────────────────
    _section("7. Head-to-head: geoparquet vs geojson (all dataset)")
    out_a = tmp / "hth.geoparquet"
    out_b = tmp / "hth.geojson"

    t0 = time.perf_counter()
    geoparquet(result_all, out_a)
    t1 = time.perf_counter()
    t_gpq = (t1 - t0) * 1000

    t0 = time.perf_counter()
    geojson(result_all, out_b)
    t1 = time.perf_counter()
    t_gj = (t1 - t0) * 1000

    print(f"geoparquet : {t_gpq:.2f} ms   ({out_a.stat().st_size // 1024} KB)")
    print(f"geojson    : {t_gj:.2f} ms   ({out_b.stat().st_size // 1024} KB)")
    print(
        f"Ratio      : geojson is {t_gj / t_gpq:.2f}× {'slower' if t_gj > t_gpq else 'faster'}"
    )


# ── Cleanup ───────────────────────────────────────────────────────────────────
AoI.deregister(MWSEntity)
print("\n✓ bench_export.py complete")
