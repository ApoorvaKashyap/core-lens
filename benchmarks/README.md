# core-lens Benchmarks

Focused Scalene profiling scripts — one per module/subsystem.

## Structure

```
benchmarks/
├── README.md
├── run_all.sh                  # Run all benchmarks in sequence
├── bench_aoi.py                # AoI init, register, named-boundary resolution, current_season/year
├── bench_entity.py             # BaseEntity._resolve, where, spatial_filter, spatial_join, caches  [NEW]
├── bench_spatial.py            # build_bbox_index, bbox_intersects_geometry, exact_spatial_filter, execute_spatial_join
├── bench_paths.py              # is_cloud_uri, resolve_fs_and_path, path_exists, join_uri  [NEW]
├── bench_view.py               # View.where, spatial_filter, between (all modes + errors), spatial_join, materialise
├── bench_result.py             # Result.derive, aggregate (all groupings + errors), with_geometry, df/gdf, stats/plot
├── bench_season.py             # resolve_time_filter, add_temporal_columns, SeasonConfig.season_for
├── bench_schema.py             # schema detection (_infer_* helpers, _is_year_col, _find_lon_companion, from_file)
├── bench_polars_utils.py       # scan_with_key_filter, collect_lf, cached_read_schema (CPU path)
└── bench_export.py             # parquet, json, csv, geoparquet, geojson, GeoJSONSeq, error paths
```

## Running

### Single benchmark

```bash
uv run scalene --html --outfile benchmarks/profiles/bench_spatial.html benchmarks/bench_spatial.py
```

### All benchmarks (sequential, JSON output via Scalene)

```bash
chmod +x benchmarks/run_all.sh
./benchmarks/run_all.sh
```

Profiles land in `benchmarks/profiles/`.

### Quick timing-only run (no Scalene overhead)

```bash
uv run python benchmarks/bench_spatial.py
```

## What to look for in Scalene output

| Signal | Meaning |
|---|---|
| High **CPU %** on line | Compute bottleneck — vectorise or push into Polars expr |
| High **memory** spike | Possible copy-on-write or large intermediate frame |
| `[native]` vs `[Python]` | Native = C ext (fast). Python = pure Python (optimise) |
| **GPU %** column | cudf path hit or not |

## Coverage map

| File | Functions / paths covered |
|---|---|
| `bench_aoi.py` | `AoI.register`, `AoI.__init__` (bbox/geometry/named/seasons), `__getattr__`, `current_season/year`, `registered_entities`, `deregister` |
| `bench_entity.py` | `_resolve` (local/cloud), `where` (single/composite/list/error), `spatial_filter` (bbox/geometry/area/error), `spatial_join`, `schema_profile` cache, `_index` cache, `geometry_lazy`, `_so_key`, `_entity_name`, `_cached_detect`, `EntityValidationError` |
| `bench_spatial.py` | `build_bbox_index`, `bbox_intersects_geometry`, `exact_spatial_filter` (centroid/area/all-India), `execute_spatial_join`, `_bbox_sidecar_path`, `_read_bbox_sidecar`, error path |
| `bench_view.py` | `where` (single/composite), `spatial_filter` (bbox), `between` (dates/season/CURRENT/year-range/errors), `spatial_join` (spec/double-error), `static`/`annual`/`sub_annual` materialisation |
| `bench_result.py` | `df`, `lazy`, `gdf` (WKB decode), `materialise`, `derive` (single/chain), `with_geometry` (noop/annual), `aggregate` (no-by/year/month/season/year_month/season_year/errors), `gdf` TypeError guard, `stats`/`plot` accessors |
| `bench_season.py` | `SeasonConfig.__post_init__` (default/custom), `season_for`, `resolve_time_filter` (date-range/kharif/rabi/year-range), `add_temporal_columns` (first/second noop call), Polars filter execution |
| `bench_schema.py` | `_read_schema`, `_infer_geometry_type`, `_infer_bbox_cols`, `_infer_time_col`, `_is_year_col_from_schema`, `_find_lon_companion`, `_require_cols` (ok/error), `detect` (static/full), `SchemaProfile.from_file`, `entity.schema_profile` (cold/warm) |
| `bench_polars_utils.py` | `_gpu_available`, `scan_with_key_filter` (build/+time_expr), `collect_lf` (1-key/small/all), `cached_read_schema` (warm/cold) |
| `bench_export.py` | `_get_duckdb_conn` (cold/cached), `parquet`/`json`/`csv`/`geoparquet`/`geojson` (small/all), `geojson(GeoJSONSeq)`, all error paths, format size comparison |

## Prerequisites

```bash
uv sync --group dev
```

Each script needs `DATA_ROOT` pointing at a directory with the expected entity Parquet files.
Edit the `DATA_ROOT` constant at the top of each file to match your local data path.
