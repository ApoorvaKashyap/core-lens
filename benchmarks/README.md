# core-lens Benchmarks

Focused Scalene profiling scripts — one per module/subsystem.

## Structure

```
benchmarks/
├── README.md
├── run_all.sh                  # Run all benchmarks in sequence
├── bench_aoi.py                # AoI init, register, named-boundary resolution
├── bench_spatial.py            # build_bbox_index, bbox_intersects_geometry, exact_spatial_filter
├── bench_view.py               # View.where, spatial_filter, between, materialise (static/annual/fortnightly)
├── bench_result.py             # Result.derive, aggregate, with_geometry, df/gdf
├── bench_season.py             # resolve_time_filter, add_temporal_columns, SeasonConfig.season_for
├── bench_schema.py             # schema detection (detect, _infer_* helpers)
├── bench_polars_utils.py       # scan_with_key_filter, collect_lf (CPU path)
└── bench_export.py             # parquet, json, csv, geoparquet, geojson
```

## Running

### Single benchmark

```bash
uv run scalene --html --outfile benchmarks/profiles/bench_spatial.html benchmarks/bench_spatial.py
```

### All benchmarks (sequential, HTML output)

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

## Prerequisites

```bash
uv sync --group dev
```

Each script needs `DATA_ROOT` pointing at a directory with the expected entity Parquet files.
Edit the `DATA_ROOT` constant at the top of each file to match your local data path.
