# core-lens Usage Guide

`core-lens` is a high-performance Python library designed for querying, filtering, and manipulating large-scale geospatial datasets using deferred execution with Polars.

This guide covers the core concepts and primary workflows for using the library.

## 0. Installation

Install `core-lens` via `pip`. The library provides several optional dependency groups depending on your use case:

```bash
# Basic installation (Polars + core utilities)
pip install "core-lens[core]"

# Recommended for geospatial querying and exporting (adds DuckDB, GeoPandas, Shapely)
pip install "core-lens[spatial]"

# Full installation with visualization and statistical modeling (excludes GPU)
pip install "core-lens[full]"

# Full installation with Nvidia GPU acceleration via cuDF (Linux only)
pip install "core-lens[full-gpu]" --extra-index-url=https://pypi.nvidia.com
```

Though it is recommended that you use [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
mkdir core-test && cd core-test
uv init --bare
uv add core-lens[full]
```


## 1. Initialization and Registration

The primary entry point for querying data is the **Area of Interest (`AoI`)**. Before an `AoI` can be constructed, the geospatial entities (e.g., Microwatersheds, Tehsils, Villages) must be registered.

```python
from core_lens import AoI
from core_lens.entities import MWSEntity, TehsilEntity

# 1. Register available entities
AoI.register(MWSEntity)
AoI.register(TehsilEntity)

# Path to the root directory containing your parquet files
DATA_ROOT = "path/to/data"
```

## 2. Defining an Area of Interest (AoI)

An `AoI` is scoped to a specific spatial boundary. You must initialize it using **exactly one** of three boundary modes: a bounding box, a Shapely geometry, or named entity filters.

```python
import shapely.geometry as sgeom

# Mode A: Bounding Box (minx, miny, maxx, maxy)
aoi = AoI(DATA_ROOT, bbox=(76.0, 31.0, 78.0, 33.0))

# Mode B: Shapely Geometry
polygon = sgeom.box(76.0, 31.0, 78.0, 33.0)
aoi_geom = AoI(DATA_ROOT, geometry=polygon)

# Mode C: Named Entity Attributes / IDs
# Resolves the geometry of the matching entity (e.g., by district or specific ID)
aoi_named = AoI(DATA_ROOT, tehsil="Pangi", district="Chamba")
aoi_id = AoI(DATA_ROOT, mws_id="13_551")
```

## 3. Querying Entities (Lazy Views)

Accessing a registered entity on the `AoI` object (e.g., `aoi.mws`) returns a lazy **`View`**. A View builds up a query plan without executing any I/O until you explicitly materialize it.

### Filtering Views

Views can be chained with different types of filters:

```python
# Base view: All MWS falling within the AoI
mws_view = aoi.mws

# 1. Attribute Filtering
filtered_view = mws_view.where(state="Himachal Pradesh")

# 2. Spatial Filtering (Narrowing down within the AoI)
spatial_view = mws_view.spatial_filter(bbox=(76.5, 31.5, 77.5, 32.5))

# 3. Temporal Filtering (Filters sub-annual/annual data)
from core_lens.base.view import Season

temporal_view = mws_view.between(
    year=(2020, 2022),       # Inclusive year range
    season=Season.KHARIF     # Filter to specific season
)
```

### Cross-Entity Spatial Joins

You can define a spatial join to enrich one entity with aggregated data from another intersecting entity. This is evaluated lazily. This is still experimental.

```python
# Join MWS with Tehsil entity to calculate the sum of tehsil areas per MWS
joined_view = mws_view.spatial_join(
    other=TehsilEntity(data_root=DATA_ROOT),
    agg={"Shape_Area": "sum"}
)
```

## 4. Materialization

Once your query is fully constructed, you materialize the `View` into a concrete `Result`. You must specify the temporal grain (`static`, `annual`, or `sub_annual`) you wish to load.

```python
# Materialize the static dataset (one row per entity)
result_static = filtered_view.static.materialise()

# Materialize annual time-series data
result_annual = filtered_view.annual.materialise()

# Materialize highly granular sub-annual time-series data
result_sub_annual = temporal_view.sub_annual.materialise()
```

## 5. Working with Results

A `Result` object provides access to the materialized data. `core-lens` uses zero-copy Polars DataFrames by default for speed, but integrates with GeoPandas when spatial manipulation is required.

```python
# 1. Zero-copy access (Instantaneous)
pl_df = result_static.df()          # Polars DataFrame
pl_lazy = result_static.lazy()      # Polars LazyFrame

# 2. Decode WKB geometries to Shapely (CPU intensive)
gdf = result_static.gdf()           # GeoPandas GeoDataFrame

# 3. Data Transformations (Returns a new Result)
# Derive new columns using Polars expressions
import polars as pl
result_derived = result_annual.derive("yield_per_ha", pl.col("yield") / pl.col("area_in_ha"))

# 4. Fast aggregations (Returns a new Result)
# Aggregate by a key column using Polars expressions
result_agg = result_derived.aggregate(
    pl.col("yield_per_ha").mean().alias("avg_yield_per_ha"),
    by="year"
)
```

## 6. Exporting Data

For interoperability, results can be exported to various standard formats. Avoid exporting large, country-wide datasets to GeoJSON due to significant file size bloat.

```python
from core_lens.export import formats as export

# Export to GeoParquet (Fast, efficient spatial format)
export.geoparquet(result_static, "output/mws.geoparquet")

# Export to GeoJSON (Slower, text-based)
export.geojson(result_static, "output/mws.geojson")

# Export without geometries (Uses Polars directly)
export.parquet(result_static, "output/mws.parquet")
export.csv(result_static, "output/mws.csv")
```

## Advanced Configuration

### Custom Seasons
By default, `core-lens` uses Indian agricultural seasons. You can inject a custom `SeasonConfig` at AoI initialization if you need to map time series data to different date ranges:

```python
from core_lens.aoi import SeasonConfig

custom_seasons = SeasonConfig(
    kharif=("06-01", "10-15"),
    rabi=("10-16", "02-28"),
    zaid=("03-01", "05-31")
)

aoi = AoI(DATA_ROOT, bbox=(...), seasons=custom_seasons)
```
