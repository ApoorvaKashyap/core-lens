---
name: core-lens
description: >
  A Python library for querying, analysing, and visualising geospatial Earth science data.
  Provides a unified interface over microwatersheds (MWS), administrative boundaries, and pluggable domain entities.
  Use this skill to understand how to interact with AoI (Area of Interest), Views, Results, and Statistical analysis.
---

# Agent Instructions: Core-Lens Library

You are an AI Agent tasked with writing or debugging code using the `core-lens` library. `core-lens` is a high-performance Python library designed for querying, filtering, and manipulating large-scale geospatial datasets using deferred execution with **Polars**.

**Read these instructions carefully before writing any code.**

## 1. Core Architectural Constraints

- **Area of Interest (AoI) is the Primary Entry Point**: Never load parquet files directly. Always create an `AoI` object to define the geographical scope and directory context.
- **Lazy Evaluation**: `core-lens` is lazy by default. Accessing an entity like `aoi.mws` creates a `View`. No data is loaded into memory until you call a materialisation method (`.static`, `.annual`, or `.sub_annual`).
- **Polars Over Pandas**: `core-lens` uses Polars for zero-copy operations. Only convert to GeoPandas (`.gdf()`) when strictly necessary for heavy geometric processing.
- **Separation of Concerns**:
  - `AoI`: Filters boundaries, resolves geography, scopes entities, and handles time constraints.
  - `Result`: Produced by materializing a view. Handles all computation, spatial joins, aggregations, stats, and plotting.
  - `Export`: Separate namespace (`from core_lens.export import geoparquet`).

## 1.5 Logging and Tracing

`core_lens` uses Loguru and is disabled by default. If you need to debug spatial resolution issues or trace validation errors, enable the internal logger:

```python
import sys
from loguru import logger

logger.enable("core_lens")
logger.add(sys.stdout, level="DEBUG")
```

## 2. Initialization and Setup

Before querying, you MUST register the specific entities you plan to use.

```python
from core_lens import AoI, SeasonConfig
from core_lens.entities import MWSEntity, TehsilEntity, DistrictEntity

# 1. Register entities FIRST
AoI.register(MWSEntity)
AoI.register(DistrictEntity)
# ... register others as needed

# 2. Scoping the AoI (Choose ONE mode)
DATA_ROOT = "data/"
# Mode A: By Administrative Hierarchy
aoi = AoI(DATA_ROOT, tehsil="Pangi", district="Chamba")
# Mode B: By Bounding Box
aoi = AoI(DATA_ROOT, bbox=(76.0, 31.0, 78.0, 33.0))
# Mode C: By specific Entity ID
aoi = AoI(DATA_ROOT, mws_id="13_551")

# Optional: Override default seasons (kharif, rabi, zaid)
custom_seasons = SeasonConfig(kharif=("06-01", "10-15"), rabi=("10-16", "02-28"), zaid=("03-01", "05-31"))
aoi_custom = AoI(DATA_ROOT, tehsil="Pangi", seasons=custom_seasons)
```

## 3. Querying & Filtering (Views)

Entities on the `AoI` object are accessed as **Views**. Views are lazy and can be chained.

```python
# Access the view
mws_view = aoi.mws

# Attribute Filtering
filtered = mws_view.where(state="Himachal Pradesh")

# Spatial Filtering (inside the AoI)
spatial_filtered = mws_view.spatial_filter(bbox=(76.5, 31.5, 77.5, 32.5))

# Temporal Filtering
# Note: You can filter by exact years, a range of years, or predefined seasons.
from core_lens.base.view import Season
temporal_view = mws_view.between(year=(2020, 2022), season=Season.KHARIF)
current_view = mws_view.between(season=Season.CURRENT)
```

## 4. Materialization (Loading Data)

You must materialize the view into a `Result` to perform computations.
Choose the resolution based on what you need:
- `.static`: One row per entity, includes `geometry`.
- `.annual`: Time-series, aggregated annually. (NO geometry by default).
- `.sub_annual`: High-frequency time-series. (NO geometry by default).

```python
static_result = filtered.static.materialise()
annual_result = temporal_view.annual.materialise()
```
*Note: Depending on library version, you may only need `.static` instead of `.static.materialise()`. If `.static` returns a Result directly, use that.*

```python
# To access underlying data structures:
df = static_result.df()        # Polars DataFrame (Zero-copy)
lazy_df = static_result.lazy() # Polars LazyFrame
gdf = static_result.gdf()      # GeoPandas GeoDataFrame (Heavy!)
```

## 5. Computation and Data Manipulation (Result API)

All data manipulation happens on the `Result` object.

```python
import polars as pl

# Derive new columns
res = annual_result.derive("water_shortage", pl.col("rainfall") - pl.col("storage"))

# Aggregation (Supported 'by' keys: "year", "month", "season", "season_year")
agg_res = res.aggregate(pl.mean("ndvi").alias("avg_ndvi"), by="year")

# Reattaching Geometry
# Time-series results drop geometry to save memory. Reattach it if you need to map them:
geo_result = agg_res.with_geometry()
```

## 6. Spatial Joins

Spatial joins are lazy and enrich a primary entity with aggregated data from a secondary entity.

```python
# Calculate the total forest tree_cover inside each MWS
joined_view = aoi.mws.spatial_join(
    aoi.forest, agg={"tree_cover": "sum", "canopy_density": "mean"}
)
# Resulting columns are prefixed, e.g., 'forest_tree_cover', 'forest_canopy_density'
```

## 7. Statistical Analysis (`Result.stats`)

All stats methods are available under `result.stats` and return a NEW `Result`.
- Computed data is in `result.data` (which is a Polars DataFrame)
- Parameters and testing metrics are stored in `result.metadata`

```python
# Descriptive stats
result.stats.describe(columns=["ndvi", "rainfall"])

# Correlation (pearson, spearman, kendall)
from core_lens.base.namespaces.stats import CorrelateMethod
result.stats.correlate(columns=["ndvi", "rainfall"], method=CorrelateMethod.PEARSON, across="entity")

# Hypothesis Testing (t-test, mann-whitney, wilcoxon, ks, chi-square)
from core_lens.base.namespaces.stats import TestMethod
result.stats.test(column="cropping_intensity", groups="temperature_zone", method=TestMethod.MANN_WHITNEY)

# Change Detection (absolute, percentage, trend)
from core_lens.base.namespaces.stats import ChangeMethod
result.stats.change(column="tree_cover", from_period=2018, to_period=2023, method=ChangeMethod.PERCENTAGE)

# Anomaly Detection
from core_lens.base.namespaces.stats import AnomalyTsMethod, AnomalyCrossMethod
# Mode 1: Cross-sectional (zscore, iqr, percentile, threshold)
result.stats.anomaly(column="ndvi", mode="cross_sectional", method=AnomalyCrossMethod.ZSCORE, baseline=(2010, 2020))
# Mode 2: Time-series (stl, cusum, mad)
result.stats.anomaly(column="ndvi", mode="timeseries", method=AnomalyTsMethod.STL, baseline=(2010, 2018))

# Similarity Matching (euclidean, cosine, mahalanobis, manhattan)
from core_lens.base.namespaces.stats import SimilarityMethod
result.stats.similarity(
    target="13_551",
    columns={"rainfall": ("annual", {"year": 2018})},
    method=SimilarityMethod.EUCLIDEAN,
    top_n=10
)
```

## 8. Plotting and Visualization (`Result.plot`)

Plots are explicitly accessed via the `result.plot` namespace. Maps use Lonboard (requires geometry); charts use Plotly. By default, charts render the top 10 entities.

```python
# Map plotting (Requires result.with_geometry() to have been called for time-series)
result.plot.choropleth("ndvi", subplot_on="season_year")

# Charts
result.plot.timeseries(x="year", y="ndvi", top_n=10)
result.plot.scatter(x="rainfall", y="ndvi", top_n=10)
result.plot.distribution(x="ndvi", top_n=10)
result.plot.correlation(columns=["ndvi", "rainfall"], top_n=10)
result.plot.heatmap(x="year", y="month", value="ndvi", top_n=10)
result.plot.matrix()

# Plotting the AoI boundary itself
aoi.plot()
aoi.plot(overlay=result)
```

## 9. Exporting Results

Exporting uses standalone functions. Do not export massive datasets (like country-wide `mws`) to GeoJSON as it will bloat the file size. Use GeoParquet instead.

```python
from core_lens.export import geoparquet, parquet, geojson

# For results with geometry
geoparquet(result, "output.geoparquet")

# For tabular results without geometry
parquet(result, "output.parquet")
```

## 10. Plugin System (Custom Entities)

You can extend core-lens with custom entities by subclassing `BaseEntity`.

```python
from core_lens.base import BaseEntity

class CustomEntity(BaseEntity):
    @property
    def key_cols(self) -> list[str]:
        return ["custom_id"]

    @property
    def geometry_col(self) -> str:
        return "geometry"

    @property
    def static_path(self) -> str:
        return "custom/static.geoparquet"

    @property
    def annual_path(self) -> str | None:
        return "custom/annual.parquet"

AoI.register(CustomEntity)
```

## Agent Reminders & Pitfalls to Avoid
- **DO NOT** attempt to read parquet files directly using `pandas.read_parquet` or `polars.read_parquet`. Always construct an `AoI` and use the library's materialized `Result` output.
- **DO NOT** assume `annual` or `sub_annual` results contain geometry. You must call `.with_geometry()` to reattach it.
- **DO NOT** forget to import and register entities before initializing the `AoI`.
- **DO** use Polars expressions (`import polars as pl`) inside `.derive()` and `.aggregate()` methods for maximum performance.
- **DO** always read parameters/testing metrics from `result.metadata` and underlying stats from `result.data` after running any methods in `result.stats`.
