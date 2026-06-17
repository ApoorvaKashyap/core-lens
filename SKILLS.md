---
name: core-lens
description: >
  A Python library for querying, analysing, and visualising geospatial Earth science data.
  Provides a unified interface over microwatersheds (MWS), administrative boundaries, and pluggable domain entities.
  Use this skill to understand how to interact with AoI (Area of Interest), Views, Results, and Statistical analysis.
---

# Core-Lens Library Skill

`core-lens` is a geospatial data manipulation library designed for Earth science primitives (like microwatersheds). It uses Polars for lazy Parquet I/O and provides spatial joins, aggregations, time-series support, and statistical analysis.

## Core Concepts

The library has three primary layers:
1. **AoI Layer (Area of Interest)**: Resolves boundaries, filters entities spatially, and handles time filtering.
2. **Result Layer**: Materialized data with compute, statistical analysis, and plotting methods.
3. **Export Layer**: Standalone functions to export data.

## 1. Initialisation and Registration
All entities must be registered explicitly before use.

```python
from core_lens import AoI, SeasonConfig
from core_lens.entities import MWSEntity, TehsilEntity, DistrictEntity

# Register entities
AoI.register(MWSEntity)
AoI.register(TehsilEntity)
AoI.register(DistrictEntity)

# Initialize an AoI scoped to a specific geometry/entity
aoi = AoI(data_root="data/", tehsil="Pangi", district="Chamba")
# Or by bounding box: AoI("data/", bbox=(minx, miny, maxx, maxy))
```

## 2. Views and Materialization
Once an AoI is created, you can access the scoped entities (e.g. `aoi.mws`, `aoi.village`). These are `View` objects.

```python
view = aoi.mws

# Time filtering
view.between("2010-01-01", "2023-12-31")
view.between(season="kharif", year=2020)

# Materialise into a Result object
static_result = view.static         # has geometry
annual_result = view.annual         # no geometry
fortnightly_result = view.fortnightly # no geometry
```

## 3. Operations on Results (Compute & Aggregation)
The `Result` object is where all data manipulation happens.

```python
# Derive new columns
res = annual_result.derive("water_shortage", pl.col("rainfall") - pl.col("storage"))

# Aggregation
res.aggregate(pl.mean("ndvi"), by="year")
res.aggregate(pl.mean("ndvi"), by="season")
```

## 4. Cross-Entity Spatial Joins
Spatial joins can be performed across different entities. Aggregation rules apply based on geometries.

```python
aoi.mws.spatial_join(
    aoi.forest,
    agg={
        "tree_cover": "area",
        "canopy_density": "mean"
    }
).annual.between(2010, 2023)
```

## 5. Statistical Analysis
All statistical operations are available under the `result.stats` namespace and return a new `Result`. The computed results live in `.data` while the parameters used are in `.metadata`.

### Describe
Computes descriptive statistics (mean, std, min, median, max) per column or entity.
```python
result.stats.describe(columns=["ndvi", "rainfall"])
result.stats.describe(by="entity")  # per entity breakdown
```

### Correlate
Calculates correlation between columns either across entities or across time. Supported methods: `"pearson"`, `"spearman"`, `"kendall"`.
```python
result.stats.correlate(
    columns=["ndvi", "rainfall", "temperature"],
    method="pearson",
    across="entity"
)
```

### Hypothesis Testing
Supports testing differences between groups, time periods, or against a baseline. Supported methods: `"t-test"`, `"mann-whitney"` (default), `"wilcoxon"`, `"ks"`, `"chi-square"`. Auto-selects method via Shapiro-Wilk if not specified.
```python
# Group-based
result.stats.test(column="cropping_intensity", groups="temperature_zone", method="mann-whitney")
# Period-based
result.stats.test(column="ndvi", periods=[(2010, 2015), (2016, 2023)], method="t-test")
```

### Change Detection
Measures structural or absolute change over time. Supported methods: `"absolute"` (default), `"percentage"`, `"trend"`.
```python
result.stats.change(column="tree_cover", from_period=2018, to_period=2023, method="percentage")
result.stats.change(column="ndvi", from_period=2010, to_period=2023, method="trend")
```

### Anomaly Detection
Identifies anomalies cross-sectionally (vs other entities) or over time.
- **Cross-sectional methods:** `"zscore"`, `"iqr"`, `"percentile"`, `"threshold"`.
- **Time-series methods:** `"stl"`, `"cusum"`, `"mad"`.
```python
# Cross-sectional
result.stats.anomaly(column="ndvi", mode="cross_sectional", method="zscore", baseline=(2010, 2020))
# Time-series
result.stats.anomaly(column="ndvi", mode="timeseries", method="stl", baseline=(2010, 2018))
```

### Similarity
Finds similar entities based on multidimensional distance. Supported methods: `"euclidean"`, `"cosine"`, `"mahalanobis"`, `"manhattan"`.
```python
result.stats.similarity(
    target="13_551",
    columns={
        "rainfall": ("annual", {"year": 2018}),
        "ndvi": ("fortnightly", {"season": "kharif", "year": 2020})
    },
    method="euclidean",
    top_n=10
)
```

## 6. Plotting
Plot methods are explicitly available under `result.plot` and return plot objects (not immediately rendered). Subplotting can be done on `"year"`, `"month"`, `"season"`, `"season_year"`. Maps use Lonboard (deck.gl) and charts use Plotly. By default, charts render the top 10 entities.

```python
# Map plotting (Lonboard)
result.plot.choropleth("ndvi", subplot_on="season_year")

# Charts (Plotly)
result.plot.timeseries(x="year", y="ndvi", top_n=10, aggregate=False)
result.plot.scatter(x="rainfall", y="ndvi", top_n=10)
result.plot.distribution(x="ndvi", top_n=10)
result.plot.correlation(columns=["ndvi", "rainfall", "temperature"], top_n=10)
result.plot.heatmap(x="year", y="month", value="ndvi", top_n=10)
result.plot.matrix()
```

## 7. Exporting
Export methods are standalone functions.

```python
from core_lens.export import geojson, geoparquet, parquet, json

geoparquet(static_result, "output.geoparquet")
parquet(annual_result, "output.parquet")
```

## Creating Custom Entities (Plugins)
You can extend core-lens with custom entities by subclassing `BaseEntity`.

```python
from core_lens.base import BaseEntity

class ForestEntity(BaseEntity):
    key_cols      = ["forest_patch_id"]
    geometry_col  = "geometry"
    static_path   = "data/forest/static.geoparquet"
    annual_path   = "data/forest/annual.parquet"

AoI.register(ForestEntity)
```

## 8. Entity Mapping and Linking

- **Implicit Spatial Linking**: You do not need explicit mapping files or manual join operations to link a specific location (e.g., "Shiroor, Dharwad, Karnataka") to its corresponding `mws_id`. This is handled seamlessly by the library's Area of Interest (`AoI`) constructor. For instance:
  ```python
  aoi = AoI(data_root="data/", state="Karnataka", district="Dharwad", village="Shiroor")
  ```
  This single command resolves the boundary for Shiroor and scopes all registered entities. The `aoi.mws` view will automatically contain only the `mws_id`s that fall within this resolved geometry.

- **Cropping Intensity Column**: In the temporal annual data, the cropping intensity metric is typically available as the `ci_cropping_intensity` column.

## 9. Execution Environment Context

- **Loading Data**: There is no explicit `AoI.load_data()` method. The `AoI(data_root="...")` constructor acts as the main entry point and defines the directory scope. Data reading happens lazily using Polars `scan_parquet()` when you explicitly call materialization methods like `.static`, `.annual`, or `.fortnightly` on a `View`.
- **Spatial and Temporal Relationship**: To conserve memory, temporal views (`.annual` and `.fortnightly`) drop geometry columns by default. To establish the spatial relationship between the static boundaries (`mws.parquet`) and the temporal time-series (`annual.parquet`), you use the `Result.with_geometry()` method. This triggers an internal join on the entity's primary key (e.g., `mws_id`) bringing the geometry back into the DataFrame for spatial operations and plotting.

- **Expected Directory Structure**: When initializing `AoI(data_root="data/")`, the library expects a specific folder structure inside the `data_root` for each registered entity. For example, the `MWSEntity` expects:
  ```text
  data/
  └── mws/
      ├── static/
      │   └── mws.parquet        ← GeoParquet with boundaries
      ├── annual/                ← Directory of Parquets (often partitioned by basin)
      └── fortnightly/           ← Directory of Parquets (partitioned by year/basin)
  ```
  *(Note: The exact paths for these files/directories are defined within the `BaseEntity` subclasses, e.g., `MWSEntity.static_path`)*

## 10. Recommended Workflow

When building pipelines with `core-lens`, you should generally follow this sequence of operations. Because `core-lens` is lazy by default, operations before Step 5 do not load the actual data into memory.

**1. Registration:** Register the specific entities you plan to query.
```python
from core_lens import AoI
from core_lens.entities import MWSEntity
AoI.register(MWSEntity)
```

**2. AoI Initialization (Scoping):** Construct the geographic scope using a geometry, bounding box, or administrative filters.
```python
aoi = AoI(data_root="data/", district="Barmer", state="Rajasthan")
```

**3. Entity & Spatial Filtering:** Access your target entity view (e.g., `aoi.mws`) and apply any further attribute or spatial constraints.
```python
view = aoi.mws.where(ba_name="Barmer")
```

**4. Temporal Filtering:** Define the time range or season of interest using `.between()`.
```python
temporal_view = view.between(season="kharif", year=2021)
```

**5. Materialization:** Trigger the actual Parquet I/O to pull the data into memory by accessing `.static`, `.annual`, or `.fortnightly`.
```python
result = temporal_view.fortnightly
```

**6. Compute & Analysis:** Chain computations, derive metrics, or run statistical tests on the returned `Result`.
```python
import polars as pl
analyzed_result = (
    result
    .derive("et_ratio", (pl.col("df_et") / pl.col("df_precipitation")))
    .stats.anomaly(column="et_ratio", mode="timeseries", method="stl", baseline=(2015, 2020))
)
```

**7. Presentation & Export:** Reattach geometries (if working with temporal data) to plot maps or export the data.
```python
from core_lens.export import geoparquet

# Reattach geometry for spatial operations
geo_result = analyzed_result.with_geometry()

# Plot on a map
geo_result.plot.choropleth("anomaly_score")

# Export to GeoParquet
geoparquet(geo_result, "output_anomalies.geoparquet")
```
