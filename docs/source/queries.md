# Example Queries

## Time & Seasonal Filtering

Filter time-series data using exact dates or agronomic seasons:

```python
# Exact date range
annual_data = aoi.mws.between("2010-01-01", "2023-12-31").annual

# Season-based filtering (defaults to Kharif, Rabi, Zaid)
kharif_2020 = aoi.mws.between(season="kharif", year=2020).fortnightly

# Current season based on today's date
current = aoi.mws.between(season="current").fortnightly
```

## Aggregations & Derived Columns

Compute aggregates or derive new columns using Polars syntax:

```python
import polars as pl

# 1. Derive new columns (works on any Result)
res_annual = aoi.mws.annual
derived_res = res_annual.derive(
    "drought_flag",
    pl.when(pl.col("rainfall") < 500).then(1).otherwise(0)
)

# 2. Aggregate (temporal grouping requires fortnightly resolution)
res_fortnightly = aoi.mws.fortnightly
monthly_avg = res_fortnightly.aggregate(pl.mean("ndvi"), by="month")
seasonal_avg = res_fortnightly.aggregate(pl.mean("ndvi"), by="season_year")
```

## Cross-Entity Spatial Joins

Spatially join data from two different entities (e.g. Microwatersheds and Forests):

```python
# Register custom entity
# AoI.register(ForestEntity)

# Join forest metrics onto MWS geometries
joined_view = aoi.mws.spatial_join(
    aoi.forest,
    agg={
        "tree_cover": "area",
        "canopy_density": "mean"
    }
)

# Materialise (Note: spatial join materialisation raises NotImplementedError in this release)
# res = joined_view.annual
```

> [!NOTE]
> Cross-entity spatial join execution (materialisation) is currently under development and will be added in a subsequent release. Calling materialisation methods (like `.annual`, `.static`, or `.fortnightly`) on a joined view will raise a `NotImplementedError` in the current version.
