# Example Queries

## Time & Seasonal Filtering

Filter time-series data using exact dates or agronomic seasons:

```python
# Exact date range
annual_data = aoi.mws.between("2010-01-01", "2023-12-31").annual

# Season-based filtering (defaults to Kharif, Rabi, Zaid)
from core_lens.base.view import Season

kharif_2020 = aoi.mws.between(season=Season.KHARIF, year=2020).sub_annual

# Current season based on today's date
current = aoi.mws.between(season=Season.CURRENT).sub_annual
```

## Aggregations & Derived Columns

Compute aggregates or derive new columns using Polars syntax:

```python
import polars as pl

# 1. Derive new columns (works on any Result)
res_annual = aoi.mws.annual
derived_res = res_annual.derive(
    "drought_flag", pl.when(pl.col("rainfall") < 500).then(1).otherwise(0)
)

# 2. Aggregate (temporal grouping like "month" requires sub-annual, but "year" works on annual too)
res_sub_annual = aoi.mws.sub_annual
monthly_avg = res_sub_annual.aggregate(pl.mean("ndvi"), by="month")
seasonal_avg = res_sub_annual.aggregate(pl.mean("ndvi"), by="season_year")
```

## Cross-Entity Spatial Joins

Spatially join data from two different entities (e.g. Microwatersheds and Forests):

```python
# Register custom entity
# AoI.register(ForestEntity)

# Join forest metrics onto MWS geometries
joined_view = aoi.mws.spatial_join(
    aoi.forest, agg={"tree_cover": "area", "canopy_density": "mean"}
)

# Materialise the joined view
res = joined_view.annual
```

```{note}
Cross-entity spatial join execution (materialisation) is fully supported. Calling materialisation methods (like `.annual`, `.static`, or `.sub_annual`) on a joined view will evaluate the deferred join and execute it.
```
