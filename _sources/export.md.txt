# Data Export

CoreLens provides a unified `core_lens.export` module for saving `Result` objects to various file formats. Standard tabular formats use Polars for high-speed writes, while geospatial formats leverage DuckDB's spatial extensions for robust geometry encoding.

## Standard Formats

Export non-spatial results to standard formats. Note that these formats cannot encode geometry columns; if your `Result` has geometry, these functions will raise a `TypeError`.

```python
from core_lens.export import parquet, csv, json

res = aoi.mws.annual

# Export to Parquet with ZSTD compression
parquet(res, "output.parquet", compression="zstd")

# Export to CSV
csv(res, "output.csv", separator=",")

# Export to JSON
json(res, "output.json", pretty=True)
```

## Geospatial Formats

Export spatial data to GeoParquet and GeoJSON. The result must have geometry attached (e.g., by calling `.with_geometry()`).

```python
from core_lens.export import geoparquet, geojson

# Make sure geometry is attached!
spatial_res = aoi.mws.annual.with_geometry()

# Export to GeoParquet with DuckDB COPY options (e.g., partitioning)
geoparquet(
    spatial_res,
    "output.geoparquet",
    compression="ZSTD",
    partition_by="year"
)

# Export to GeoJSON
geojson(spatial_res, "output.json")
```

> [!NOTE]
> Exporting to geospatial formats automatically uses DuckDB to cast WKB geometries to proper spatial types via the `spatial` extension. CoreLens handles the installation and loading of the extension seamlessly under the hood.
