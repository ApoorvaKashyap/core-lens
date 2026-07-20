# Core Concepts

CoreLens is built on three main layers:

1. **AoI (Area of Interest)**: The primary entry point. It represents a spatial boundary and acts as a gateway to all registered entities that intersect that boundary.
2. **View**: A lazy, immutable query definition. Filtering by attributes (`where`), space (`spatial_filter`), or time (`between`) returns a new `View` without reading any Parquet data.
3. **Result**: The lazy computation graph. Accessing `.static`, `.annual`, or `.sub_annual` on a `View` resolves the query plan and returns a `Result` object wrapping a Polars `LazyFrame`. Data is not actually read until you materialise it by calling `.df()` or `.gdf()`. This delays execution and reduces memory footprint, allowing the Polars streaming engine (automatically routed to the GPU if RAPIDS `cudf-polars` is installed) to optimize the query globally. All statistical operations, aggregations, and plotting are done on `Result` objects.
