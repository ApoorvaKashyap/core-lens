# Core Concepts

CoreLens is built on three main layers:

1. **AoI (Area of Interest)**: The primary entry point. It represents a spatial boundary and acts as a gateway to all registered entities that intersect that boundary.
2. **View**: A lazy, immutable query definition. Filtering by attributes (`where`), space (`spatial_filter`), or time (`between`) returns a new `View` without reading any Parquet data.
3. **Result**: The materialised data. Accessing `.static`, `.annual`, or `.fortnightly` on a `View` executes the query using the Polars streaming engine (automatically routed to the GPU if RAPIDS `cudf-polars` is installed) and returns a `Result` object. All statistical operations, aggregations, and plotting are done on `Result` objects.
