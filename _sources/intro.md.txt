# Introduction

A Python library for querying, analysing, and visualising geospatial Earth science data built on CoreStack.

CoreLens provides a unified interface over microwatersheds, administrative boundaries, and pluggable domain entities — with lazy Parquet I/O (powered by Polars), seasonal time-series support, and spatial statistical analysis.

## Key Features

- **Area of Interest (AoI) First**: Define your spatial boundary once and instantly access all underlying entities (microwatersheds, villages, tehsils) scoped to that boundary.
- **Lazy Evaluation**: Uses Polars for lazy evaluation and predicate pushdown. Data is only read from Parquet files when explicitly materialised.
- **GPU Acceleration**: Zero-code GPU acceleration for query execution and aggregations via NVIDIA RAPIDS (`cudf-polars`), capable of automatically routing compatible queries to the GPU.
- **Pluggable Entities**: Built-in support for standard units (MWS, Tehsil) with a simple plugin architecture for adding new domain entities.
- **Temporal & Seasonal Awareness**: Native support for agronomic seasons (Kharif, Rabi, Zaid) and time-range filtering.
- **Spatial Statistics & Analysis**: Built-in methods for anomaly detection, spatial similarity, temporal correlation, and hypothesis testing.
- **Interactive Visualisation**: Generate interactive maps using Lonboard with zero-copy GeoArrow rendering directly from Polars, and timeseries/scatter plots using Bokeh.
- **Cloud & Local Data**: Native support for reading Parquet data from local filesystems or directly from cloud storage (e.g., `s3://`).
- **High-Performance Exports**: Export results directly to Parquet, JSON, CSV, or spatially-enabled GeoParquet and GeoJSON (with a fast streaming writer).
- **Advanced Caching**: Transparently caches boundaries and pre-computes spatial indices (via on-disk sidecars) to achieve sub-second query initialisation times.
