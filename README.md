# CoreLens

![GitHub last commit](https://img.shields.io/github/last-commit/ApoorvaKashyap/core-lens)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ApoorvaKashyap/core-lens/ci.yml)
![GitHub repo size](https://img.shields.io/github/repo-size/ApoorvaKashyap/core-lens)

A Python library for querying, analysing, and visualising geospatial Earth science data built on CoreStack.

Provides a unified interface over microwatershed hydrology, administrative boundaries, and pluggable domain entities — with lazy Parquet I/O, seasonal time-series support, and spatial statistical analysis.

## Features

- **Lazy I/O**: Uses Polars for fast, out-of-core reading of large Parquet datasets (e.g., NLCD, streamflows, MODIS).
- **Domain Entities**: First-class support for `Microwatershed`, `Subbasin`, `HydrologicElement`, and `AdminBoundary`.
- **Hydrology**: Tools for upstream routing, flow-accumulation analysis, and connectivity inference.
- **Time Series**: Flexible seasonal windowing and resamplling for analysis of time-varying data.
