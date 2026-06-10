# CoreLens

![GitHub last commit](https://img.shields.io/github/last-commit/ApoorvaKashyap/core-lens)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ApoorvaKashyap/core-lens/ci.yml)
![GitHub repo size](https://img.shields.io/github/repo-size/ApoorvaKashyap/core-lens)

A Python library for querying, analysing, and visualising geospatial Earth science data built on CoreStack.

CoreLens provides a unified interface over microwatersheds, administrative boundaries, and pluggable domain entities — with lazy Parquet I/O (powered by Polars), seasonal time-series support, and spatial statistical analysis.

## Quickstart

```bash
pip install core-lens
```

```python
from core_lens import AoI
from core_lens.entities import MWSEntity, TehsilEntity

# Register entities
AoI.register(MWSEntity)
AoI.register(TehsilEntity)

# Create an AoI scoped to a specific Tehsil
aoi = AoI(data_root="/path/to/data", tehsil="Pangi")

# Access entities scoped to this AoI
mws_view = aoi.mws

# Materialise and plot
static_result = mws_view.static
aoi.plot(overlay=static_result)
```

## Documentation

Full documentation is available in the `docs/` directory. It includes:
- **Core Concepts**: Understanding AoI, Views, and Results.
- **Example Queries**: Time filtering, aggregations, spatial joins, and statistics.
- **Plugin System**: How to add custom entities.

To build the docs locally:
```bash
cd docs
make html
```
