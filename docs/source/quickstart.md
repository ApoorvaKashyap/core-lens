# Quickstart

## Installation

```bash
pip install core-lens
```

To enable GPU-accelerated queries (requires an NVIDIA GPU and Linux), install with the `gpu` extra. You will need to configure your package manager to use the NVIDIA PyPI index for RAPIDS dependencies:

```bash
pip install "core-lens[gpu]" --extra-index-url=https://pypi.nvidia.com
```

## Basic Usage

Before querying, you must register the entities you plan to use:

```python
from core_lens import AoI
from core_lens.entities import MWSEntity, TehsilEntity

# Register entities
AoI.register(MWSEntity)
AoI.register(TehsilEntity)
```

Create an `AoI` (Area of Interest) by specifying the `data_root` (which can be a local path or a cloud URI like `s3://bucket/data`) and your boundary (either by name, bounding box, or Shapely geometry):

```python
# Create an AoI scoped to a specific Tehsil using a local path
aoi = AoI(
    data_root="/path/to/data",
    tehsil="Pangi"
)

# Or scope it directly to a specific list of entity IDs using an S3 bucket
aoi_specific = AoI(
    data_root="s3://my-cloud-bucket/core-lens-data",
    mws_id=["13_001", "13_002"]
)

# You can now access entities scoped to this AoI:
mws_view = aoi.mws

# Resolve the static data query
static_result = mws_view.static
# Materialise as a GeoDataFrame (or use .df() for a Polars DataFrame)
gdf = static_result.gdf()

# Plot the AoI and its microwatersheds on an interactive map
aoi.plot(overlay=static_result)
```
