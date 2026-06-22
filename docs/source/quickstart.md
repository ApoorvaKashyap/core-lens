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

Create an `AoI` (Area of Interest) by specifying the `data_root` and your boundary (either by name, bounding box, or Shapely geometry):

```python
# Create an AoI scoped to a specific Tehsil
aoi = AoI(
    data_root="/path/to/data",
    tehsil="Pangi"
)

# Or scope it directly to a specific list of entity IDs
aoi_specific = AoI(
    data_root="/path/to/data",
    mws_id=["13_001", "13_002"]
)

# You can now access entities scoped to this AoI:
mws_view = aoi.mws

# Materialise the static data (GeoDataFrame)
static_result = mws_view.static
gdf = static_result.gdf()

# Plot the AoI and its microwatersheds on an interactive map
aoi.plot(overlay=static_result)
```
