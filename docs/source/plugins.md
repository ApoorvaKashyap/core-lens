# Plugin System

You can extend CoreLens by defining your own entities. Simply subclass `BaseEntity`:

```python
from core_lens.base import BaseEntity

class CustomEntity(BaseEntity):
    @property
    def key_cols(self) -> list[str]:
        return ["custom_id"]

    @property
    def geometry_col(self) -> str:
        return "geometry"

    @property
    def static_path(self) -> str:
        return "custom/static.geoparquet"  # Relative to AoI data_root

    @property
    def annual_path(self) -> str | None:
        return "custom/annual.parquet"

# Register before use
AoI.register(CustomEntity)
```
