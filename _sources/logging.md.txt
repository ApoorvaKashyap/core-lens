# Logging

`core_lens` uses [Loguru](https://loguru.readthedocs.io/) for internal logging and tracing.

To avoid polluting the standard output of your downstream applications, the logger is disabled by default upon import.

## Enabling Logs

If you want to view the internal tracing logs (useful for debugging spatial resolution issues, inspecting generated time filters, or tracing validation errors), you need to explicitly enable the `core_lens` logger in your application and add a handler (e.g., `sys.stdout` or a log file).

```python
import sys
from loguru import logger
from core_lens import AoI

# 1. Enable the internal logger
logger.enable("core_lens")

# 2. Add a handler with your desired level and formatting
logger.add(sys.stdout, level="DEBUG")

# 3. Operations will now emit tracing logs
aoi = AoI("data/", tehsil="Pangi", district="Chamba")
```

## Log Levels

`core_lens` emits logs at different levels depending on the operational context:

- **INFO**: Materialisation events (reading static, annual, or fortnightly Parquet files), entity registrations, AoI initialisation, and cross-entity spatial joins.
- **DEBUG**: Sub-operations such as spatial bounding box index generation, exact geometry relationships, lazy attribute filters, and query generation.
- **ERROR**: Incorrect initialisation arguments, failed path validations, missing geometries, unknown schema structures, or conflicting date configurations.
