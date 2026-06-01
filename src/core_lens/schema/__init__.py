"""Schema introspection utilities for core_lens."""

__module__: str = "core_lens.schema"

from core_lens.schema.profile import SchemaProfile
from core_lens.schema.detection import SchemaDetectionError, detect

__all__ = ["SchemaDetectionError", "SchemaProfile", "detect"]
