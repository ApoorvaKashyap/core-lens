"""Schema introspection utilities for core_lens."""

from core_lens.schema.profile import Resolution, SchemaProfile
from core_lens.schema.detection import SchemaDetectionError, detect

__all__ = ["Resolution", "SchemaDetectionError", "SchemaProfile", "detect"]
