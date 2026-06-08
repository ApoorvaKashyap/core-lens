"""Pydantic model describing the schema of one entity's data files."""

from __future__ import annotations

from enum import Enum
import pathlib
from typing import Literal

from pydantic import BaseModel, model_validator


class Resolution(str, Enum):
    """Temporal resolution of a dataset."""

    STATIC = "static"
    ANNUAL = "annual"
    FORTNIGHTLY = "fortnightly"


class SchemaProfile(BaseModel):
    """Validated descriptor for the column layout of one entity's Parquet files.

    ``SchemaProfile`` is a Pydantic model, so all fields are validated at
    construction time.  It is built once per entity — either declared
    explicitly in a concrete :class:`~core_lens.base.entity.BaseEntity`
    subclass, or produced lazily by the schema detection layer when a path is
    first accessed.

    The model is intentionally narrow: it captures only what the rest of the
    library needs to adapt to schema changes at runtime (key identities, geometry
    encoding, time column names, optional bbox columns).  Columns not listed in
    one of the ``extra_*`` fields are ignored rather than rejected, so adding
    a new attribute column to a Parquet file is always a non-breaking change.

    Attributes:
        key_cols: Column name(s) that form the unique identifier for one entity
            instance.  Must be present in the static file and in every temporal
            file that references the entity.
        geometry_col: Name of the geometry column in the static GeoParquet file.
        geometry_type: Encoding of the geometry column.  ``"wkb"`` for WKB-encoded
            bytes (standard GeoParquet), ``"wkt"`` for WKT strings, ``"latlon"``
            for separate latitude/longitude columns (requires ``bbox_cols`` to be
            set or a specific lat/lon pair to be documented separately).
        annual_time_col: Name of the date/year column in the annual Parquet file,
            or ``None`` if the entity has no annual data.
        fortnightly_time_col: Name of the date column in the fortnightly Parquet
            file, or ``None`` if the entity has no fortnightly data.
        bbox_cols: Four-tuple ``(minx, miny, maxx, maxy)`` of column names
            holding the pre-computed bounding box for each entity instance in the
            static file.  Used to build the in-memory index without a geometry
            decode pass.  ``None`` if the static file does not carry bbox columns
            (the index will be built by decoding the geometry column instead).
        extra_static_cols: Additional attribute columns in the static file that
            the library should surface.  Columns absent from this list are still
            readable via :meth:`~core_lens.base.result.Result.df`; this field
            is used for validation and documentation only.
        extra_annual_cols: Additional attribute columns in the annual file.
        extra_fortnightly_cols: Additional attribute columns in the fortnightly file.
    """

    model_config = {"frozen": True}

    key_cols: list[str]
    geometry_col: str
    geometry_type: Literal["wkb", "wkt", "latlon"]
    annual_time_col: str | None
    fortnightly_time_col: str | None
    bbox_cols: tuple[str, str, str, str] | None
    extra_static_cols: list[str] = []
    extra_annual_cols: list[str] = []
    extra_fortnightly_cols: list[str] = []

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "SchemaProfile":
        """Load a schema profile from a JSON file.

        Args:
            path: Path to the JSON configuration file.

        Returns:
            A new :class:`SchemaProfile` instance.

        Raises:
            ValueError: If the file is not valid JSON.
            FileNotFoundError: If the file does not exist.
        """
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @model_validator(mode="after")
    def _check_key_cols_non_empty(self) -> "SchemaProfile":
        if not self.key_cols:
            raise ValueError("key_cols must contain at least one column name.")
        return self

    @model_validator(mode="after")
    def _check_latlon_needs_bbox(self) -> "SchemaProfile":
        # For latlon encoding the library cannot derive a bounding box without
        # the bbox_cols hint, because there is no single geometry column to decode.
        if self.geometry_type == "latlon" and self.bbox_cols is None:
            raise ValueError(
                "geometry_type='latlon' requires bbox_cols to be set so the "
                "in-memory spatial index can be built without a geometry decode pass."
            )
        return self
