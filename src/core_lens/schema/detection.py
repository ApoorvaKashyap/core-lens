"""Schema auto-detection from Parquet file metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from core_lens.schema.profile import SchemaProfile

if TYPE_CHECKING:
    pass

_BBOX_PATTERNS: list[tuple[str, str, str, str]] = [
    ("minx", "miny", "maxx", "maxy"),
    ("xmin", "ymin", "xmax", "ymax"),
    ("bbox_minx", "bbox_miny", "bbox_maxx", "bbox_maxy"),
    ("lon_min", "lat_min", "lon_max", "lat_max"),
]

_TIME_COL_NAMES: set[str] = {
    "year",
    "date",
    "time",
    "period",
    "annual_date",
    "fortnightly_date",
    "obs_date",
}


class SchemaDetectionError(Exception):
    """Raised when the detection layer cannot produce a valid SchemaProfile.

    The message identifies which file and which inference step failed so
    entity authors have an actionable starting point.
    """


def detect(
    static_path: str,
    key_cols: list[str],
    geometry_col: str,
    annual_path: str | None = None,
    fortnightly_path: str | None = None,
) -> SchemaProfile:
    """Introspect Parquet file schemas and return a validated SchemaProfile.

    Reads only Parquet footer metadata — no row data is scanned.  Intended
    to be called once per entity at ``AoI.register()`` time and the result
    cached on the entity instance.

    Detection order for each inferred field:

    * **geometry_type** — derived from the dtype of ``geometry_col``:
      ``Binary`` → ``"wkb"``, ``String`` → ``"wkt"``, ``Float``
      with a detected lon companion column → ``"latlon"``.
    * **bbox_cols** — checked against a fixed set of four-column name
      patterns.  ``None`` if no known pattern is present in the static schema.
    * **annual_time_col** / **fortnightly_time_col** — the first column with
      a ``Date``, ``Datetime``, or ``Int`` dtype whose name matches a known
      set of time-column names.  Falls back to the first ``Date``/``Datetime``
      column regardless of name.
    * **extra_*_cols** — every remaining column after key, geometry, bbox,
      and time columns are accounted for.

    Args:
        static_path: Path to the static GeoParquet file.
        key_cols: Entity key column name(s), as declared on the entity.
        geometry_col: Geometry column name, as declared on the entity.
        annual_path: Path to the annual Parquet file, or ``None``.
        fortnightly_path: Path to the fortnightly Parquet file, or ``None``.

    Returns:
        A validated :class:`~core_lens.schema.profile.SchemaProfile`.

    Raises:
        SchemaDetectionError: If a required column is absent, the geometry
            type cannot be inferred, or any other detection step fails.
    """
    static_schema = _read_schema(static_path, label="static")

    _require_cols(static_schema, key_cols, static_path)
    _require_cols(static_schema, [geometry_col], static_path)

    geometry_type, lon_col = _infer_geometry_type(
        static_schema, geometry_col, static_path
    )
    bbox_cols = _infer_bbox_cols(static_schema)

    reserved_static = set(key_cols) | {geometry_col}
    if bbox_cols:
        reserved_static |= set(bbox_cols)
    if lon_col:
        reserved_static.add(lon_col)
    extra_static_cols = [c for c in static_schema if c not in reserved_static]

    annual_time_col: str | None = None
    extra_annual_cols: list[str] = []
    if annual_path:
        annual_schema = _read_schema(annual_path, label="annual")
        annual_time_col = _infer_time_col(annual_schema, annual_path)
        reserved_annual = set(key_cols) | (
            {annual_time_col} if annual_time_col else set()
        )
        extra_annual_cols = [c for c in annual_schema if c not in reserved_annual]

    fortnightly_time_col: str | None = None
    extra_fortnightly_cols: list[str] = []
    if fortnightly_path:
        fortnightly_schema = _read_schema(fortnightly_path, label="fortnightly")
        fortnightly_time_col = _infer_time_col(fortnightly_schema, fortnightly_path)
        reserved_fn = set(key_cols) | (
            {fortnightly_time_col} if fortnightly_time_col else set()
        )
        extra_fortnightly_cols = [c for c in fortnightly_schema if c not in reserved_fn]

    return SchemaProfile(
        key_cols=key_cols,
        geometry_col=geometry_col,
        geometry_type=geometry_type,
        annual_time_col=annual_time_col,
        fortnightly_time_col=fortnightly_time_col,
        bbox_cols=bbox_cols,
        extra_static_cols=extra_static_cols,
        extra_annual_cols=extra_annual_cols,
        extra_fortnightly_cols=extra_fortnightly_cols,
    )


def _read_schema(path: str, label: str) -> pl.Schema:
    try:
        return pl.read_parquet_schema(path)
    except Exception as exc:
        raise SchemaDetectionError(
            f"Could not read Parquet schema from {label} file {path!r}: {exc}"
        ) from exc


def _require_cols(schema: pl.Schema, cols: list[str], path: str) -> None:
    missing = [c for c in cols if c not in schema]
    if missing:
        raise SchemaDetectionError(
            f"Required column(s) {missing} not found in {path!r}. "
            f"Available columns: {list(schema.names())}."
        )


def _infer_bbox_cols(schema: pl.Schema) -> tuple[str, str, str, str] | None:
    for pattern in _BBOX_PATTERNS:
        if all(col in schema for col in pattern):
            return pattern
    return None


def _infer_geometry_type(
    schema: pl.Schema,
    geometry_col: str,
    path: str,
) -> tuple[str, str | None]:
    dtype = schema[geometry_col]

    if dtype == pl.Binary:
        return "wkb", None

    if dtype in (pl.String, pl.Utf8):
        return "wkt", None

    # Float geometry column → assume it is a latitude column; look for a lon companion.
    if dtype in (pl.Float32, pl.Float64):
        lon_col = _find_lon_companion(schema, geometry_col)
        if lon_col:
            return "latlon", lon_col
        raise SchemaDetectionError(
            f"Geometry column {geometry_col!r} in {path!r} has a float dtype but no "
            "recognisable longitude companion column could be found. "
            "Declare geometry_type='latlon' and bbox_cols explicitly on the entity."
        )

    raise SchemaDetectionError(
        f"Cannot infer geometry_type from column {geometry_col!r} "
        f"(dtype={dtype}) in {path!r}. "
        "Supported dtypes: Binary (wkb), String (wkt), Float with lon companion (latlon)."
    )


def _find_lon_companion(schema: pl.Schema, lat_col: str) -> str | None:
    # Heuristic: if the lat column is named "lat" or "latitude", look for the
    # obvious lon counterpart.  Returns None if nothing matches.
    candidates = {
        "lat": ["lon", "lng", "longitude"],
        "latitude": ["lon", "lng", "longitude"],
    }
    for lat_name, lon_names in candidates.items():
        if lat_col.lower() == lat_name:
            for lon_name in lon_names:
                if lon_name in schema:
                    return lon_name
    return None


def _infer_time_col(schema: pl.Schema, path: str) -> str | None:
    # Prefer columns whose names are in the known set first so that e.g. a
    # column named "year" wins over an unnamed Datetime column.
    for col in schema:
        if col in _TIME_COL_NAMES:
            return col

    # Fall back: first Date or Datetime column regardless of name.
    for col, dtype in schema.items():
        if dtype in (pl.Date, pl.Datetime):
            return col

    # An integer column named after a year pattern is acceptable for annual files.
    for col, dtype in schema.items():
        if dtype in (pl.Int32, pl.Int64) and "year" in col.lower():
            return col

    return None
