"""Spatial helpers for in-memory index construction and geometry filtering."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import polars as pl
import shapely.wkb as swkb
import shapely.wkt as swkt

if TYPE_CHECKING:
    import shapely


def resolve_path(path: str) -> str:
    """Return an absolute path string, resolving relative paths against cwd.

    Args:
        path: A filesystem path, absolute or relative.

    Returns:
        An absolute path string.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = pathlib.Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(
            f"Entity path {path!r} (resolved to {p}) does not exist. "
            "Provide an absolute path or ensure the file exists relative to "
            "the current working directory."
        )
    return str(p)


def build_bbox_index(
    static_path: str,
    key_cols: list[str],
    bbox_cols: tuple[str, str, str, str] | None,
    geometry_col: str,
    geometry_type: str,
) -> pl.DataFrame:
    """Build the in-memory ``(key_cols..., minx, miny, maxx, maxy)`` index.

    Reads the static GeoParquet file using only the key and spatial columns —
    no attribute data is pulled into memory.  If pre-computed bbox columns are
    present in the file they are read directly; otherwise the geometry column
    is decoded and bounds are computed via Shapely.

    Args:
        static_path: Absolute path to the static GeoParquet file.
        key_cols: Column name(s) that form the entity's unique key.
        bbox_cols: Four-column ``(minx, miny, maxx, maxy)`` tuple if the
            static file carries pre-computed bounding boxes, otherwise ``None``.
        geometry_col: Name of the geometry column.
        geometry_type: One of ``"wkb"``, ``"wkt"``, or ``"latlon"``.

    Returns:
        A ``pl.DataFrame`` with columns ``(*key_cols, minx, miny, maxx, maxy)``.
    """
    if bbox_cols is not None:
        cols_to_read = key_cols + list(bbox_cols)
        df = pl.read_parquet(static_path, columns=cols_to_read)
        minx_col, miny_col, maxx_col, maxy_col = bbox_cols
        return df.rename(
            {minx_col: "minx", miny_col: "miny", maxx_col: "maxx", maxy_col: "maxy"}
        )

    cols_to_read = key_cols + [geometry_col]
    if geometry_type == "latlon":
        # For latlon, bbox_cols must be set — this path should not be reached.
        raise ValueError(
            "geometry_type='latlon' requires bbox_cols to be declared on the entity. "
            + "Cannot compute bounds from separate lat/lon columns without bbox hints."
        )

    df = pl.read_parquet(static_path, columns=cols_to_read)
    geom_series = df[geometry_col]

    if geometry_type == "wkb":
        geoms = [swkb.loads(v) for v in geom_series.to_list()]
    else:
        geoms = [swkt.loads(v) for v in geom_series.to_list()]

    bounds = [g.bounds for g in geoms]
    minx, miny, maxx, maxy = zip(*bounds) if bounds else ([], [], [], [])

    return df.select(key_cols).with_columns(
        pl.Series("minx", list(minx), dtype=pl.Float64),
        pl.Series("miny", list(miny), dtype=pl.Float64),
        pl.Series("maxx", list(maxx), dtype=pl.Float64),
        pl.Series("maxy", list(maxy), dtype=pl.Float64),
    )


def bbox_intersects_geometry(
    index_df: pl.DataFrame,
    geometry: "shapely.Geometry",
) -> pl.DataFrame:
    """Return the index rows whose bounding box overlaps ``geometry``'s bounds.

    This is a fast rectangular pre-filter in pure Polars — no STRtree.  It
    returns a superset of the exact result, which is then refined by
    :func:`exact_spatial_filter`.

    Args:
        index_df: The in-memory index DataFrame with ``minx, miny, maxx, maxy``
            columns produced by :func:`build_bbox_index`.
        geometry: Any Shapely geometry representing the area of interest.

    Returns:
        The subset of ``index_df`` whose rows overlap the geometry bounds.
    """
    gminx, gminy, gmaxx, gmaxy = geometry.bounds  # type: ignore[attr-defined]
    return index_df.filter(
        (pl.col("maxx") >= gminx)
        & (pl.col("minx") <= gmaxx)
        & (pl.col("maxy") >= gminy)
        & (pl.col("miny") <= gmaxy)
    )


def exact_spatial_filter(
    candidates: pl.DataFrame,
    static_path: str,
    key_cols: list[str],
    geometry_col: str,
    geometry_type: str,
    aoi_geometry: "shapely.Geometry",
) -> pl.DataFrame:
    """Refine a bbox candidate set to rows that exactly intersect ``aoi_geometry``.

    Reads only the key and geometry columns for the candidate rows, decodes
    each geometry, and tests for intersection with ``aoi_geometry`` using a
    Shapely STRtree for efficiency.

    Args:
        candidates: The DataFrame from :func:`bbox_intersects_geometry` —
            only the key columns are used here; the bbox columns are ignored.
        static_path: Absolute path to the static GeoParquet file.
        key_cols: Column name(s) that form the entity's unique key.
        geometry_col: Name of the geometry column in the static file.
        geometry_type: One of ``"wkb"`` or ``"wkt"``.
        aoi_geometry: The Area of Interest geometry to test intersection with.

    Returns:
        A ``pl.DataFrame`` containing only the key columns for entities that
        exactly intersect ``aoi_geometry``.
    """
    import shapely

    if candidates.is_empty():
        return candidates.select(key_cols)

    # Build a filter expression to read only the candidate rows.
    full_df = pl.read_parquet(static_path, columns=key_cols + [geometry_col])

    # Filter to candidate keys.
    if len(key_cols) == 1:
        key = key_cols[0]
        candidate_keys = candidates[key].to_list()
        full_df = full_df.filter(pl.col(key).is_in(candidate_keys))
    else:
        # Composite key: join on all key columns.
        full_df = full_df.join(candidates.select(key_cols), on=key_cols, how="inner")

    geom_series = full_df[geometry_col].to_list()
    if geometry_type == "wkb":
        geoms = [swkb.loads(v) for v in geom_series]
    else:
        geoms = [swkt.loads(v) for v in geom_series]

    # STRtree for efficient exact intersection.
    tree = shapely.STRtree(geoms)
    hit_indices = tree.query(aoi_geometry, predicate="intersects").tolist()

    matched: pl.DataFrame = full_df[hit_indices].select(key_cols)
    return matched
