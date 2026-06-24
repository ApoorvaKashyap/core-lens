"""Spatial helpers for in-memory index construction and geometry filtering."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import polars as pl
import shapely

if TYPE_CHECKING:
    pass


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

    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    pf = pq.ParquetFile(static_path)
    chunks = []

    for batch in pf.iter_batches(batch_size=25_000, columns=cols_to_read):
        batch_df = pl.from_arrow(batch)
        assert isinstance(batch_df, pl.DataFrame)

        geom_array = batch_df.get_column(geometry_col).to_numpy()

        if geometry_type == "wkb":
            geoms = shapely.from_wkb(geom_array)
        else:
            geoms = shapely.from_wkt(geom_array)

        if len(geoms) > 0:
            bnds = shapely.bounds(geoms)
            chunk = batch_df.select(key_cols).with_columns(
                pl.Series("minx", bnds[:, 0], dtype=pl.Float64),
                pl.Series("miny", bnds[:, 1], dtype=pl.Float64),
                pl.Series("maxx", bnds[:, 2], dtype=pl.Float64),
                pl.Series("maxy", bnds[:, 3], dtype=pl.Float64),
            )
            chunks.append(chunk)

    if chunks:
        return pl.concat(chunks)

    empty_df = pl.read_parquet(static_path, columns=key_cols)
    return empty_df.with_columns(
        pl.Series("minx", [], dtype=pl.Float64),
        pl.Series("miny", [], dtype=pl.Float64),
        pl.Series("maxx", [], dtype=pl.Float64),
        pl.Series("maxy", [], dtype=pl.Float64),
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
    relationship: str = "centroid",
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Refine a bbox candidate set to rows that match ``aoi_geometry``.

    Reads only the key and geometry columns for the candidate rows, decodes
    each geometry, and tests for the requested spatial relationship.

    Args:
        candidates: The DataFrame from :func:`bbox_intersects_geometry` —
            only the key columns are used here; the bbox columns are ignored.
        static_path: Absolute path to the static GeoParquet file.
        key_cols: Column name(s) that form the entity's unique key.
        geometry_col: Name of the geometry column in the static file.
        geometry_type: One of ``"wkb"`` or ``"wkt"``.
        aoi_geometry: The Area of Interest geometry to test against.
        relationship: Spatial relationship mode.

            * ``"centroid"`` (default) — entity centroid must lie **within**
              ``aoi_geometry``.
            * ``"area"`` — fraction of the entity's area covered by the
              intersection must exceed ``threshold``.

        threshold: Minimum intersection-to-entity area ratio used in
            ``"area"`` mode.  Ignored in ``"centroid"`` mode.  Default 0.5.

    Returns:
        A ``pl.DataFrame`` containing only the key columns for entities that
        satisfy the spatial relationship.

    Raises:
        ValueError: If ``relationship`` is not one of the valid options.
    """
    import shapely

    if relationship not in ("centroid", "area"):
        raise ValueError(
            f"exact_spatial_filter: Unknown relationship={relationship!r}. "
            "Valid options: 'centroid', 'area'."
        )

    if candidates.is_empty():
        return candidates.select(key_cols)

    # Build a lazy scan and push down an inner join to load only required geometries.
    full_df = (
        pl.scan_parquet(static_path)
        .select(key_cols + [geometry_col])
        .join(candidates.select(key_cols).lazy(), on=key_cols, how="inner")
        .collect()
    )

    geom_array = full_df[geometry_col].to_numpy()
    if geometry_type == "wkb":
        geoms = shapely.from_wkb(geom_array)
    else:
        geoms = shapely.from_wkt(geom_array)

    if relationship == "centroid":
        # Centroid mode: entity centroid must lie within the AoI geometry.
        # STRtree.query(aoi_geometry, predicate="contains") returns i where
        # aoi_geometry.contains(centroid[i]) — i.e. centroid inside the AoI.
        test_geoms = [g.centroid for g in geoms]
        tree = shapely.STRtree(test_geoms)
        hit_indices = tree.query(aoi_geometry, predicate="contains").tolist()

    else:  # area mode
        # Area mode: intersection area / entity area > threshold.
        tree = shapely.STRtree(geoms)
        # Candidate indices whose envelope intersects aoi_geometry.
        intersect_idx = tree.query(aoi_geometry, predicate="intersects").tolist()
        hit_indices = []
        for i in intersect_idx:
            entity_geom = geoms[i]
            entity_area = entity_geom.area  # pyright: ignore[reportAttributeAccessIssue]
            if entity_area == 0.0:
                continue
            inter_area = entity_geom.intersection(aoi_geometry).area  # pyright: ignore[reportAttributeAccessIssue]
            if inter_area / entity_area >= threshold:
                hit_indices.append(i)

    matched: pl.DataFrame = full_df[hit_indices].select(key_cols)
    return matched


def execute_spatial_join(
    primary_df: pl.DataFrame,
    primary_key_cols: list[str],
    primary_geom_col: str,
    primary_geom_type: str,
    other_entity: Any,
    agg: dict[str, str],
    other_entity_name: str,
) -> pl.DataFrame:
    """Materialise a cross-entity spatial join and return enriched DataFrame.

    For each entity in ``primary_df``, finds the overlapping entities in
    ``other_entity`` and aggregates the requested columns.  Result columns are
    named ``{other_entity_name}_{column_name}`` to avoid clashes.

    Args:
        primary_df: The primary entity DataFrame (must contain
            ``primary_geom_col`` unless geometry is in a separate column).
        primary_key_cols: Key column(s) of the primary entity.
        primary_geom_col: Name of the WKB geometry column in ``primary_df``.
        primary_geom_type: Geometry encoding — ``"wkb"`` or ``"wkt"``.
        other_entity: A :class:`~core_lens.base.entity.BaseEntity` instance
            to join against.
        agg: Mapping ``{column: aggregation}`` — which columns from
            ``other_entity`` to bring in and how to aggregate them.  Valid
            aggregations: ``"count"``, ``"mean"``, ``"sum"``, ``"min"``,
            ``"max"``, ``"area"``.
        other_entity_name: Used to prefix result column names.

    Returns:
        ``primary_df`` with additional columns
        ``{other_entity_name}_{col}`` appended for each ``agg`` entry.
    """
    import shapely

    other_profile = other_entity.schema_profile
    other_static = other_entity._resolve(other_entity.static_path)
    other_geom_col = other_profile.geometry_col
    other_geom_type = other_profile.geometry_type
    other_key_cols = other_entity.key_cols

    # Read other entity: key + geometry + agg columns.
    agg_col_names = [c for c in agg if c not in ("count", "area")]
    other_cols = list(dict.fromkeys(other_key_cols + [other_geom_col] + agg_col_names))
    other_df = pl.read_parquet(other_static, columns=other_cols)

    # Decode other geometries.
    other_geom_array = other_df[other_geom_col].to_numpy()
    if other_geom_type == "wkb":
        other_geoms = shapely.from_wkb(other_geom_array)
    else:
        other_geoms = shapely.from_wkt(other_geom_array)

    # Build STRtree from other entity geometries.
    other_tree = shapely.STRtree(other_geoms)

    # Only convert the columns we actually need to aggregate to Pandas, avoiding
    # a massive memory copy of the WKB geometry strings and unused keys.
    agg_cols_only = [
        c for c in agg.keys() if c not in ("count", "area") and c in other_df.columns
    ]
    if agg_cols_only:
        other_df_np = other_df.select(agg_cols_only).to_pandas()
    else:
        other_df_np = None

    # Decode primary geometries.
    if primary_geom_col not in primary_df.columns:
        raise ValueError(
            f"execute_spatial_join: geometry column {primary_geom_col!r} not found "
            "in primary DataFrame.  Materialise static resolution or call "
            "with_geometry() first."
        )

    primary_geom_array = primary_df[primary_geom_col].to_numpy()
    if primary_geom_type == "wkb":
        primary_geoms = shapely.from_wkb(primary_geom_array)
    else:
        primary_geoms = shapely.from_wkt(primary_geom_array)

    # For each primary entity, find overlapping other entities and aggregate.
    result_rows: list[dict[str, Any]] = []
    primary_keys = primary_df.select(primary_key_cols).to_dicts()

    # Vectorized STRtree query over all primary geometries at once
    # Returns a 2D array of shape (2, N): [[primary_indices], [other_indices]]
    idx_pairs = other_tree.query(primary_geoms, predicate="intersects")

    from collections import defaultdict

    hits_by_primary = defaultdict(list)
    for p_idx, o_idx in zip(idx_pairs[0], idx_pairs[1]):
        hits_by_primary[p_idx].append(o_idx)

    for i, (pgeom, pkey) in enumerate(zip(primary_geoms, primary_keys)):
        hit_idx = hits_by_primary.get(i, [])
        row: dict[str, Any] = dict(pkey)

        if not hit_idx:
            for col, fn in agg.items():
                out_col = f"{other_entity_name}_{col}"
                row[out_col] = None
            result_rows.append(row)
            continue

        if other_df_np is not None:
            matched_other = other_df_np.iloc[hit_idx]
        else:
            matched_other = None

        for col, fn in agg.items():
            out_col = f"{other_entity_name}_{col}"
            if fn == "count":
                row[out_col] = len(hit_idx)
            elif fn == "area":
                # Sum of intersection areas (geographic, approx).
                total = sum(pgeom.intersection(other_geoms[j]).area for j in hit_idx)  # pyright: ignore[reportAttributeAccessIssue]
                row[out_col] = total
            elif matched_other is not None and col in matched_other.columns:
                vals = matched_other[col].dropna()
                if fn == "mean":
                    row[out_col] = float(vals.mean()) if len(vals) else None
                elif fn == "sum":
                    row[out_col] = float(vals.sum()) if len(vals) else None
                elif fn == "min":
                    row[out_col] = float(vals.min()) if len(vals) else None
                elif fn == "max":
                    row[out_col] = float(vals.max()) if len(vals) else None
                else:
                    row[out_col] = None
            else:
                row[out_col] = None
        result_rows.append(row)

    join_df = pl.from_dicts(result_rows)
    return primary_df.join(
        join_df.select(
            [c for c in join_df.columns if c not in primary_df.columns]
            + primary_key_cols
        ),
        on=primary_key_cols,
        how="left",
    )
