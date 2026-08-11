"""Spatial helpers for in-memory index construction and geometry filtering."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pyarrow.dataset as ds
import shapely
from loguru import logger

from core_lens.utils.paths import is_cloud_uri, path_exists, resolve_fs_and_path
from core_lens.utils.polars_utils import parquet_scan_path

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Bbox index sidecar helpers
#
# A sidecar Parquet file caches the pre-computed bounding-box index so that
# the expensive WKB-decode loop in build_bbox_index is only paid once per
# dataset (not once per process).  Subsequent runs load the index directly
# from the sidecar — no geometry decoding required.
#
# Sidecar location priority:
#   1. Alongside the static parquet directory/file (requires write access).
#   2. ~/.cache/core_lens/ (fallback for read-only data roots).
# ---------------------------------------------------------------------------

_SIDECAR_SUFFIX = ".bbox_index.parquet"
_CACHE_DIR = pathlib.Path.home() / ".cache" / "core_lens"


def _bbox_sidecar_path(static_path: str) -> pathlib.Path | None:
    """Return the preferred path for the bbox index sidecar, or ``None`` for cloud URIs.

    Tries the directory alongside ``static_path`` first; if that directory is
    not writable (e.g. read-only data root), falls back to
    ``~/.cache/core_lens/``.

    Args:
        static_path (str): Absolute local path to the static parquet file/directory.

    Returns:
        pathlib.Path | None: Sidecar path, or ``None`` if ``static_path`` is a cloud URI.

    """
    if is_cloud_uri(static_path):
        return None
    p = pathlib.Path(static_path)
    # For a partitioned directory dataset, the sidecar lives inside the directory.
    # For a single file, it lives next to it.
    parent = p if p.is_dir() else p.parent
    sidecar = parent / (p.name + _SIDECAR_SUFFIX)
    # Check if the parent directory is writable.
    if not parent.exists() or not parent.stat().st_mode & 0o200:
        # Fallback to user cache dir.
        import hashlib

        fingerprint = hashlib.sha256(static_path.encode()).hexdigest()[:16]
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sidecar = _CACHE_DIR / (fingerprint + _SIDECAR_SUFFIX)
    return sidecar


def _read_bbox_sidecar(sidecar: pathlib.Path, static_path: str) -> pl.DataFrame | None:
    """Read the sidecar if it exists and is newer than the static data.

    Args:
        sidecar (pathlib.Path): Candidate sidecar path.
        static_path (str): Absolute local path to the static parquet file/directory.

    Returns:
        pl.DataFrame | None: The cached index, or ``None`` if the sidecar is
        absent or stale.

    """
    if not sidecar.exists():
        return None
    try:
        static_mtime = pathlib.Path(static_path).stat().st_mtime
        sidecar_mtime = sidecar.stat().st_mtime
        if sidecar_mtime < static_mtime:
            logger.debug(
                "Bbox sidecar {} is stale (older than {}), rebuilding.",
                sidecar,
                static_path,
            )
            return None
        logger.debug("Loading bbox index from sidecar: {}", sidecar)
        return pl.read_parquet(str(sidecar))
    except Exception as exc:
        logger.warning("Failed to read bbox sidecar {}: {}", sidecar, exc)
        return None


def _write_bbox_sidecar(df: pl.DataFrame, sidecar: pathlib.Path) -> None:
    """Write ``df`` to the sidecar file, silently ignoring any write errors.

    Args:
        df (pl.DataFrame): The bbox index DataFrame to persist.
        sidecar (pathlib.Path): Destination sidecar path.

    """
    try:
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(str(sidecar))
        logger.debug("Wrote bbox index sidecar: {}", sidecar)
    except Exception as exc:
        logger.debug("Could not write bbox index sidecar {}: {}", sidecar, exc)


def resolve_path(path: str) -> str:
    """Return an absolute path or cloud URI string, resolving local relative paths against cwd.

    Args:
        path (str): A filesystem path (absolute or relative) or cloud URI.

    Returns:
        str: An absolute path string or cloud URI.

    Raises:
        FileNotFoundError: If the resolved local path does not exist.

    """
    if is_cloud_uri(path):
        return path
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = pathlib.Path.cwd() / p
    resolved = str(p)
    if not path_exists(resolved):
        logger.error(
            "Path resolution failed: '{}' (resolved to {}) does not exist.",
            path,
            resolved,
        )
        raise FileNotFoundError(
            f"Entity path {path!r} (resolved to {resolved!r}) does not exist. "
            "Provide an absolute path or ensure the file exists relative to "
            "the current working directory."
        )
    return resolved


def build_bbox_index(
    static_path: str,
    key_cols: list[str],
    bbox_cols: tuple[str, str, str, str] | None,
    geometry_col: str,
    geometry_type: str,
    storage_options: dict[str, Any] | None = None,
) -> pl.DataFrame:
    """Build the in-memory ``(key_cols..., minx, miny, maxx, maxy)`` index.

    Reads the static GeoParquet file using only the key and spatial columns —
    no attribute data is pulled into memory.  If pre-computed bbox columns are
    present in the file they are read directly; otherwise the geometry column
    is decoded and bounds are computed via Shapely.

    Args:
        static_path (str): Absolute path or cloud URI to the static GeoParquet file.
        key_cols (list[str]): Column name(s) that form the entity's unique key.
        bbox_cols (tuple[str, str, str, str] | None): Four-column ``(minx, miny, maxx, maxy)`` tuple if the
            static file carries pre-computed bounding boxes, otherwise ``None``.
        geometry_col (str): Name of the geometry column.
        geometry_type (str): One of ``"wkb"``, ``"wkt"``, or ``"latlon"``.
        storage_options (dict[str, Any] | None, optional): Cloud credential / configuration
            options forwarded to ``pyarrow.fs`` and ``polars.read_parquet``.
            ``None`` uses ambient credentials.

    Returns:
        pl.DataFrame: A ``pl.DataFrame`` with columns ``(*key_cols, minx, miny, maxx, maxy)``.

    """
    _so = storage_options or {}
    if bbox_cols is not None:
        logger.debug("Using pre-computed bbox columns: {}", bbox_cols)
        base_bbox_cols = list(dict.fromkeys(c.split(".")[0] for c in bbox_cols))
        cols_to_read = key_cols + base_bbox_cols
        df = pl.read_parquet(
            static_path, columns=cols_to_read, storage_options=_so or None
        )
        minx_col, miny_col, maxx_col, maxy_col = bbox_cols

        for alias, col_path in zip(
            ["minx", "miny", "maxx", "maxy"],
            [minx_col, miny_col, maxx_col, maxy_col],
        ):
            parts = col_path.split(".")
            if len(parts) == 1:
                df = df.rename({col_path: alias})
            else:
                expr = pl.col(parts[0])
                for p in parts[1:]:
                    expr = expr.struct.field(p)
                df = df.with_columns(expr.alias(alias))

        return df.select(key_cols + ["minx", "miny", "maxx", "maxy"])

    cols_to_read = key_cols + [geometry_col]
    if geometry_type == "latlon":
        # For latlon, bbox_cols must be set — this path should not be reached.
        logger.error(
            "build_bbox_index failed: geometry_type='latlon' requires bbox_cols to be declared."
        )
        raise ValueError(
            "geometry_type='latlon' requires bbox_cols to be declared on the entity. "
            + "Cannot compute bounds from separate lat/lon columns without bbox hints."
        )

    # --- Sidecar fast path (local paths only) --------------------------------
    # Check for a pre-built index sidecar before doing the expensive WKB loop.
    sidecar: pathlib.Path | None = None
    if not is_cloud_uri(static_path):
        sidecar = _bbox_sidecar_path(static_path)
        if sidecar is not None:
            cached = _read_bbox_sidecar(sidecar, static_path)
            if cached is not None:
                return cached

    logger.debug(
        "No pre-computed bbox found in {}; falling back to Shapely decoding",
        static_path,
    )

    # Resolve filesystem for cloud paths so pyarrow.dataset can open the file.
    fs, arrow_path = resolve_fs_and_path(static_path)
    dataset = ds.dataset(arrow_path, filesystem=fs)
    chunks = []

    for batch in dataset.to_batches(columns=cols_to_read, batch_size=25_000):
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
        result = pl.concat(chunks)
    else:
        empty_df = pl.read_parquet(
            static_path, columns=key_cols, storage_options=_so or None
        )
        result = empty_df.with_columns(
            pl.Series("minx", [], dtype=pl.Float64),
            pl.Series("miny", [], dtype=pl.Float64),
            pl.Series("maxx", [], dtype=pl.Float64),
            pl.Series("maxy", [], dtype=pl.Float64),
        )

    # Persist the sidecar for future runs (local paths only, errors silenced).
    if sidecar is not None:
        _write_bbox_sidecar(result, sidecar)

    return result


def bbox_intersects_geometry(
    index_df: pl.DataFrame,
    geometry: "shapely.Geometry",
) -> pl.DataFrame:
    """Return the index rows whose bounding box overlaps ``geometry``'s bounds.

    This is a fast rectangular pre-filter in pure Polars — no STRtree.  It
    returns a superset of the exact result, which is then refined by
    :func:`exact_spatial_filter`.

    Args:
        index_df (pl.DataFrame): The in-memory index DataFrame with ``minx, miny, maxx, maxy``
            columns produced by :func:`build_bbox_index`.
        geometry (shapely.Geometry): Any Shapely geometry representing the area of interest.

    Returns:
        pl.DataFrame: The subset of ``index_df`` whose rows overlap the geometry bounds.

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
        candidates (pl.DataFrame): The DataFrame from :func:`bbox_intersects_geometry` —
            only the key columns are used here; the bbox columns are ignored.
        static_path (str): Absolute path to the static GeoParquet file.
        key_cols (list[str]): Column name(s) that form the entity's unique key.
        geometry_col (str): Name of the geometry column in the static file.
        geometry_type (str): One of ``"wkb"`` or ``"wkt"``.
        aoi_geometry (shapely.Geometry): The Area of Interest geometry to test against.
        relationship (str, optional): Spatial relationship mode.

            * ``"centroid"`` (default) — entity centroid must lie **within**
              ``aoi_geometry``.
            * ``"area"`` — fraction of the entity's area covered by the
              intersection must exceed ``threshold``.

        threshold (float, optional): Minimum intersection-to-entity area ratio used in
            ``"area"`` mode.  Ignored in ``"centroid"`` mode.  Default 0.5.

    Returns:
        pl.DataFrame: A ``pl.DataFrame`` containing only the key columns for entities that
        satisfy the spatial relationship.

    Raises:
        ValueError: If ``relationship`` is not one of the valid options.

    """
    logger.debug(
        "Refining {} candidates with exact spatial filter (relationship='{}')",
        len(candidates),
        relationship,
    )

    if relationship not in ("centroid", "area"):
        logger.error(
            "exact_spatial_filter failed: Unknown relationship='{}'.", relationship
        )
        raise ValueError(
            f"exact_spatial_filter: Unknown relationship={relationship!r}. "
            "Valid options: 'centroid', 'area'."
        )

    if candidates.is_empty():
        return candidates.select(key_cols)

    # Build a lazy scan and push down an inner join to load only required geometries.
    full_df = (
        pl.scan_parquet(parquet_scan_path(static_path))
        .select(key_cols + [geometry_col])
        .join(candidates.select(key_cols).lazy(), on=key_cols, how="semi")
        .collect()
    )

    geom_array = full_df[geometry_col].to_numpy()
    if geometry_type == "wkb":
        geoms = shapely.from_wkb(geom_array)
    else:
        geoms = shapely.from_wkt(geom_array)

    if relationship == "centroid":
        # Centroid mode: entity centroid must lie within the AoI geometry.
        # STRtree is overkill for a single-query containment test — building
        # the tree costs O(n log n) with no payoff when there is only one query
        # geometry.  Vectorised shapely ops stay in C and avoid the overhead.
        centroids = shapely.centroid(geoms)  # vectorised, C-level
        mask = shapely.contains(aoi_geometry, centroids)  # vectorised bool array
        hit_indices = np.where(mask)[0].tolist()

    else:  # area mode
        # Area mode: intersection area / entity area >= threshold.
        # Vectorise: compute all areas and intersections in C via numpy arrays.
        tree = shapely.STRtree(geoms)
        intersect_idx = np.asarray(
            tree.query(aoi_geometry, predicate="intersects"), dtype=np.intp
        )
        if len(intersect_idx) == 0:
            hit_indices = []
        else:
            candidate_geoms = geoms[intersect_idx]
            entity_areas = shapely.area(candidate_geoms)  # vectorised
            valid = entity_areas > 0
            if not valid.any():
                hit_indices = []
            else:
                inter_geoms = shapely.intersection(
                    candidate_geoms[valid], aoi_geometry
                )  # vectorised
                inter_areas = shapely.area(inter_geoms)  # vectorised
                ratios = inter_areas / entity_areas[valid]
                hit_indices = intersect_idx[valid][ratios >= threshold].tolist()

    del geoms  # free all decoded Shapely geometries before assembling result

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
        primary_df (pl.DataFrame): The primary entity DataFrame (must contain
            ``primary_geom_col`` unless geometry is in a separate column).
        primary_key_cols (list[str]): Key column(s) of the primary entity.
        primary_geom_col (str): Name of the WKB geometry column in ``primary_df``.
        primary_geom_type (str): Geometry encoding — ``"wkb"`` or ``"wkt"``.
        other_entity (BaseEntity): A :class:`~core_lens.base.entity.BaseEntity` instance
            to join against.
        agg (dict[str, str]): Mapping ``{column: aggregation}`` — which columns from
            ``other_entity`` to bring in and how to aggregate them.  Valid
            aggregations: ``"count"``, ``"mean"``, ``"sum"``, ``"min"``,
            ``"max"``, ``"area"``.
        other_entity_name (str): Used to prefix result column names.

    Returns:
        pl.DataFrame: ``primary_df`` with additional columns
        ``{other_entity_name}_{col}`` appended for each ``agg`` entry.

    """
    logger.info(
        "Starting execute_spatial_join for other_entity_name={}", other_entity_name
    )

    # Decode primary geometries first so we can bound the other entity read.
    if primary_geom_col not in primary_df.columns:
        logger.error(
            "execute_spatial_join failed: geometry column '{}' not found in primary DataFrame.",
            primary_geom_col,
        )
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

    if len(primary_geoms) == 0:
        logger.debug("execute_spatial_join: primary_df is empty; returning early.")
        for col in agg.keys():
            primary_df = primary_df.with_columns(
                pl.lit(None).alias(f"{other_entity_name}_{col}")
            )
        return primary_df

    # Compute bounding box of all primary geometries
    bnds = shapely.bounds(primary_geoms)
    valid_bnds = bnds[~np.isnan(bnds).any(axis=1)]
    if len(valid_bnds) > 0:
        minx, miny, maxx, maxy = (
            valid_bnds[:, 0].min(),
            valid_bnds[:, 1].min(),
            valid_bnds[:, 2].max(),
            valid_bnds[:, 3].max(),
        )
        primary_total_box = shapely.box(minx, miny, maxx, maxy)
    else:
        primary_total_box = None

    other_profile = other_entity.schema_profile
    other_static = other_entity._resolve(other_entity.static_path)
    other_geom_col = other_profile.geometry_col
    other_geom_type = other_profile.geometry_type
    other_key_cols = other_entity.key_cols

    agg_col_names = [c for c in agg if c not in ("count", "area")]
    other_cols = list(dict.fromkeys(other_key_cols + [other_geom_col] + agg_col_names))

    if (
        primary_total_box is not None
        and getattr(other_entity, "_index", None) is not None
    ):
        # Pre-filter other_entity using its in-memory bounding box index
        candidates = bbox_intersects_geometry(other_entity._index, primary_total_box)
        if candidates.is_empty():
            other_df = pl.DataFrame(schema={c: pl.Utf8 for c in other_cols})
        else:
            other_df = (
                pl.scan_parquet(other_static)
                .select(other_cols)
                .join(
                    candidates.select(other_key_cols).lazy(),
                    on=other_key_cols,
                    how="semi",
                )
                .collect()
            )
    else:
        other_df = pl.read_parquet(other_static, columns=other_cols)

    if other_df.is_empty():
        other_geoms = np.array([])
        other_tree = shapely.STRtree([])
    else:
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
    if agg_cols_only and not other_df.is_empty():
        other_df_np = other_df.select(agg_cols_only).to_pandas()
    else:
        other_df_np = None

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


def point_in_entities(
    latlon: dict[str, float],
    entities: list[Any],
) -> dict[str, Any | None]:
    """Return the entity-id in which a lat/lon point falls, per entity.

    For each entity, performs a two-phase lookup:

    1. **Bbox pre-filter** — eliminates entities whose bounding box doesn't
       contain the point using the in-memory ``_index`` (no I/O).
    2. **Exact containment** — loads geometry only for bbox candidates and
       runs a vectorised Shapely ``contains`` test.

    The function is designed for repeated hot-path calls; it relies on
    ``entity._index`` and ``entity.geometry_lazy`` which are both
    process-level cached after first access.

    Args:
        latlon (dict[str, float]): Point coordinates with keys ``"lat"`` and
            ``"lon"`` (or ``"lng"`` as an alias for longitude).
        entities (list[BaseEntity]): List of entity instances to test against.
            Each must expose ``_index``, ``key_cols``, ``schema_profile``,
            ``_resolve``, and ``static_path``.

    Returns:
        dict[str, Any | None]: Mapping of ``entity_class_name → entity_id``
        (the value of the first ``key_col``). ``None`` if the point doesn't
        fall inside any geometry of that entity.

    Raises:
        KeyError: If neither ``"lon"`` nor ``"lng"`` key exists in ``latlon``.

    Example::

        result = point_in_entities(
            {"lat": 28.6139, "lon": 77.2090},
            [district_entity, block_entity],
        )
        # {"DistrictEntity": "DEL_001", "BlockEntity": None}

    """
    lat: float = latlon["lat"]
    lon: float = latlon.get("lon", latlon.get("lng"))  # type: ignore[assignment]

    # Shapely Point — created once, reused across all entities.
    pt = shapely.Point(lon, lat)

    result: dict[str, Any | None] = {}

    for entity in entities:
        entity_name = type(entity).__name__

        # Phase 1: bbox pre-filter (pure in-memory, no I/O) ---------------
        index_df = (
            entity._index
        )  # pl.DataFrame with (key_cols…, minx, miny, maxx, maxy)
        candidates = index_df.filter(
            (pl.col("minx") <= lon)
            & (pl.col("maxx") >= lon)
            & (pl.col("miny") <= lat)
            & (pl.col("maxy") >= lat)
        )

        if candidates.is_empty():
            result[entity_name] = None
            continue

        # Phase 2: exact containment (load only candidate geometries) ------
        profile = entity.schema_profile
        geom_col = profile.geometry_col
        geom_type = profile.geometry_type
        key_cols: list[str] = entity.key_cols
        static_path: str = entity._resolve(entity.static_path)

        geom_df = (
            pl.scan_parquet(parquet_scan_path(static_path))
            .select(key_cols + [geom_col])
            .join(candidates.select(key_cols).lazy(), on=key_cols, how="semi")
            .collect()
        )

        if geom_df.is_empty():
            result[entity_name] = None
            continue

        raw = geom_df[geom_col].to_numpy()
        if geom_type == "wkb":
            geoms = shapely.from_wkb(raw)
        else:
            geoms = shapely.from_wkt(raw)

        # Vectorised contains — after bbox pre-filter candidates are O(1~10),
        # so tree construction would cost more than the query itself.
        mask = shapely.contains(geoms, pt)  # C-level, no Python loop
        hit_indices = np.where(mask)[0]

        if len(hit_indices) == 0:
            result[entity_name] = None
        else:
            # Return first key_col value of the first match.
            primary_key = key_cols[0]
            result[entity_name] = geom_df[primary_key][int(hit_indices[0])]

    return result
