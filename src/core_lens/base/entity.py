"""Base entity contract for core_lens.

All entities — built-in or plugin — must subclass :class:`BaseEntity` and
implement its abstract interface.  Concrete implementations live in
``core_lens.entities.*`` (built-ins) or in third-party packages (plugins).

Plugin authors import from the public surface::

    from core_lens.base import BaseEntity
"""

from __future__ import annotations

import functools
import pathlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from core_lens.utils.paths import (
    is_cloud_uri,
    join_uri,
    path_exists,
)
from core_lens.utils.spatial import (
    bbox_intersects_geometry,
    build_bbox_index,
    exact_spatial_filter,
)
from core_lens.utils.polars_utils import cached_read_schema

if TYPE_CHECKING:
    import shapely
    from core_lens.base.view import View
    from core_lens.schema.profile import SchemaProfile


# ---------------------------------------------------------------------------
# Process-level caches — shared across all BaseEntity instances in a process.
#
# Both caches are backed by functools.cache (lru_cache with maxsize=None).
# maxsize=None means unbounded — intentional: evicting an index or schema
# profile forces an expensive rebuild on the next access.
#
# _cached_detect       eliminates repeated detect() calls (OPT-2).
# _cached_build_index  eliminates repeated build_bbox_index() calls (OPT-1).
#
# Cache diagnostics:  _cached_detect.cache_info() / _cached_build_index.cache_info()
# Cache reset (tests): _cached_detect.cache_clear() / _cached_build_index.cache_clear()
# ---------------------------------------------------------------------------


def _so_key(storage_options: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Convert a ``storage_options`` dict to a hashable, order-stable tuple.

    All args passed to :func:`_cached_detect` and :func:`_cached_build_index`
    must be hashable (``functools.cache`` requirement).  ``storage_options``
    is a plain dict, so we convert it here before passing to the cached
    functions.  Different cloud credentials for the same path produce
    different cache keys.

    Args:
        storage_options (dict[str, Any]): Cloud credential / configuration dict.

    Returns:
        tuple[tuple[str, Any], ...]: A sorted tuple of ``(key, value)`` pairs.

    """
    return tuple(sorted(storage_options.items()))


@functools.cache
def _cached_detect(
    static_path: str,
    key_cols: tuple[str, ...],
    geometry_col: str,
    annual_path: str | None,
    fortnightly_path: str | None,
    storage_options_key: tuple[tuple[str, Any], ...],
) -> "SchemaProfile":
    """Cached wrapper around :func:`~core_lens.schema.detection.detect`.

    All arguments must be hashable so they can form the cache key.  Pass
    ``storage_options`` through :func:`_so_key` before calling this function.

    Returns:
        SchemaProfile: Detected schema profile for the given entity paths.

    """
    from core_lens.schema.detection import detect

    return detect(
        static_path=static_path,
        key_cols=list(key_cols),
        geometry_col=geometry_col,
        annual_path=annual_path,
        fortnightly_path=fortnightly_path,
        storage_options=dict(storage_options_key) or None,
    )


@functools.cache
def _cached_build_index(
    static_path: str,
    key_cols: tuple[str, ...],
    bbox_cols: tuple[str, str, str, str] | None,
    geometry_col: str,
    geometry_type: str,
    storage_options_key: tuple[tuple[str, Any], ...],
) -> pl.DataFrame:
    """Cached wrapper around :func:`~core_lens.utils.spatial.build_bbox_index`.

    All arguments must be hashable so they can form the cache key.  Pass
    ``storage_options`` through :func:`_so_key` before calling this function.

    Returns:
        pl.DataFrame: The bounding-box index DataFrame.

    """
    logger.debug("Building bbox index for {} (cache miss)", static_path)
    return build_bbox_index(
        static_path=static_path,
        key_cols=list(key_cols),
        bbox_cols=bbox_cols,
        geometry_col=geometry_col,
        geometry_type=geometry_type,
        storage_options=dict(storage_options_key) or None,
    )


class BaseEntity(ABC):
    r"""Abstract base class for every entity in the core_lens plugin system.

    An entity represents a geospatial primitive (e.g. microwatershed, village,
    district) backed by one or more Parquet/GeoParquet files.  Entities are
    *descriptors* — they carry path and schema metadata but hold no per-row
    state themselves.  Row-level data lives in :class:`~core_lens.base.view.View`
    and :class:`~core_lens.base.result.Result`.

    Subclasses **must** implement:

    * :attr:`key_cols`      — column(s) that uniquely identify one entity instance
    * :attr:`geometry_col`  — geometry column name in the static GeoParquet file
    * :attr:`static_path`   — path to the static GeoParquet file (mandatory).
                              May be relative; resolved against ``data_root`` when
                              the entity is instantiated by :class:`~core_lens.aoi.AoI`.

    Subclasses **may** override:

    * :attr:`annual_path`       — path to the annual time-series Parquet file
    * :attr:`fortnightly_path`  — path to the fortnightly time-series Parquet file
    * :attr:`schema_profile`    — override auto-detection by returning an
                                  explicit :class:`~core_lens.schema.profile.SchemaProfile`

    ``where``, ``spatial_filter``, ``spatial_join``, and ``schema_profile`` are
    all implemented on this base class.  Subclasses only need to declare paths
    and keys.

    Plugin example::

        from core_lens.base import BaseEntity

        class ForestEntity(BaseEntity):
            key_cols     = [\"forest_patch_id\"]
            geometry_col = \"geometry\"
            static_path  = \"forest/static.geoparquet\"  # relative to AoI data_root
            annual_path  = \"forest/annual.parquet\"

        AoI.register(ForestEntity)

    Validation rules enforced at :meth:`AoI` instantiation time (relative paths)
    or at :meth:`AoI.register` time (absolute paths):

    1. ``static_path`` exists and is readable.
    2. ``key_cols`` are present and unique in the static file.
    3. ``geometry_col`` is present and contains valid geometries.
    4. ``annual_path`` and ``fortnightly_path`` exist if declared.

    Any failure raises :class:`~core_lens.base.entity.EntityValidationError`.
    """

    def __init__(
        self,
        data_root: str | pathlib.Path | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialise the entity with an optional data root directory.

        Args:
            data_root (str | pathlib.Path | None, optional): Root data directory or cloud
                URI prefix (e.g. ``"s3://bucket/data"``).  Relative
                :attr:`static_path`, :attr:`annual_path`, and
                :attr:`fortnightly_path` values are resolved against this
                root.  Defaults to ``None`` (resolved against cwd).
            storage_options (dict[str, Any] | None, optional): Cloud credential /
                configuration options forwarded to ``pyarrow.fs`` and
                ``polars.scan_parquet``.  For S3 the common keys are
                ``"region"``, ``"access_key"``, and ``"secret_key"``.
                ``None`` uses ambient credentials (env-vars / ``~/.aws/``).

        """
        # Store as str so cloud URIs (s3://…) are never coerced through pathlib.
        self._data_root: str | None = str(data_root) if data_root is not None else None
        self._storage_options: dict[str, Any] = storage_options or {}

    def _resolve(self, path: str) -> str:
        """Return an absolute path or cloud URI string for *path*.

        Relative paths are resolved against :attr:`_data_root` (which may be a
        cloud URI such as ``s3://bucket/prefix``) using :func:`~core_lens.utils.paths.join_uri`.
        Absolute local paths and fully-qualified cloud URIs are returned as-is.

        **Cloud paths**: existence is *not* checked here to avoid unnecessary
        remote ``HeadObject`` calls on every resolve.  Validation happens once
        inside :func:`~core_lens.aoi._validate_entity` via the schema-read path.

        Args:
            path (str): A filesystem path (absolute or relative) or a cloud URI.

        Returns:
            str: An absolute path or cloud URI string.

        Raises:
            FileNotFoundError: For *local* paths that do not exist after resolution.

        """
        import pathlib as _pathlib

        # Fully-qualified cloud URI — return as-is.
        if is_cloud_uri(path):
            if not path.endswith("/") and not _pathlib.Path(path).suffix:
                return path + "/"
            return path

        # Relative path — join against data_root (which may itself be a cloud URI).
        p = _pathlib.Path(path)
        if not p.is_absolute():
            root = (
                self._data_root
                if self._data_root is not None
                else str(_pathlib.Path.cwd())
            )
            resolved = join_uri(root, path)
        else:
            resolved = str(p)

        # Polars cloud directory scanning requires a trailing slash.
        if is_cloud_uri(resolved) and not resolved.endswith("/") and not p.suffix:
            resolved += "/"

        # For local resolved paths only: eager existence check.
        if not is_cloud_uri(resolved) and not path_exists(resolved):
            if path_exists(resolved + ".parquet"):
                resolved += ".parquet"
            else:
                logger.error(
                    "Path resolution failed: '{}' (resolved to {}) does not exist.",
                    path,
                    resolved,
                )
                raise FileNotFoundError(
                    f"Entity path {path!r} (resolved to {resolved!r}) does not exist. "
                    "Provide an absolute path or ensure the file exists relative to "
                    "the AoI data_root directory."
                )
        return resolved

    @property
    @abstractmethod
    def key_cols(self) -> list[str]:
        r"""Columns that uniquely identify one instance of this entity.

        For built-in entities this is always a single-element list (e.g.
        ``[\"mws_id\"]``), but the contract allows composite keys for plugins.

        Returns:
            list[str]: A list of column name strings present in the static file.

        """

    @property
    @abstractmethod
    def geometry_col(self) -> str:
        """Name of the geometry column in the static GeoParquet file.

        The column must contain a geometry type understood by GeoPandas
        (WKB bytes, WKT string, or a native geometry column).

        Returns:
            str: The column name as a string.

        """

    @property
    @abstractmethod
    def static_path(self) -> str:
        """Absolute filesystem path to the static GeoParquet file.

        The path must be absolute.  If a relative path is provided it is
        resolved against the current working directory at first use.  A
        ``FileNotFoundError`` is raised if the file does not exist.

        Returns:
            str: A path string.

        """

    @property
    def annual_path(self) -> str | None:
        """Path to the annual time-series Parquet file, or ``None``.

        Override in subclasses that carry annual temporal data.  If declared,
        the file must exist at :meth:`AoI.register` time or
        :class:`EntityValidationError` is raised.

        Returns:
            str | None: A path string, or ``None`` if the entity has no annual data.

        """
        return None

    @property
    def fortnightly_path(self) -> str | None:
        """Path to the fortnightly time-series Parquet file, or ``None``.

        Override in subclasses that carry fortnightly temporal data.  If
        declared, the file must exist at :meth:`AoI.register` time or
        :class:`EntityValidationError` is raised.

        Returns:
            str | None: A path string, or ``None`` if the entity has no fortnightly data.

        """
        return None

    @property
    def schema_profile(self) -> "SchemaProfile":
        """Validated schema descriptor for this entity's data files.

        Auto-detected from Parquet file metadata on first access.  The result
        is cached at two levels:

        1. **Instance level** — ``_schema_profile`` attribute set on ``self``
           so that repeated property access within the same instance is a bare
           attribute lookup with no dict overhead.
        2. **Process level** — :func:`_cached_detect` (backed by
           ``functools.cache``) keyed on the resolved paths and entity
           configuration.  Different ``AoI()`` calls that point at the same
           data directory share one ``SchemaProfile`` object, avoiding
           redundant ``collect_schema()`` I/O.

        Override in subclasses to provide an explicit profile instead of
        relying on detection.

        Returns:
            SchemaProfile: A fully-validated :class:`~core_lens.schema.profile.SchemaProfile`.

        """
        if not hasattr(self, "_schema_profile"):
            _so = self._storage_options or {}
            self._schema_profile: SchemaProfile = _cached_detect(
                static_path=self._resolve(self.static_path),
                key_cols=tuple(self.key_cols),
                geometry_col=self.geometry_col,
                annual_path=(
                    self._resolve(self.annual_path)
                    if self.annual_path is not None
                    else None
                ),
                fortnightly_path=(
                    self._resolve(self.fortnightly_path)
                    if self.fortnightly_path is not None
                    else None
                ),
                storage_options_key=_so_key(_so),
            )
        return self._schema_profile

    @property
    def geometry_lazy(self) -> pl.LazyFrame:
        """A cached LazyFrame representing the geometry of this entity.

        Contains only the key columns and the geometry column. Used to avoid
        repeatedly scanning and selecting geometries in downstream operations.
        """
        if not hasattr(self, "_geometry_lazy"):
            self._geometry_lazy = pl.scan_parquet(
                self._resolve(self.static_path),
                storage_options=self._storage_options or None,
            ).select(self.key_cols + [self.schema_profile.geometry_col])
        return self._geometry_lazy

    @property
    def _index(self) -> pl.DataFrame:
        """In-memory bounding-box index for spatial pre-filtering.

        Cached at two levels (same strategy as :attr:`schema_profile`):

        1. **Instance level** — ``_cached_index`` attribute on ``self``.
        2. **Process level** — :func:`_cached_build_index` (backed by
           ``functools.cache``) keyed on the resolved static path, key
           columns, and bbox/geometry schema.  All entity instances that
           resolve to the same static file share one index ``pl.DataFrame``,
           eliminating repeated geometry decodes.

        Returns:
            pl.DataFrame: DataFrame with columns ``(*key_cols, minx, miny, maxx, maxy)``.

        """
        if not hasattr(self, "_cached_index"):
            profile = self.schema_profile  # already cached — no extra I/O
            _so = self._storage_options or {}
            self._cached_index: pl.DataFrame = _cached_build_index(
                static_path=self._resolve(self.static_path),
                key_cols=tuple(self.key_cols),
                bbox_cols=profile.bbox_cols,
                geometry_col=profile.geometry_col,
                geometry_type=profile.geometry_type,
                storage_options_key=_so_key(_so),
            )
        return self._cached_index

    def where(self, **kwargs: Any) -> "View":
        """Return a lazy :class:`~core_lens.base.view.View` filtered by attributes.

        Each keyword argument is interpreted **attribute-first**: if the kwarg
        key exists as a column in the static file the filter is applied
        directly.  If a kwarg key does *not* exist as a column it is resolved
        as a registered entity name and the matching entity's geometry is used
        for a spatial filter (e.g. ``district="Shimla"`` finds all MWS whose
        centroid falls within Shimla district).

        Multiple attribute kwargs are AND-ed.  Multiple spatial-entity kwargs
        are AND-ed via sequential spatial filters.

        Args:
            **kwargs: Column–value pairs to filter on.  Unknown column names
                are resolved as entity-name lookups.

        Returns:
            View: A lazy :class:`~core_lens.base.view.View` with resolved key pairs.

        Raises:
            ValueError: If a kwarg cannot be resolved as either an attribute
                column or a registered entity name.

        """
        from core_lens.aoi import _REGISTRY
        from core_lens.base.view import View

        logger.debug(
            "Applying attribute filters on {}: {}", self.__class__.__name__, kwargs
        )

        static = self._resolve(self.static_path)
        schema = cached_read_schema(static, self._storage_options)

        attr_kwargs = {k: v for k, v in kwargs.items() if k in schema}
        entity_kwargs = {k: v for k, v in kwargs.items() if k not in schema}

        # Validate entity-kwargs early so we give a useful error message.
        for k in entity_kwargs:
            if k not in _REGISTRY:
                logger.error(
                    "BaseEntity.where failed: '{}' is neither a column nor a registered entity.",
                    k,
                )
                raise ValueError(
                    f"BaseEntity.where: {k!r} is not a column in {self.static_path!r} "
                    f"and is not a registered entity name. "
                    f"Registered entities: {sorted(_REGISTRY)}. "
                    f"Available columns: {sorted(schema)}."
                )

        # --- Combine attribute and spatial filters into a single lazy pipeline ---
        lf = pl.scan_parquet(static)

        if attr_kwargs:
            filter_expr = pl.lit(True)
            for col, val in attr_kwargs.items():
                dtype = schema[col]
                is_list_col = isinstance(dtype, pl.List) or dtype == pl.List
                is_val_list = isinstance(val, (list, tuple, set))

                if is_list_col:
                    if is_val_list:
                        filter_expr = filter_expr & (
                            pl.col(col).list.set_intersection(list(val)).list.len() > 0
                        )
                    else:
                        filter_expr = filter_expr & pl.col(col).list.contains(val)
                else:
                    if is_val_list:
                        filter_expr = filter_expr & pl.col(col).is_in(list(val))
                    else:
                        filter_expr = filter_expr & (pl.col(col) == val)
            lf = lf.filter(filter_expr)

        if not entity_kwargs:
            keys = lf.select(self.key_cols).collect()
        else:
            import shapely
            import shapely.wkb as swkb
            import shapely.wkt as swkt
            import shapely.ops as sops
            import numpy as np
            from core_lens.utils.spatial import bbox_intersects_geometry

            lookup_geoms = []
            for entity_kwarg_name, entity_kwarg_val in entity_kwargs.items():
                other_entity = _REGISTRY[entity_kwarg_name](data_root=self._data_root)
                other_profile = other_entity.schema_profile
                other_static = other_entity._resolve(other_entity.static_path)

                other_lf = pl.scan_parquet(other_static)
                match_expr = pl.lit(False)
                other_schema = cached_read_schema(
                    other_static, other_entity._storage_options
                )
                for col in other_entity.key_cols + list(
                    other_profile.extra_static_cols
                ):
                    if col in other_schema:
                        match_expr = match_expr | (pl.col(col) == entity_kwarg_val)
                matched = (
                    other_lf.filter(match_expr)
                    .select([other_profile.geometry_col])
                    .collect()
                )

                if matched.is_empty():
                    logger.error(
                        "BaseEntity.where failed: No rows matched {}={!r} in {}",
                        entity_kwarg_name,
                        entity_kwarg_val,
                        other_entity.static_path,
                    )
                    raise ValueError(
                        f"BaseEntity.where: No rows matched {entity_kwarg_name}={entity_kwarg_val!r} "
                        f"in {other_entity.static_path!r}."
                    )

                raw_geoms = matched[other_profile.geometry_col].to_list()
                if other_profile.geometry_type == "wkb":
                    geoms = [swkb.loads(v) for v in raw_geoms]
                else:
                    geoms = [swkt.loads(v) for v in raw_geoms]
                lookup_geom = sops.unary_union(geoms) if len(geoms) > 1 else geoms[0]
                lookup_geoms.append(lookup_geom)

            candidates = self._index
            for geom in lookup_geoms:
                candidates = bbox_intersects_geometry(candidates, geom)

            if candidates.is_empty():
                keys = candidates.select(self.key_cols)
            else:
                lf = lf.select(self.key_cols + [self.schema_profile.geometry_col]).join(
                    candidates.select(self.key_cols).lazy(),
                    on=self.key_cols,
                    how="semi",
                )
                df = lf.collect()

                if df.is_empty():
                    keys = df.select(self.key_cols)
                else:
                    geom_array = df[self.schema_profile.geometry_col].to_numpy()
                    if self.schema_profile.geometry_type == "wkb":
                        decoded_geoms = shapely.from_wkb(geom_array)
                    else:
                        decoded_geoms = shapely.from_wkt(geom_array)

                    centroids = shapely.centroid(decoded_geoms)
                    mask = np.ones(len(centroids), dtype=bool)
                    for geom in lookup_geoms:
                        mask = mask & shapely.contains(geom, centroids)

                    keys = df.filter(mask).select(self.key_cols)

        entity_name = _entity_name(type(self))
        return View(keys=keys, entity=self, entity_name=entity_name)

    def spatial_filter(
        self,
        geometry: "shapely.Geometry | None" = None,
        bbox: tuple[float, float, float, float] | None = None,
        relationship: str = "centroid",
        threshold: float = 0.5,
    ) -> "View":
        """Return a lazy :class:`~core_lens.base.view.View` filtered by geometry.

        Uses the in-memory bbox index for a fast rectangular pre-filter, then
        refines with a Shapely STRtree exact-relationship check.

        Args:
            geometry (shapely.Geometry | None, optional): A Shapely geometry representing the spatial extent.
            bbox (tuple[float, float, float, float] | None, optional): Bounding box as ``(minx, miny, maxx, maxy)`` in WGS-84.
                Converted to a ``shapely.geometry.box`` internally.
            relationship (str, optional): Spatial relationship mode.

                * ``"centroid"`` (default) — entity centroid must lie within
                  the geometry.
                * ``"area"`` — intersection area / entity area must exceed
                  ``threshold``.

            threshold (float, optional): Area coverage threshold for ``"area"`` mode (0–1).
                Default 0.5.

        Returns:
            View: A lazy :class:`~core_lens.base.view.View` scoped to the given
            spatial extent.

        Raises:
            ValueError: If neither ``geometry`` nor ``bbox`` is provided.

        """
        import shapely.geometry as sgeom

        from core_lens.base.view import View

        if geometry is None and bbox is None:
            logger.error(
                "BaseEntity.spatial_filter failed: neither 'geometry' nor 'bbox' provided."
            )
            raise ValueError(
                "spatial_filter() requires either 'geometry' or 'bbox' to be provided."
            )
        if bbox is not None and geometry is None:
            geometry = sgeom.box(*bbox)

        assert geometry is not None  # guaranteed by the guards above

        logger.debug(
            "Applying spatial filter on {} (relationship='{}')",
            self.__class__.__name__,
            relationship,
        )

        profile = self.schema_profile
        candidates = bbox_intersects_geometry(self._index, geometry)
        keys = exact_spatial_filter(
            candidates=candidates,
            static_path=self._resolve(self.static_path),
            key_cols=self.key_cols,
            geometry_col=profile.geometry_col,
            geometry_type=profile.geometry_type,
            aoi_geometry=geometry,
            relationship=relationship,
            threshold=threshold,
        )

        entity_name = _entity_name(type(self))
        return View(keys=keys, entity=self, entity_name=entity_name)

    def spatial_join(self, other: "BaseEntity", agg: dict[str, str]) -> "View":
        r"""Return a lazy :class:`~core_lens.base.view.View` with a cross-entity join pending.

        The join is recorded in the View's ``join_spec`` and computed only at
        materialisation time (``.static``, ``.annual``, or ``.fortnightly``).
        Joined columns are namespaced as ``{entity_name}_{column_name}``.

        Args:
            other (BaseEntity): The secondary :class:`BaseEntity` whose columns will be
                joined and aggregated onto ``self``.
            agg (dict[str, str]): Mapping of ``{column: aggregation}`` specifying which columns
                from ``other`` to bring in and how to aggregate them.  Valid
                aggregation strings are ``\"area\"``, ``\"count\"``, ``\"mean\"``,
                ``\"sum\"``, ``\"min\"``, and ``\"max\"``.

        Returns:
            View: A lazy :class:`~core_lens.base.view.View` with the join spec
            recorded for deferred execution.

        """
        from core_lens.base.view import View

        logger.debug(
            "Deferred spatial join registered for {} with {}",
            self.__class__.__name__,
            other.__class__.__name__,
        )

        entity_name = _entity_name(type(self))
        join_spec = {"other": other, "agg": agg}
        return View(
            keys=self._index.select(self.key_cols),
            entity=self,
            entity_name=entity_name,
            join_spec=join_spec,
        )


class EntityValidationError(Exception):
    """Raised when an entity fails validation at :meth:`AoI.register` time.

    The message will describe exactly which check failed (missing file, absent
    key column, invalid geometry column, etc.) to give plugin authors
    actionable feedback.
    """


def _entity_name(entity_cls: type[BaseEntity]) -> str:
    name = entity_cls.__name__
    if name.endswith("Entity"):
        name = name[: -len("Entity")]
    return name.lower()
