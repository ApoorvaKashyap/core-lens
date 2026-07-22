"""AoI (Area of Interest) — primary entry point for core_lens."""

from __future__ import annotations

import functools
import os
import pathlib
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core_lens.base.result import Result

import polars as pl
import shapely
import shapely.geometry as sgeom
import shapely.ops as sops
from loguru import logger

from core_lens.base.entity import BaseEntity, EntityValidationError
from core_lens.utils.paths import is_cloud_uri
from core_lens.utils.polars_utils import cached_read_schema

if TYPE_CHECKING:
    from core_lens.base.view import View


@dataclass(frozen=True)
class SeasonConfig:
    """Date-range definitions for the three Indian crop seasons.

    Each season is a ``(MM-DD, MM-DD)`` inclusive range.  Seasons that cross
    the calendar year-end (e.g. rabi: Nov → Mar) are handled by comparing
    the month-day portion of a date against each range, rolling over where
    necessary.

    The library ships with agronomic defaults for the Indo-Gangetic plain.
    Override at AoI construction time to match a different agro-climatic zone::

        aoi = AoI("data/", district="Dharwad", seasons=SeasonConfig(
            kharif=("06-01", "10-15"),
            rabi=("10-16", "02-28"),
            zaid=("03-01", "05-31"),
        ))

    Attributes:
        kharif: Kharif (monsoon) season range as ``(start_MM-DD, end_MM-DD)``.
        rabi: Rabi (winter) season range.
        zaid: Zaid (summer) season range.

    """

    kharif: tuple[str, str] = ("07-01", "10-31")
    rabi: tuple[str, str] = ("11-01", "03-31")
    zaid: tuple[str, str] = ("04-01", "06-30")

    def __post_init__(self) -> None:
        """Initialize after dataclass creation.

        Validates that all configured seasons have valid MM-DD ranges.

        Raises:
            ValueError: If a season date format is invalid.
        """
        from datetime import datetime

        for attr in ("kharif", "rabi", "zaid"):
            start, end = getattr(self, attr)
            try:
                # Use a leap year (2000) to allow "02-29"
                datetime.strptime(f"2000-{start}", "%Y-%m-%d")
                datetime.strptime(f"2000-{end}", "%Y-%m-%d")
            except ValueError as e:
                raise ValueError(
                    f"SeasonConfig: Invalid date format for {attr}=('{start}', '{end}'). "
                    f"Expected valid 'MM-DD' strings. Original error: {e}"
                ) from e

    def season_for(self, d: date) -> str:
        """Return the season name for a given date.

        Args:
            d (date): The date to classify.

        Returns:
            str: ``"kharif"``, ``"rabi"``, or ``"zaid"``.

        """
        md = f"{d.month:02d}-{d.day:02d}"
        for name in ("kharif", "rabi", "zaid"):
            start, end = getattr(self, name)
            if _md_in_range(md, start, end):
                return name
        # A date that falls in no season (can happen if ranges have gaps) is
        # assigned to the nearest season.  In practice the defaults cover the
        # whole year, so this branch should never fire with library defaults.
        raise ValueError(
            f"Date {d} does not fall within any configured season. "
            "Ensure the three SeasonConfig ranges together cover the full year."
        )


def _md_in_range(md: str, start: str, end: str) -> bool:
    """Return True if a MM-DD string falls within [start, end], handling year rollover."""
    if start <= end:
        return start <= md <= end
    # Year-crossing range (e.g. rabi: 11-01 to 03-31)
    return md >= start or md <= end


# Singleton for the default SeasonConfig — constructed once per process.
# SeasonConfig.__post_init__ runs 6× datetime.strptime calls for validation;
# sharing the default instance eliminates that cost for every AoI() call
# that does not supply a custom season config.
_DEFAULT_SEASON_CONFIG: SeasonConfig | None = None


def _default_season_config() -> SeasonConfig:
    """Return the shared default :class:`SeasonConfig` instance.

    Constructed once per process and reused thereafter.  Callers that need
    a custom config should construct their own :class:`SeasonConfig` and
    pass it to :class:`AoI`.

    Returns:
        SeasonConfig: The shared default instance.

    """
    global _DEFAULT_SEASON_CONFIG
    if _DEFAULT_SEASON_CONFIG is None:
        _DEFAULT_SEASON_CONFIG = SeasonConfig()
    return _DEFAULT_SEASON_CONFIG


# Class-level registry: entity name → entity **class**.
# Shared across all AoI instances in a process.  Explicit registration is
# required; there is no auto-discovery (design §6.1).
_REGISTRY: dict[str, type[BaseEntity]] = {}


# ---------------------------------------------------------------------------
# Process-level boundary cache — shared across all AoI instances.
#
# Keyed on (data_root, storage_options_key, entity_kwargs_key).
# Returns a (geometry_wkb, entity_name, key_rows_tuple) triple so the
# expensive scan + WKB decode + unary_union is only paid once per process
# per unique boundary specification.
#
# Cache diagnostics: _cached_resolve_boundary.cache_info()
# Cache reset (tests): _cached_resolve_boundary.cache_clear()
# ---------------------------------------------------------------------------


@functools.cache
def _cached_resolve_boundary(
    data_root: str,
    storage_options_key: tuple[tuple[str, Any], ...],
    entity_kwargs_key: tuple[tuple[str, Any], ...],
) -> tuple[bytes, str, tuple[tuple[Any, ...], ...]]:
    """Return cached boundary resolution: ``(geometry_wkb, entity_name, key_rows)``.

    All arguments must be hashable.  ``entity_kwargs`` must be converted to a
    sorted tuple of ``(key, value)`` pairs before calling this function.

    The returned triple contains:

    * **geometry_wkb** — WKB bytes of the unioned boundary geometry.
    * **entity_name** — name of the entity that defined the boundary
      (e.g. ``"mws"`` or ``"tehsil"``).
    * **key_rows** — tuple of key-value tuples for the matched rows
      (e.g. ``(("mws_id", "13_551"),)`` for a single-row result).

    Args:
        data_root (str): Resolved data root path or cloud URI.
        storage_options_key (tuple): Sorted ``storage_options`` items (hashable).
        entity_kwargs_key (tuple): Sorted ``entity_kwargs`` items (hashable).

    Returns:
        tuple: ``(geometry_wkb, entity_name, key_rows)``.

    """
    from core_lens.base.entity import _entity_name as _ename

    entity_kwargs: dict[str, Any] = dict(entity_kwargs_key)
    storage_options: dict[str, Any] = dict(storage_options_key)

    # Instantiate a throw-away AoI-like context to run the resolution logic.
    # We cannot use a live AoI instance here (would be circular), so we
    # inline the logic from _resolve_named_boundary.
    candidate: BaseEntity | None = None

    # First pass: find entity whose schema columns match the kwargs.
    for ename in _REGISTRY:
        entity_cls = _REGISTRY[ename]
        entity = entity_cls(
            data_root=data_root,
            storage_options=storage_options or None,
        )
        schema = entity.schema_profile  # uses _cached_detect — I/O only once
        if any(
            k in schema.key_cols or k in schema.extra_static_cols for k in entity_kwargs
        ):
            candidate = entity
            break

    # Fallback: entity name matches a kwarg key.
    if candidate is None:
        for ename in _REGISTRY:
            if ename in entity_kwargs:
                entity_cls = _REGISTRY[ename]
                candidate = entity_cls(
                    data_root=data_root,
                    storage_options=storage_options or None,
                )
                break

    if candidate is None:
        raise EntityValidationError(
            f"No registered entity can satisfy the filters {entity_kwargs}. "
            f"Registered entities: {sorted(_REGISTRY)}."
        )

    schema = candidate.schema_profile
    geom_col = schema.geometry_col
    static_path = candidate._resolve(candidate.static_path)

    lf = pl.scan_parquet(static_path, storage_options=storage_options or None)
    filter_expr = pl.lit(True)
    schema_types = cached_read_schema(static_path, storage_options)
    for col, val in entity_kwargs.items():
        if col in schema.key_cols or col in schema.extra_static_cols:
            is_list_col = isinstance(schema_types.get(col), pl.List)
            if isinstance(val, list):
                if is_list_col:
                    filter_expr = (
                        filter_expr
                        & pl.col(col).list.eval(pl.element().is_in(val)).list.any()
                    )
                else:
                    filter_expr = filter_expr & pl.col(col).is_in(val)
            else:
                if is_list_col:
                    filter_expr = filter_expr & pl.col(col).list.contains(val)
                else:
                    filter_expr = filter_expr & (pl.col(col) == val)

    df = lf.filter(filter_expr).select(candidate.key_cols + [geom_col]).collect()

    if df.is_empty():
        raise ValueError(
            f"No rows matched the filters {entity_kwargs} in {candidate.static_path!r}."
        )

    geoms = shapely.from_wkb(df[geom_col].to_numpy())
    unified = sops.unary_union(geoms) if len(geoms) > 1 else geoms[0]
    geom_wkb: bytes = shapely.to_wkb(unified)

    entity_name = _ename(type(candidate))
    # Serialise key DataFrame as a tuple-of-tuples for the cache return value.
    key_rows = tuple(
        tuple(row[c] for c in candidate.key_cols)
        for row in df.select(candidate.key_cols).to_dicts()
    )

    return geom_wkb, entity_name, key_rows


class AoI:
    """Area of Interest — the primary entry point for querying geospatial data.

    An ``AoI`` is two things simultaneously:

    1. **A geometry** — the resolved boundary of the named administrative unit
       (or a raw bbox / Shapely polygon), stored as :attr:`geometry`.
    2. **A collection of scoped entities** — every registered entity
       pre-filtered to instances that fall within :attr:`geometry`.  Accessed
       as attributes: ``aoi.mws``, ``aoi.village``, ``aoi.forest``, etc.

    ``AoI`` holds no data itself.  Entity attributes are lazy
    :class:`~core_lens.base.view.View` objects; no Parquet I/O occurs until a
    materialisation property (``.static``, ``.annual``, ``.sub_annual``) is
    accessed on a View.

    **Registration** must happen before any ``AoI`` is constructed::

        from core_lens import AoI, MWSEntity, TehsilEntity

        AoI.register(MWSEntity)
        AoI.register(TehsilEntity)

    **Initialisation** — exactly one boundary argument is required::

        aoi = AoI("data/", tehsil="Pangi", district="Chamba", state="Himachal Pradesh")
        aoi = AoI("data/", bbox=(minx, miny, maxx, maxy))
        aoi = AoI("data/", geometry=some_shapely_polygon)
        aoi = AoI("data/", village="Shiroor")
        aoi = AoI("data/", mws_id="13_551")

    **Entity access**::

        aoi.mws        # View — all MWS within the AoI boundary
        aoi.tehsil     # View — all tehsils within the AoI boundary
        aoi.forest     # View — plugin entity (if registered)

    Attributes:
        data_root: Resolved path to the data directory.
        geometry: Shapely polygon representing the AoI boundary.
        seasons: :class:`SeasonConfig` in effect for this AoI.

    """

    def __init__(
        self,
        data_root: str,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        geometry: "shapely.Geometry | None" = None,
        seasons: SeasonConfig | None = None,
        storage_options: dict[str, Any] | None = None,
        validate_all: bool = False,
        **entity_kwargs: str | list[str],
    ) -> None:
        """Resolve the AoI boundary and scope all registered entities.

        Args:
            data_root (str): Path to the root data directory or cloud URI
                prefix (e.g. ``"s3://my-bucket/data"``).  Prepended to every
                relative entity path.
            bbox (tuple[float, float, float, float] | None, optional): Bounding box as ``(minx, miny, maxx, maxy)`` in WGS-84.
                Mutually exclusive with ``geometry`` and ``entity_kwargs``.
            geometry (shapely.Geometry | None, optional): Arbitrary Shapely geometry.  Used as-is.  Mutually
                exclusive with ``bbox`` and ``entity_kwargs``.
            seasons (SeasonConfig | None, optional): :class:`SeasonConfig` override.  Defaults to the library
                agronomic defaults.
            storage_options (dict[str, Any] | None, optional): Cloud credential / configuration
                options forwarded to all ``pyarrow.fs`` and ``polars.scan_parquet``
                calls.  For S3 the common keys are ``"region"``,
                ``"access_key"``, and ``"secret_key"``.  ``None`` (default)
                uses ambient credentials (env-vars / ``~/.aws/``).
            validate_all (bool, optional): When ``True``, eagerly instantiate
                and validate every registered entity at construction time,
                preserving the original fail-fast behaviour.  When ``False``
                (default), entities are built and validated lazily on first
                access — a broken entity only surfaces an error when it is
                actually used.  Use ``True`` in startup health-checks or
                long-running services where early failure is preferable.
            **entity_kwargs (str | list[str]): Named filter pairs that identify the boundary,
                e.g. ``tehsil="Pangi"``, ``district="Chamba"``,
                ``state="Himachal Pradesh"``, ``mws_id="13_551"``.
                Mutually exclusive with ``bbox`` and ``geometry``.

        Raises:
            ValueError: If no boundary argument is supplied, or if more than
                one boundary mode is used simultaneously.
            :class:`~core_lens.base.EntityValidationError`: If a boundary entity referenced in
                ``entity_kwargs`` is not registered.

        """
        # Preserve cloud URIs as-is; resolve local paths to absolute.
        data_root_str = str(data_root)
        if is_cloud_uri(data_root_str):
            self.data_root: pathlib.Path | str = data_root_str
        else:
            self.data_root = pathlib.Path(data_root_str).resolve()

        self._storage_options: dict[str, Any] = storage_options or {}
        self.seasons: SeasonConfig = seasons or _default_season_config()
        logger.info("Initializing AoI with data_root={}", self.data_root)

        n_modes = sum(
            [
                bbox is not None,
                geometry is not None,
                bool(entity_kwargs),
            ]
        )
        if n_modes == 0:
            logger.error("AoI initialization failed: no boundary argument supplied.")
            raise ValueError(
                "AoI requires exactly one boundary argument: "
                "bbox=, geometry=, or entity keyword filters such as tehsil='Pangi'."
            )
        if n_modes > 1:
            logger.error("AoI initialization failed: multiple boundary modes supplied.")
            raise ValueError(
                "bbox, geometry, and entity keyword filters are mutually exclusive. "
                "Provide exactly one boundary mode."
            )

        # Entity instances are created lazily on first access via _get_entity().
        self._entity_instances: dict[str, BaseEntity] = {}

        # Initialise boundary kwargs and lazily computed geometry.
        self._boundary_kwargs: dict[str, str | list[str]] | None = None
        self._geometry: shapely.Geometry | None = None

        if geometry is not None:
            self._geometry = geometry
        elif bbox is not None:
            self._geometry = _bbox_to_polygon(bbox)
        else:
            self._boundary_kwargs = entity_kwargs

        # Entity views are created lazily on demand in __getattr__.
        self._scoped: dict[str, "View"] = {}

        # Opt-in eager validation: instantiate + validate every registered
        # entity now so that a bad entity fails immediately rather than on
        # first access.  Useful for startup health-checks.
        if validate_all:
            self.validate()

    @property
    def geometry(self) -> "shapely.Geometry":
        """The resolved boundary of this AoI as a Shapely geometry."""
        if self._geometry is None:
            if self._boundary_kwargs is not None:
                self._geometry = self._resolve_named_boundary(self._boundary_kwargs)
            else:
                # Should not happen if initialization was correct
                raise ValueError("No boundary defined.")
        return self._geometry

    @property
    def current_season(self) -> str:
        """The season name for today's date under the AoI's SeasonConfig.

        Returns:
            str: ``"kharif"``, ``"rabi"``, or ``"zaid"``.

        """
        return self.seasons.season_for(date.today())

    @property
    def current_year(self) -> int:
        """The current calendar year.

        Returns:
            int: Current year as an integer.

        """
        return date.today().year

    def plot(self, overlay: "Result | None" = None) -> Any:
        """Render an interactive Lonboard map of the AoI and its entity layers.

        Args:
            overlay (Result | None, optional): An optional :class:`~core_lens.base.result.Result` to
                overlay on the map.

        Returns:
            Any: A Lonboard Map object.

        """
        import lonboard
        import geopandas as gpd

        layers = []

        # Base layer: single AoI boundary (one Shapely geom — GeoPandas is fine here).
        aoi_gdf = gpd.GeoDataFrame(geometry=[self.geometry], crs="EPSG:4326")
        base_layer = lonboard.PolygonLayer.from_geopandas(
            aoi_gdf,
            get_fill_color=[0, 0, 0, 0],
            get_line_color=[0, 0, 0, 255],
            line_width_min_pixels=2,
        )
        layers.append(base_layer)

        if overlay is not None:
            # We duck-type the overlay to avoid circular imports of Result
            if hasattr(overlay, "has_geometry") and not overlay.has_geometry:
                overlay = overlay.with_geometry()

            if hasattr(overlay, "entity") and hasattr(overlay, "data"):
                # --- Direct Polars → GeoArrow → Lonboard (no Shapely, no GeoPandas) ---
                from core_lens.base.namespaces.plot import _wkb_to_arrow_table

                geom_col = overlay.entity.geometry_col
                key_cols = (
                    list(overlay.key_cols) if hasattr(overlay, "key_cols") else []
                )
                arrow_table = _wkb_to_arrow_table(
                    overlay.df(), geom_col, extra_cols=key_cols
                )
                overlay_layer = lonboard.PolygonLayer(
                    arrow_table,
                    get_fill_color=[255, 0, 0, 100],
                    get_line_color=[255, 0, 0, 200],
                    line_width_min_pixels=1,
                )
                layers.append(overlay_layer)

        return lonboard.Map(layers=layers)

    def _get_entity(self, name: str) -> BaseEntity:
        """Instantiate, validate, and cache an entity on first access.

        Subsequent calls for the same *name* return the cached instance.
        Validation delegates to :func:`_validate_entity` which routes
        through the process-level :func:`_cached_detect` cache, so
        repeated ``AoI()`` calls against the same ``data_root`` pay no
        redundant Parquet footer reads.

        Args:
            name (str): The registered entity name (e.g. ``"mws"``).

        Returns:
            BaseEntity: The validated entity instance.

        Raises:
            KeyError: If *name* is not in :data:`_REGISTRY`.
            :class:`~core_lens.base.EntityValidationError`: If validation fails.

        """
        if name not in self._entity_instances:
            entity_cls = _REGISTRY[name]
            entity = entity_cls(
                data_root=self.data_root,
                storage_options=self._storage_options or None,
            )
            # Only validate path existence here — schema detection (parquet
            # footer reads) is deferred to first data access via schema_profile.
            _validate_entity_paths(entity, name)
            self._entity_instances[name] = entity
        return self._entity_instances[name]

    def validate(self) -> None:
        """Eagerly validate every registered entity.

        Instantiates and validates all entities in :data:`_REGISTRY` that
        have not yet been accessed.  Raises on the first failure.

        This is equivalent to the old eager-validation behaviour and is
        useful for startup health-checks in long-running services::

            aoi = AoI("data/", bbox=(...), validate_all=True)
            # or:
            aoi = AoI("data/", bbox=(...))
            aoi.validate()  # explicit call, same effect

        Raises:
            :class:`~core_lens.base.EntityValidationError`: If any registered
                entity fails validation.

        """
        for name in _REGISTRY:
            self._get_entity(name)

    def __getattr__(self, name: str) -> "View":
        """Get view by entity name.

        Resolves and caches a scoped view for a registered entity on first access.

        Args:
            name: The registered entity name (e.g. 'mws').

        Returns:
            The scoped View for the entity.

        Raises:
            AttributeError: If the entity is not registered.
        """
        # Called only when normal attribute lookup has already failed, so this
        # never shadows real attributes.  Maps entity names to their scoped Views.
        if name in _REGISTRY:
            if name not in self._scoped:
                entity = self._get_entity(name)

                if self._boundary_kwargs is not None:
                    view = entity.where(**self._boundary_kwargs)
                else:
                    view = entity.spatial_filter(geometry=self.geometry)

                view._season_config = self.seasons
                self._scoped[name] = view
            return self._scoped[name]
        raise AttributeError(
            f"'AoI' object has no attribute {name!r}. "
            f"Registered entities: {sorted(_REGISTRY)}. "
            "Use AoI.register(EntityClass) to add a new entity."
        )

    def _resolve_named_boundary(
        self, entity_kwargs: dict[str, str | list[str]]
    ) -> "shapely.Geometry":
        """Resolve a set of named attribute filters to a Shapely geometry.

        Delegates to the process-level :func:`_cached_resolve_boundary` cache
        so that the expensive Parquet scan + WKB decode + ``unary_union`` is
        only paid once per unique ``(data_root, entity_kwargs)`` combination
        within a Python process.  Subsequent ``AoI()`` calls with the same
        boundary specification reconstruct the geometry from cached WKB bytes.

        Args:
            entity_kwargs (dict[str, str | list[str]]): Column–value pairs used to identify the boundary.

        Returns:
            shapely.Geometry: The union of all matching entity geometries as a Shapely object.

        Raises:
            :class:`~core_lens.base.EntityValidationError`: If no registered entity can satisfy the filters.
            ValueError: If the filters match zero rows.

        """
        logger.debug("Resolving named boundary using kwargs: {}", entity_kwargs)

        from core_lens.base.entity import _so_key

        # Build hashable cache key from entity_kwargs.
        # list values are converted to tuples so they are hashable.
        kwargs_hashable: dict[str, Any] = {
            k: tuple(v) if isinstance(v, list) else v for k, v in entity_kwargs.items()
        }
        entity_kwargs_key = tuple(sorted(kwargs_hashable.items()))
        data_root_str = str(self.data_root)
        so_key = _so_key(self._storage_options)

        # Delegate to the process-level cache.  On a cache miss this runs the
        # full scan + WKB decode + unary_union; on a hit it returns instantly.
        geom_wkb, boundary_entity_name, key_rows = _cached_resolve_boundary(
            data_root_str, so_key, entity_kwargs_key
        )

        # Reconstruct the boundary geometry from cached WKB bytes (fast).
        geometry: shapely.Geometry = shapely.from_wkb(geom_wkb)

        return geometry

    @classmethod
    def register(cls, entity_cls: type[BaseEntity]) -> None:
        """Register an entity class so it is available on all future AoI instances.

        The entity name is derived from the class name by stripping a trailing
        ``"Entity"`` suffix and lower-casing the result
        (``MWSEntity`` → ``"mws"``, ``ForestEntity`` → ``"forest"``).

        For entities with **absolute** paths, validation (file existence, key cols,
        geometry col) runs immediately at registration time.
        For entities with **relative** paths, validation is deferred until an
        :class:`AoI` is instantiated (when ``data_root`` is known).

        Args:
            entity_cls (type[BaseEntity]): A concrete subclass of
                :class:`~core_lens.base.entity.BaseEntity`.

        Raises:
            :class:`~core_lens.base.EntityValidationError`: If any validation check
                fails (absolute-path entities only at register time).

        """
        import pathlib as _pathlib

        name = _entity_name(entity_cls)
        logger.info("Registering entity class: {}", name)
        # Probe with a no-root instance to check if paths are absolute.
        probe = entity_cls()
        static = probe.static_path
        if _pathlib.Path(static).is_absolute() and not is_cloud_uri(static):
            # Absolute local path entity: validate now.
            _validate_entity(probe, name)
        # Relative path or cloud URI entity: validation deferred to AoI.__init__.
        _REGISTRY[name] = entity_cls

    @classmethod
    def deregister(cls, entity_cls: type[BaseEntity]) -> None:
        """Remove a previously registered entity.

        Primarily useful in tests where a clean registry is needed between runs.

        Args:
            entity_cls (type[BaseEntity]): The entity class to remove.

        """
        name = _entity_name(entity_cls)
        logger.info("Deregistering entity class: {}", name)
        _REGISTRY.pop(name, None)

    @classmethod
    def registered_entities(cls) -> list[str]:
        """Return the names of all currently registered entities.

        Returns:
            list[str]: A sorted list of entity name strings.

        """
        return sorted(_REGISTRY)


def _entity_name(entity_cls: type[BaseEntity]) -> str:
    name = entity_cls.__name__
    if name.endswith("Entity"):
        name = name[: -len("Entity")]
    return name.lower()


def _bbox_to_polygon(
    bbox: tuple[float, float, float, float],
) -> "shapely.Geometry":
    minx, miny, maxx, maxy = bbox
    return sgeom.box(minx, miny, maxx, maxy)


def _validate_entity_paths(entity: BaseEntity, name: str) -> None:
    """Validate an entity's path existence only — no Parquet I/O.

    Checks that ``static_path``, ``annual_path``, and ``sub_annual_path``
    resolve to existing local files.  Schema validation (key columns,
    geometry column, geometry type) is intentionally deferred to first data
    access via :attr:`BaseEntity.schema_profile`, which routes through the
    process-level :func:`~core_lens.base.entity._cached_detect` cache.

    This is the fast variant called from :meth:`AoI._get_entity`.  Use
    :func:`_validate_entity` for the full eager check (e.g. in
    :meth:`AoI.validate` or for absolute-path entities at register time).

    Args:
        entity (BaseEntity): The entity instance to validate.
        name (str): Human-readable entity name for error messages.

    Raises:
        EntityValidationError: If any path does not exist.

    """
    # --- Static path existence check ----------------------------------------
    try:
        static = entity._resolve(entity.static_path)
    except FileNotFoundError:
        logger.error(
            "Validation failed for entity {}: static_path '{}' does not exist.",
            name,
            entity.static_path,
        )
        raise EntityValidationError(
            f"Entity {name!r}: static_path {entity.static_path!r} does not exist."
        )

    if not is_cloud_uri(static) and not os.path.exists(static):
        logger.error(
            "Validation failed for entity {}: static path '{}' does not exist.",
            name,
            static,
        )
        raise EntityValidationError(
            f"Entity {name!r}: static_path {static!r} does not exist."
        )

    # --- Temporal path existence checks (no schema read) --------------------
    for attr, label in [("annual_path", "annual"), ("sub_annual_path", "sub_annual")]:
        path = getattr(entity, attr)
        if path is not None:
            try:
                abs_path = entity._resolve(path)
            except FileNotFoundError:
                abs_path = None
            if abs_path is None or (
                not is_cloud_uri(abs_path) and not os.path.exists(abs_path)
            ):
                logger.error(
                    "Validation failed for entity {}: {} path '{}' does not exist.",
                    name,
                    label,
                    path,
                )
                raise EntityValidationError(
                    f"Entity {name!r}: {label}_path {path!r} does not exist."
                )


def _validate_entity(entity: BaseEntity, name: str) -> None:
    """Validate an entity's paths and schema (full eager check).

    Runs path existence checks via :func:`_validate_entity_paths`, then
    triggers schema detection (Parquet footer reads) by accessing
    :attr:`BaseEntity.schema_profile`.  The result is cached by
    :func:`~core_lens.base.entity._cached_detect`, so the second call for the
    same ``data_root`` pays no I/O.

    Called by :meth:`AoI.validate` and by :meth:`AoI.register` for
    entities with absolute paths.  Normal lazy instantiation via
    :meth:`AoI._get_entity` uses the cheaper :func:`_validate_entity_paths`.

    Args:
        entity (BaseEntity): The entity instance to validate.
        name (str): Human-readable entity name for error messages.

    Raises:
        EntityValidationError: If any validation check fails.

    """
    # Fast path-only check first.
    _validate_entity_paths(entity, name)

    # --- Schema validation (key_cols, geometry_col, geometry_type) -----------
    # Accessing schema_profile triggers _cached_detect, which reads Parquet
    # footer metadata.  Subsequent calls for the same data_root are free.
    try:
        static = entity._resolve(entity.static_path)
        _ = entity.schema_profile
    except EntityValidationError:
        raise
    except Exception as exc:
        static = entity.static_path  # best-effort for the error message
        logger.error(
            "Validation failed for entity {}: could not read schema from '{}': {}",
            name,
            static,
            exc,
        )
        raise EntityValidationError(
            f"Entity {name!r}: could not read schema from {static!r}: {exc}"
        ) from exc
