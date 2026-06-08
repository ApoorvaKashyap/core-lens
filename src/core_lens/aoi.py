"""AoI (Area of Interest) — primary entry point for core_lens."""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core_lens.base.result import Result

import polars as pl

from core_lens.base.entity import BaseEntity, EntityValidationError

if TYPE_CHECKING:
    import shapely
    from core_lens.base.view import View


@dataclass
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

    kharif: tuple[str, str] = ("07-01", "10-30")
    rabi: tuple[str, str] = ("11-01", "03-31")
    zaid: tuple[str, str] = ("04-01", "06-30")

    def __post_init__(self) -> None:
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
            d: The date to classify.

        Returns:
            ``"kharif"``, ``"rabi"``, or ``"zaid"``.
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


# Class-level registry: entity name → entity instance.
# Shared across all AoI instances in a process.  Explicit registration is
# required; there is no auto-discovery (design §6.1).
_REGISTRY: dict[str, BaseEntity] = {}


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
    materialisation property (``.static``, ``.annual``, ``.fortnightly``) is
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
        **entity_kwargs: str,
    ) -> None:
        """Resolve the AoI boundary and scope all registered entities.

        Args:
            data_root: Path to the root data directory.  Prepended to every
                relative entity path that does not start with ``/`` or a URI
                scheme.
            bbox: Bounding box as ``(minx, miny, maxx, maxy)`` in WGS-84.
                Mutually exclusive with ``geometry`` and ``entity_kwargs``.
            geometry: Arbitrary Shapely geometry.  Used as-is.  Mutually
                exclusive with ``bbox`` and ``entity_kwargs``.
            seasons: :class:`SeasonConfig` override.  Defaults to the library
                agronomic defaults.
            **entity_kwargs: Named filter pairs that identify the boundary,
                e.g. ``tehsil="Pangi"``, ``district="Chamba"``,
                ``state="Himachal Pradesh"``, ``mws_id="13_551"``.
                Mutually exclusive with ``bbox`` and ``geometry``.

        Raises:
            ValueError: If no boundary argument is supplied, or if more than
                one boundary mode is used simultaneously.
            :class:`~core_lens.base.EntityValidationError`: If a boundary entity referenced in
                ``entity_kwargs`` is not registered.
        """
        self.data_root = pathlib.Path(data_root).resolve()
        self.seasons: SeasonConfig = seasons or SeasonConfig()

        n_modes = sum(
            [
                bbox is not None,
                geometry is not None,
                bool(entity_kwargs),
            ]
        )
        if n_modes == 0:
            raise ValueError(
                "AoI requires exactly one boundary argument: "
                "bbox=, geometry=, or entity keyword filters such as tehsil='Pangi'."
            )
        if n_modes > 1:
            raise ValueError(
                "bbox, geometry, and entity keyword filters are mutually exclusive. "
                "Provide exactly one boundary mode."
            )

        if geometry is not None:
            self.geometry: "shapely.Geometry" = geometry
        elif bbox is not None:
            self.geometry = _bbox_to_polygon(bbox)
        else:
            self.geometry = self._resolve_named_boundary(entity_kwargs)

        # Pre-filter every registered entity to the resolved geometry.
        # The season_config is threaded through to each View so that
        # season-based time filters can be resolved at materialisation time.
        self._scoped: dict[str, "View"] = {}
        for name, entity in _REGISTRY.items():
            view = entity.spatial_filter(geometry=self.geometry)
            view._season_config = self.seasons
            self._scoped[name] = view

    @property
    def current_season(self) -> str:
        """The season name for today's date under the AoI's SeasonConfig.

        Returns:
            ``"kharif"``, ``"rabi"``, or ``"zaid"``.
        """
        return self.seasons.season_for(date.today())

    @property
    def current_year(self) -> int:
        """The current calendar year.

        Returns:
            Current year as an integer.
        """
        return date.today().year

    def plot(self, overlay: "Result | None" = None) -> Any:
        """Render an interactive Lonboard map of the AoI and its entity layers.

        Args:
            overlay: An optional :class:`~core_lens.base.result.Result` to
                overlay on the map.

        Returns:
            A Lonboard Map object.
        """
        import lonboard
        import geopandas as gpd

        layers = []

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

            if hasattr(overlay, "gdf"):
                gdf = overlay.gdf()
                overlay_layer = lonboard.PolygonLayer.from_geopandas(
                    gdf,
                    get_fill_color=[255, 0, 0, 100],
                    get_line_color=[255, 0, 0, 200],
                )
                layers.append(overlay_layer)

        return lonboard.Map(layers=layers)

    def __getattr__(self, name: str) -> "View":
        # Called only when normal attribute lookup has already failed, so this
        # never shadows real attributes.  Maps entity names to their scoped Views.
        if name in _REGISTRY:
            return self._scoped[name]
        raise AttributeError(
            f"'AoI' object has no attribute {name!r}. "
            f"Registered entities: {sorted(_REGISTRY)}. "
            "Use AoI.register(EntityClass) to add a new entity."
        )

    def _resolve_named_boundary(
        self, entity_kwargs: dict[str, str]
    ) -> "shapely.Geometry":
        """Resolve a set of named attribute filters to a Shapely geometry.

        The entity whose key column matches one of the kwargs is queried.
        Multiple kwargs act as AND-filters (e.g. tehsil + district narrows to
        the unique matching row).

        Args:
            entity_kwargs: Column–value pairs used to identify the boundary.

        Returns:
            The union of all matching entity geometries as a Shapely object.

        Raises:
            :class:`~core_lens.base.EntityValidationError`: If no registered entity can satisfy the filters.
            ValueError: If the filters match zero rows.
        """
        import shapely.ops as sops
        import shapely.wkb as swkb

        # Find the registered entity whose key_col or known attribute column
        # matches one of the filter keys.
        candidate: BaseEntity | None = None
        for entity in _REGISTRY.values():
            schema = entity.schema_profile
            if any(
                k in schema.key_cols or k in schema.extra_static_cols
                for k in entity_kwargs
            ):
                candidate = entity
                break

        # Fall back: look for an entity whose name matches a kwarg key
        # (e.g. tehsil="Pangi" → TehsilEntity if registered as "tehsil").
        if candidate is None:
            for name, entity in _REGISTRY.items():
                if name in entity_kwargs:
                    candidate = entity
                    break

        if candidate is None:
            raise EntityValidationError(
                f"No registered entity can satisfy the filters {entity_kwargs}. "
                f"Registered entities: {sorted(_REGISTRY)}."
            )

        schema = candidate.schema_profile
        geom_col = schema.geometry_col

        # Read only the geometry and key columns — no full scan.
        filters = [
            (col, "==", val)
            for col, val in entity_kwargs.items()
            if col in candidate.schema_profile.key_cols
            or col in candidate.schema_profile.extra_static_cols
        ]

        df = pl.read_parquet(
            candidate.static_path,
            columns=candidate.key_cols + [geom_col],
        )
        for col, _, val in filters:
            if col in df.columns:
                df = df.filter(pl.col(col) == val)

        if df.is_empty():
            raise ValueError(
                f"No rows matched the filters {entity_kwargs} "
                f"in {candidate.static_path!r}."
            )

        geoms = [swkb.loads(row) for row in df[geom_col].to_list()]
        return sops.unary_union(geoms) if len(geoms) > 1 else geoms[0]

    @classmethod
    def register(cls, entity_cls: type[BaseEntity]) -> None:
        """Register an entity class so it is available on all future AoI instances.

        The entity name is derived from the class name by stripping a trailing
        ``"Entity"`` suffix and lower-casing the result
        (``MWSEntity`` → ``"mws"``, ``ForestEntity`` → ``"forest"``).

        Validation checks (§6.4):

        1. ``static_path`` exists and is readable.
        2. ``key_cols`` are present in the static file schema.
        3. ``geometry_col`` is present in the static file schema.
        4. ``annual_path`` and ``fortnightly_path`` exist if declared.

        Args:
            entity_cls: A concrete subclass of
                :class:`~core_lens.base.entity.BaseEntity`.

        Raises:
            :class:`~core_lens.base.EntityValidationError`: If any validation check fails.
        """
        entity = entity_cls()
        name = _entity_name(entity_cls)
        _validate_entity(entity, name)
        _REGISTRY[name] = entity

    @classmethod
    def deregister(cls, entity_cls: type[BaseEntity]) -> None:
        """Remove a previously registered entity.

        Primarily useful in tests where a clean registry is needed between runs.

        Args:
            entity_cls: The entity class to remove.
        """
        name = _entity_name(entity_cls)
        _REGISTRY.pop(name, None)

    @classmethod
    def registered_entities(cls) -> list[str]:
        """Return the names of all currently registered entities.

        Returns:
            A sorted list of entity name strings.
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
    import shapely.geometry as sgeom

    minx, miny, maxx, maxy = bbox
    return sgeom.box(minx, miny, maxx, maxy)


def _validate_entity(entity: BaseEntity, name: str) -> None:
    static = entity.static_path

    if not os.path.exists(static):
        raise EntityValidationError(
            f"Entity {name!r}: static_path {static!r} does not exist."
        )

    try:
        schema = pl.read_parquet_schema(static)
    except Exception as exc:
        raise EntityValidationError(
            f"Entity {name!r}: could not read schema from {static!r}: {exc}"
        ) from exc

    missing_keys = [c for c in entity.key_cols if c not in schema]
    if missing_keys:
        raise EntityValidationError(
            f"Entity {name!r}: key_cols {missing_keys} not found in {static!r}. "
            f"Available columns: {list(schema.keys())}."
        )

    if entity.geometry_col not in schema:
        raise EntityValidationError(
            f"Entity {name!r}: geometry_col {entity.geometry_col!r} "
            f"not found in {static!r}. Available columns: {list(schema.keys())}."
        )

    for attr, label in [("annual_path", "annual"), ("fortnightly_path", "fortnightly")]:
        path = getattr(entity, attr)
        if path is not None and not os.path.exists(path):
            raise EntityValidationError(
                f"Entity {name!r}: {label}_path {path!r} does not exist."
            )
