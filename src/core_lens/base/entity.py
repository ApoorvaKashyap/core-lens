"""Base entity contract for core_lens.

All entities — built-in or plugin — must subclass :class:`BaseEntity` and
implement its abstract interface.  Concrete implementations live in
``core_lens.entities.*`` (built-ins) or in third-party packages (plugins).

Plugin authors import from the public surface::

    from core_lens.base import BaseEntity
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import polars as pl

from core_lens.utils.spatial import (
    bbox_intersects_geometry,
    build_bbox_index,
    exact_spatial_filter,
    resolve_path,
)

if TYPE_CHECKING:
    import shapely
    from core_lens.base.view import View
    from core_lens.schema.profile import SchemaProfile


class BaseEntity(ABC):
    """Abstract base class for every entity in the core_lens plugin system.

    An entity represents a geospatial primitive (e.g. microwatershed, village,
    district) backed by one or more Parquet/GeoParquet files.  Entities are
    *descriptors* — they carry path and schema metadata but hold no per-row
    state themselves.  Row-level data lives in :class:`~core_lens.base.view.View`
    and :class:`~core_lens.base.result.Result`.

    Subclasses **must** implement:

    * :attr:`key_cols`      — column(s) that uniquely identify one entity instance
    * :attr:`geometry_col`  — geometry column name in the static GeoParquet file
    * :attr:`static_path`   — path to the static GeoParquet file (mandatory)

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
            static_path  = \"/data/forest/static.geoparquet\"
            annual_path  = \"/data/forest/annual.parquet\"

        AoI.register(ForestEntity)

    Validation rules enforced at :meth:`AoI.register` time:

    1. ``static_path`` exists and is readable.
    2. ``key_cols`` are present and unique in the static file.
    3. ``geometry_col`` is present and contains valid geometries.
    4. ``annual_path`` and ``fortnightly_path`` exist if declared.

    Any failure raises :class:`~core_lens.base.entity.EntityValidationError`.
    """

    @property
    @abstractmethod
    def key_cols(self) -> list[str]:
        """Columns that uniquely identify one instance of this entity.

        For built-in entities this is always a single-element list (e.g.
        ``[\"mws_id\"]``), but the contract allows composite keys for plugins.

        Returns:
            A list of column name strings present in the static file.
        """

    @property
    @abstractmethod
    def geometry_col(self) -> str:
        """Name of the geometry column in the static GeoParquet file.

        The column must contain a geometry type understood by GeoPandas
        (WKB bytes, WKT string, or a native geometry column).

        Returns:
            The column name as a string.
        """

    @property
    @abstractmethod
    def static_path(self) -> str:
        """Absolute filesystem path to the static GeoParquet file.

        The path must be absolute.  If a relative path is provided it is
        resolved against the current working directory at first use.  A
        ``FileNotFoundError`` is raised if the file does not exist.

        Returns:
            A path string.
        """

    @property
    def annual_path(self) -> str | None:
        """Path to the annual time-series Parquet file, or ``None``.

        Override in subclasses that carry annual temporal data.  If declared,
        the file must exist at :meth:`AoI.register` time or
        :class:`EntityValidationError` is raised.

        Returns:
            A path string, or ``None`` if the entity has no annual data.
        """
        return None

    @property
    def fortnightly_path(self) -> str | None:
        """Path to the fortnightly time-series Parquet file, or ``None``.

        Override in subclasses that carry fortnightly temporal data.  If
        declared, the file must exist at :meth:`AoI.register` time or
        :class:`EntityValidationError` is raised.

        Returns:
            A path string, or ``None`` if the entity has no fortnightly data.
        """
        return None

    @property
    def schema_profile(self) -> "SchemaProfile":
        """Validated schema descriptor for this entity's data files.

        Auto-detected from Parquet file metadata on first access and cached on
        the instance.  Override in subclasses to provide an explicit profile
        instead of relying on detection.

        Returns:
            A fully-validated :class:`~core_lens.schema.profile.SchemaProfile`.
        """
        if not hasattr(self, "_schema_profile"):
            from core_lens.schema.detection import detect

            self._schema_profile: SchemaProfile = detect(
                static_path=resolve_path(self.static_path),
                key_cols=self.key_cols,
                geometry_col=self.geometry_col,
                annual_path=(
                    resolve_path(self.annual_path)
                    if self.annual_path is not None
                    else None
                ),
                fortnightly_path=(
                    resolve_path(self.fortnightly_path)
                    if self.fortnightly_path is not None
                    else None
                ),
            )
        return self._schema_profile

    # ------------------------------------------------------------------
    # In-memory index — built lazily, cached on the instance.
    # ------------------------------------------------------------------

    @property
    def _index(self) -> pl.DataFrame:
        if not hasattr(self, "_cached_index"):
            profile = self.schema_profile
            self._cached_index: pl.DataFrame = build_bbox_index(
                static_path=resolve_path(self.static_path),
                key_cols=self.key_cols,
                bbox_cols=profile.bbox_cols,
                geometry_col=profile.geometry_col,
                geometry_type=profile.geometry_type,
            )
        return self._cached_index

    def where(self, **kwargs: Any) -> "View":
        """Return a lazy :class:`~core_lens.base.view.View` filtered by attributes.

        Each keyword argument is interpreted as ``column=value`` applied to
        the static file's attribute columns.  Multiple arguments are AND-ed.
        The static file is scanned with predicate pushdown so only the matching
        rows are read.

        Note: Attribute columns not present in the in-memory index (which
        contains only key and bbox columns) are resolved by scanning the static
        file.

        Args:
            **kwargs: Arbitrary column–value pairs to filter on.

        Returns:
            A lazy :class:`~core_lens.base.view.View` with resolved key pairs.
        """
        from core_lens.base.view import View

        static = resolve_path(self.static_path)

        # Build a filter expression for each kwarg — pushed into scan_parquet.
        filter_expr = pl.lit(True)
        for col, val in kwargs.items():
            filter_expr = filter_expr & (pl.col(col) == val)

        lf = pl.scan_parquet(static).filter(filter_expr).select(self.key_cols)
        keys = lf.collect()

        entity_name = _entity_name(type(self))
        return View(keys=keys, entity=self, entity_name=entity_name)

    def spatial_filter(
        self,
        geometry: "shapely.Geometry | None" = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> "View":
        """Return a lazy :class:`~core_lens.base.view.View` filtered by geometry.

        Uses the in-memory bbox index for a fast rectangular pre-filter, then
        refines with a Shapely STRtree exact-intersection check.  If ``bbox``
        is provided it is converted to a Shapely box before filtering.

        Args:
            geometry: A Shapely geometry representing the spatial extent.
            bbox: Bounding box as ``(minx, miny, maxx, maxy)`` in WGS-84.
                Converted to a ``shapely.geometry.box`` internally.

        Returns:
            A lazy :class:`~core_lens.base.view.View` scoped to the given
            spatial extent.

        Raises:
            ValueError: If neither ``geometry`` nor ``bbox`` is provided.
        """
        import shapely.geometry as sgeom

        from core_lens.base.view import View

        if geometry is None and bbox is None:
            raise ValueError(
                "spatial_filter() requires either 'geometry' or 'bbox' to be provided."
            )
        if bbox is not None and geometry is None:
            geometry = sgeom.box(*bbox)

        assert geometry is not None  # guaranteed by the guards above

        profile = self.schema_profile
        candidates = bbox_intersects_geometry(self._index, geometry)
        keys = exact_spatial_filter(
            candidates=candidates,
            static_path=resolve_path(self.static_path),
            key_cols=self.key_cols,
            geometry_col=profile.geometry_col,
            geometry_type=profile.geometry_type,
            aoi_geometry=geometry,
        )

        entity_name = _entity_name(type(self))
        return View(keys=keys, entity=self, entity_name=entity_name)

    def spatial_join(self, other: "BaseEntity", agg: dict[str, str]) -> "View":
        """Return a lazy :class:`~core_lens.base.view.View` with a cross-entity join pending.

        The join is recorded in the View's ``join_spec`` and computed only at
        materialisation time (``.static``, ``.annual``, or ``.fortnightly``).
        Joined columns are namespaced as ``{entity_name}_{column_name}``.

        Args:
            other: The secondary :class:`BaseEntity` whose columns will be
                joined and aggregated onto ``self``.
            agg: Mapping of ``{column: aggregation}`` specifying which columns
                from ``other`` to bring in and how to aggregate them.  Valid
                aggregation strings are ``\"area\"``, ``\"count\"``, ``\"mean\"``,
                ``\"sum\"``, ``\"min\"``, and ``\"max\"``.

        Returns:
            A lazy :class:`~core_lens.base.view.View` with the join spec
            recorded for deferred execution.
        """
        from core_lens.base.view import View

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
