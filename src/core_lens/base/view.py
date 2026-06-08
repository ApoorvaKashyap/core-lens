"""Lazy, immutable view of a scoped entity pending materialisation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from core_lens.schema.profile import Resolution
from core_lens.utils.polars_utils import scan_with_key_filter
from core_lens.utils.spatial import resolve_path

if TYPE_CHECKING:
    from core_lens.aoi import SeasonConfig
    from core_lens.base.entity import BaseEntity
    from core_lens.base.result import Result

_VALID_SEASONS = {"kharif", "rabi", "zaid", "current"}


class View:
    """Lazy, immutable snapshot of a scoped entity pending materialisation.

    A ``View`` is produced by the filter methods on :class:`~core_lens.base.entity.BaseEntity`
    (:meth:`~BaseEntity.where`, :meth:`~BaseEntity.spatial_filter`,
    :meth:`~BaseEntity.spatial_join`).  It records *what* to load without
    touching any Parquet file.  Data is only read when one of the three
    materialisation properties is accessed.

    ``View`` is immutable.  Every method that would mutate state instead
    returns a new ``View`` with the updated field, leaving the original
    untouched.  This makes it safe to branch a view and materialise it in
    different ways::

        view = aoi.mws.where(tehsil="Pangi")
        annual = view.between("2010-01-01", "2023-12-31").annual
        static = view.static

    Attributes:
        keys: A ``pl.DataFrame`` containing the resolved key column(s) for
            all entity instances that passed the spatial/attribute filters.
            This is the in-memory index slice — no data columns, only IDs.
        entity: Reference to the parent :class:`~core_lens.base.entity.BaseEntity`
            that produced this view.  Used during materialisation to resolve
            file paths and schema.
        entity_name: Human-readable name for the entity (e.g. ``"mws"``).
            Used to namespace joined columns and populate
            :attr:`~core_lens.base.result.Result.entity_name`.
        time_filter: A dict encoding the pending time constraint set by
            :meth:`between`, or ``None`` if no time filter has been applied.
            The materialisation step reads this dict to build predicate
            pushdown expressions.
        join_spec: A dict encoding a deferred :meth:`~BaseEntity.spatial_join`
            request, or ``None``.  Evaluated during materialisation after the
            primary scan is complete.
    """

    def __init__(
        self,
        keys: pl.DataFrame,
        entity: BaseEntity,
        entity_name: str,
        time_filter: dict[str, Any] | None = None,
        join_spec: dict[str, Any] | None = None,
        season_config: SeasonConfig | None = None,
    ) -> None:
        self.keys = keys
        self.entity = entity
        self.entity_name = entity_name
        self.time_filter = time_filter
        self.join_spec = join_spec
        # season_config is only needed at materialisation time when a season
        # time_filter is present.  Carried forward by between() to new Views.
        self._season_config = season_config

    def between(
        self,
        start: str | None = None,
        end: str | None = None,
        *,
        season: str | None = None,
        year: int | tuple[int, int] | None = None,
    ) -> "View":
        """Return a new ``View`` with a pending time filter applied.

        Two mutually exclusive modes are supported:

        **Date range mode** — pass ``start`` and ``end`` as ISO-8601 strings::

            view.between("2010-01-01", "2023-12-31")

        **Season mode** — pass ``season`` as a keyword argument.  ``year`` is
        optional and may be a single year or an inclusive ``(from, to)`` tuple::

            view.between(season="kharif")
            view.between(season="kharif", year=2020)
            view.between(season="kharif", year=(2018, 2023))
            view.between(season="current")

        The returned ``View`` carries the filter in :attr:`time_filter` but
        does not execute any I/O.  The filter is applied during materialisation
        via predicate pushdown on the Parquet scan.

        Args:
            start: Start of the date range (ISO-8601).  Required in date range
                mode; must be ``None`` in season mode.
            end: End of the date range (ISO-8601).  Required in date range
                mode; must be ``None`` in season mode.
            season: Season name — one of ``"kharif"``, ``"rabi"``,
                ``"zaid"``, or ``"current"``.  Activates season mode.
            year: Year or inclusive year range to restrict the season filter.
                Only valid in season mode.  ``"current"`` season ignores this.

        Returns:
            A new :class:`View` with :attr:`time_filter` set.

        Raises:
            ValueError: If the arguments are inconsistent (e.g. mixing date
                range and season arguments, omitting required args, or
                supplying ``year`` with ``season="current"``).
        """
        if season is not None and (start is not None or end is not None):
            raise ValueError(
                "Date range (start/end) and season mode are mutually exclusive. "
                "Use either positional date strings or the 'season' keyword, not both."
            )

        if season is None and year is not None:
            raise ValueError("'year' is only valid when 'season' is also provided.")

        if season is not None:
            if season not in _VALID_SEASONS:
                raise ValueError(
                    f"Unknown season {season!r}. "
                    f"Valid values: {sorted(_VALID_SEASONS)}."
                )
            if season == "current" and year is not None:
                raise ValueError(
                    "Cannot combine year with season='current'. "
                    "The current season is always resolved to the present calendar date."
                )
            time_filter: dict[str, Any] = {"season": season}
            if year is not None:
                time_filter["year"] = year
        else:
            if start is None or end is None:
                raise ValueError(
                    "Both 'start' and 'end' must be provided for date range mode."
                )
            time_filter = {"start": start, "end": end}

        return View(
            keys=self.keys,
            entity=self.entity,
            entity_name=self.entity_name,
            time_filter=time_filter,
            join_spec=self.join_spec,
            season_config=self._season_config,
        )

    @property
    def static(self) -> "Result":
        """Materialise the static GeoParquet file and return a :class:`~core_lens.base.result.Result`.

        The result always has ``has_geometry=True`` because the static file is
        a GeoParquet carrying geometry for every entity instance.

        Returns:
            A :class:`~core_lens.base.result.Result` with
            ``resolution="static"`` and ``has_geometry=True``.

        Raises:
            AttributeError: If the entity has no ``static_path`` (should not
                happen in practice since ``static_path`` is mandatory).
        """
        return self._materialise(Resolution.STATIC)

    @property
    def annual(self) -> "Result":
        """Materialise the annual Parquet file and return a :class:`~core_lens.base.result.Result`.

        Returns:
            A :class:`~core_lens.base.result.Result` with
            ``resolution="annual"`` and ``has_geometry=False``.

        Raises:
            AttributeError: If the entity has no ``annual_path``.
        """
        return self._materialise(Resolution.ANNUAL)

    @property
    def fortnightly(self) -> "Result":
        """Materialise the fortnightly Parquet file and return a :class:`~core_lens.base.result.Result`.

        Returns:
            A :class:`~core_lens.base.result.Result` with
            ``resolution="fortnightly"`` and ``has_geometry=False``.

        Raises:
            AttributeError: If the entity has no ``fortnightly_path``.
        """
        return self._materialise(Resolution.FORTNIGHTLY)

    def _materialise(self, resolution: Resolution) -> "Result":
        from core_lens.base.result import Result

        if self.join_spec is not None:
            raise NotImplementedError(
                "spatial_join materialisation is not yet implemented. "
                "The join_spec is recorded but cross-entity join execution "
                "will be added in a subsequent release."
            )

        profile = self.entity.schema_profile

        path: str
        if resolution == Resolution.STATIC:
            path = self.entity.static_path
        elif resolution == Resolution.ANNUAL:
            annual_path = self.entity.annual_path
            if annual_path is None:
                raise AttributeError(
                    f"Entity {self.entity_name!r} has no annual_path declared."
                )
            path = annual_path
        else:
            fn_path = self.entity.fortnightly_path
            if fn_path is None:
                raise AttributeError(
                    f"Entity {self.entity_name!r} has no fortnightly_path declared."
                )
            path = fn_path

        abs_path = resolve_path(path)

        # Build a time filter expression when a time_filter is present and the
        # resolution is not static (static files have no time column).
        time_expr: pl.Expr | None = None
        if self.time_filter is not None and resolution != Resolution.STATIC:
            time_col = (
                profile.annual_time_col
                if resolution == Resolution.ANNUAL
                else profile.fortnightly_time_col
            )
            if time_col is not None:
                from core_lens.utils.season import resolve_time_filter

                # Fall back to a no-op SeasonConfig if none was supplied on this View.
                from core_lens.aoi import SeasonConfig

                season_cfg = self._season_config or SeasonConfig()
                time_expr = resolve_time_filter(self.time_filter, time_col, season_cfg)

        lf = scan_with_key_filter(
            path=abs_path,
            key_cols=self.entity.key_cols,
            key_values=self.keys,
            time_expr=time_expr,
        )
        data = lf.collect()

        # For fortnightly results, inject temporal grouping columns so that
        # aggregate(by="year"), aggregate(by="season"), etc. work out of the
        # box without callers having to derive them manually.
        if (
            resolution == Resolution.FORTNIGHTLY
            and profile.fortnightly_time_col is not None
        ):
            from core_lens.utils.season import add_temporal_columns
            from core_lens.aoi import SeasonConfig

            season_cfg = self._season_config or SeasonConfig()
            data = add_temporal_columns(data, profile.fortnightly_time_col, season_cfg)

        return Result(
            data=data,
            resolution=resolution,
            has_geometry=(resolution == Resolution.STATIC),
            key_cols=self.entity.key_cols,
            entity_name=self.entity_name,
            entity=self.entity,
        )
