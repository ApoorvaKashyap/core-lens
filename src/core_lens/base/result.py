"""Concrete, immutable result of a materialised View."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from core_lens.schema.profile import Resolution
from core_lens.utils.polars_utils import collect_lf

if TYPE_CHECKING:
    import geopandas as gpd
    from core_lens.base.entity import BaseEntity
    from core_lens.base.namespaces.stats import StatsNamespace
    from core_lens.base.namespaces.plot import PlotNamespace

_FORTNIGHTLY_ONLY_BY = {"month", "year_month", "season", "season_year"}
_ANNUAL_OR_FORTNIGHTLY_BY = {"year"}
_VALID_BY = {None} | _FORTNIGHTLY_ONLY_BY | _ANNUAL_OR_FORTNIGHTLY_BY


class Result:
    """Concrete, immutable result of a materialised :class:`~core_lens.base.view.View`.

    ``Result`` is the shared output type for every entity and every
    materialisation path.  It wraps a ``pl.DataFrame``, carries enough
    context to know how to re-attach geometry, and exposes compute methods
    that always return a fresh ``Result``, keeping the chain composable::

        result = aoi.mws.where(tehsil="Pangi").annual.between("2010-01-01", "2023-12-31")
        mean_ndvi = result.aggregate(pl.mean("ndvi"), by="year")
        mean_ndvi.plot.timeseries(x="year", y="ndvi")

    Attributes:
        data: The materialised ``pl.DataFrame``.  All compute methods
            produce a new ``Result`` whose ``data`` is the transformed frame.
        metadata: Free-form dict populated by stats methods to carry
            method parameters (e.g. ``{"method": "pearson", "p_value": 0.003}``).
            Empty on freshly materialised results.
        resolution: A :class:`~core_lens.schema.profile.Resolution` member
            (``STATIC``, ``ANNUAL``, or ``FORTNIGHTLY``).  Used to validate
            which ``aggregate`` groupings are legal.
        has_geometry: ``True`` only for ``resolution="static"`` results and
            results on which :meth:`with_geometry` has been called.  When
            ``True``, :meth:`gdf` is available.
        columns: Column names present in :attr:`data` at construction time.
            Snapshots the schema so callers can introspect without touching
            the frame.
        key_cols: The entity's key column(s) as carried from
            :class:`~core_lens.base.entity.BaseEntity`.
        entity_name: Human-readable entity identifier (e.g. ``"mws"``).
        entity: Reference to the parent entity, retained so
            :meth:`with_geometry` can locate the static file without
            requiring callers to pass it again.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        resolution: Resolution,
        has_geometry: bool,
        key_cols: list[str],
        entity_name: str,
        entity: "BaseEntity",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.data = data
        self.resolution = resolution
        self.has_geometry = has_geometry
        self.columns: list[str] = data.columns
        self.key_cols = key_cols
        self.entity_name = entity_name
        self.entity = entity
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}

    def df(self) -> pl.DataFrame:
        """Return the underlying ``pl.DataFrame``.

        Returns:
            The materialised data frame.
        """
        return self.data

    def gdf(self) -> "gpd.GeoDataFrame":
        """Return the data as a ``GeoDataFrame``.

        Only valid when :attr:`has_geometry` is ``True``.  Use
        :meth:`with_geometry` first on non-static results.

        Returns:
            A ``geopandas.GeoDataFrame`` built from :attr:`data`.

        Raises:
            TypeError: If :attr:`has_geometry` is ``False``.
        """
        if not self.has_geometry:
            raise TypeError(
                f"Result.gdf: This Result (resolution={self.resolution!r}) has no geometry. "
                "Call .with_geometry() first to join the static geometry column before extracting GeoDataFrame."
            )
        import geopandas as gpd
        import shapely.wkb as wkb

        geometry_col = self.entity.geometry_col
        geometries = [wkb.loads(b) for b in self.data[geometry_col].to_list()]
        geo_series = gpd.GeoSeries(geometries, crs="EPSG:4326")
        return gpd.GeoDataFrame(
            self.data.drop(geometry_col).to_pandas(),
            geometry=geo_series,
        )

    def lazy(self) -> pl.LazyFrame:
        """Return a ``pl.LazyFrame`` for arbitrary further transformations.

        This is an escape hatch for operations not covered by the Result API.
        The resulting ``LazyFrame`` is disconnected from ``Result`` — callers
        are responsible for collecting and wrapping the output themselves.

        Returns:
            A lazy frame backed by :attr:`data`.
        """
        return self.data.lazy()

    def with_geometry(self) -> "Result":
        """Return a new ``Result`` with the static geometry column joined in.

        Reads only the key and geometry columns from the entity's static file
        (no full scan), joins on :attr:`key_cols`, and returns a new
        ``Result`` with ``has_geometry=True``.

        This is the intended path for attaching coordinates to annual or
        fortnightly results before calling :meth:`gdf` or
        ``result.plot.choropleth()``.

        Returns:
            A new :class:`Result` with the geometry column merged in and
            ``has_geometry=True``.  If ``has_geometry`` is already ``True``,
            returns ``self`` unchanged.
        """
        if self.has_geometry:
            return self

        geom_col = self.entity.geometry_col
        key_cols = self.key_cols
        static_path = self.entity._resolve(self.entity.static_path)

        geo_df = collect_lf(
            pl.scan_parquet(static_path)
            .select(key_cols + [geom_col])
            .filter(
                pl.col(key_cols[0]).is_in(self.data[key_cols[0]].to_list())
                if len(key_cols) == 1
                else pl.lit(True)
            )
        )

        joined = self.data.join(geo_df, on=key_cols, how="left")
        return self._replace(data=joined, has_geometry=True)

    def derive(self, name: str, expr: pl.Expr) -> "Result":
        """Return a new ``Result`` with a computed column appended.

        The derived column is a regular Polars column — no special tagging.
        It is fully chainable::

            result.derive("ndwi", (pl.col("green") - pl.col("nir")) / (pl.col("green") + pl.col("nir")))
                  .derive("drought_flag", pl.when(pl.col("rainfall") < 500).then(1).otherwise(0))

        Args:
            name: Name for the new column.
            expr: A Polars expression that evaluates to the column values.

        Returns:
            A new :class:`Result` with ``name`` appended to :attr:`data`.
        """
        new_data = self.data.with_columns(expr.alias(name))
        return self._replace(data=new_data)

    def aggregate(self, *exprs: pl.Expr, by: str | None = None) -> "Result":
        """Return a new ``Result`` with the data grouped and aggregated.

        The ``by`` parameter controls the grouping dimension.  Not all
        groupings are valid for every resolution — the matrix below is
        enforced at call time:

        .. list-table::
           :header-rows: 1

           * - ``by``
             - static
             - annual
             - fortnightly
           * - ``None``
             - ❌
             - ✅
             - ✅
           * - ``"year"``
             - ❌
             - ✅
             - ✅
           * - ``"month"`` / ``"year_month"`` / ``"season"`` / ``"season_year"``
             - ❌
             - ❌
             - ✅

        Args:
            *exprs: One or more Polars aggregation expressions
                (e.g. ``pl.mean("ndvi")``, ``pl.max("rainfall")``).
            by: Grouping dimension.  ``None`` collapses all rows to one.
                ``"year"`` groups by entity + year and is valid for both
                annual and fortnightly resolution.
                Other temporal groupings (``"month"``, ``"year_month"``,
                ``"season"``, ``"season_year"``) require
                ``resolution="fortnightly"``.

        Returns:
            A new :class:`Result` whose ``data`` is the aggregated frame.

        Raises:
            ValueError: If ``by`` is incompatible with :attr:`resolution`, or
                if ``by`` is not a recognised grouping name.
        """
        if by is not None and by not in _VALID_BY:
            raise ValueError(
                f"Result.aggregate: Unknown grouping by={by!r}. "
                f"Valid options: {sorted(v for v in _VALID_BY if v is not None)}."
            )

        if self.resolution == Resolution.STATIC:
            raise ValueError(
                "Result.aggregate: Aggregation is not supported on static results. "
                "Static data has one row per entity with no time dimension to collapse."
            )

        if by in _FORTNIGHTLY_ONLY_BY and self.resolution != Resolution.FORTNIGHTLY:
            raise ValueError(
                f"Result.aggregate: Grouping by={by!r} requires data at fortnightly resolution, "
                f"but this result has resolution={self.resolution!r}."
            )

        if by is None:
            new_data = self.data.group_by(self.key_cols).agg(*exprs)
        else:
            # Temporal grouping columns are expected to already exist on the
            # frame (added by the materialisation layer from the time column).
            group_cols = self.key_cols + [by]
            new_data = self.data.group_by(group_cols).agg(*exprs)

        return self._replace(data=new_data)

    @property
    def stats(self) -> "StatsNamespace":
        """Return the statistical analysis namespace for this result.

        All methods on this namespace return a fresh :class:`Result` with
        computed values in ``data`` and method parameters in ``metadata``.

        Example::

            result.stats.describe()
            result.stats.correlate(["ndvi", "rainfall"], method="spearman")
            result.stats.anomaly("ndvi", mode="cross_sectional", method="zscore")
        """
        from core_lens.base.namespaces.stats import StatsNamespace

        return StatsNamespace(self)

    @property
    def plot(self) -> "PlotNamespace":
        """Return the visualisation namespace for this result.

        Methods here return Lonboard or Plotly objects.

        Example::

            result.plot.choropleth("ndvi")
            result.plot.timeseries(x="year", y="rainfall")
        """
        from core_lens.base.namespaces.plot import PlotNamespace

        return PlotNamespace(self)

    def _replace(self, **overrides: Any) -> "Result":
        # Thin copy-with-modification helper to keep the public methods clean.
        # metadata is intentionally carried forward so chained derive() calls
        # preserve stats context from an earlier step.
        return Result(
            data=overrides.get("data", self.data),
            resolution=overrides.get("resolution", self.resolution),
            has_geometry=overrides.get("has_geometry", self.has_geometry),
            key_cols=overrides.get("key_cols", self.key_cols),
            entity_name=overrides.get("entity_name", self.entity_name),
            entity=overrides.get("entity", self.entity),
            metadata=overrides.get("metadata", self.metadata),
        )
