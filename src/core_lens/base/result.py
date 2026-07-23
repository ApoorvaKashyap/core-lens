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

_SUB_ANNUAL_ONLY_BY = {"month", "year_month", "season", "season_year"}
_ANNUAL_OR_SUB_ANNUAL_BY = {"year"}
_VALID_BY = {None} | _SUB_ANNUAL_ONLY_BY | _ANNUAL_OR_SUB_ANNUAL_BY


class Result:
    """Concrete, immutable result of a materialised :class:`~core_lens.base.view.View`.

    ``Result`` is the shared output type for every entity and every
    materialisation path.  It wraps a ``pl.LazyFrame``, carries enough
    context to know how to re-attach geometry, and exposes compute methods
    that always return a fresh ``Result``, keeping the chain composable::

        result = aoi.mws.where(tehsil="Pangi").annual.between("2010-01-01", "2023-12-31")
        mean_ndvi = result.aggregate(pl.mean("ndvi"), by="year")
        mean_ndvi.plot.timeseries(x="year", y="ndvi")

    Attributes:
        data: The underlying ``pl.LazyFrame``. Use :meth:`df` to materialise it
            into a ``pl.DataFrame``. All compute methods produce a new
            ``Result`` whose ``data`` is the transformed frame.
        metadata: Free-form dict populated by stats methods to carry
            method parameters (e.g. ``{"method": "pearson", "p_value": 0.003}``).
            Empty on freshly materialised results.
        resolution: A :class:`~core_lens.schema.profile.Resolution` member
            (``STATIC``, ``ANNUAL``, or ``SUB_ANNUAL``).  Used to validate
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
        data: pl.DataFrame | pl.LazyFrame,
        resolution: Resolution,
        has_geometry: bool,
        key_cols: list[str],
        entity_name: str,
        entity: "BaseEntity",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Result.

        Args:
            data: The underlying dataframe or lazyframe.
            resolution: The spatial and temporal resolution of the data.
            has_geometry: True if the geometry column is populated.
            key_cols: Primary key columns linking back to the entity.
            entity_name: Registered name of the parent entity.
            entity: The BaseEntity object that produced this result.
            metadata: Additional metadata dictionary to store.
        """
        self.data = data.lazy() if isinstance(data, pl.DataFrame) else data
        self.resolution = resolution
        self.has_geometry = has_geometry
        self.columns: list[str] = (
            data.columns
            if isinstance(data, pl.DataFrame)
            else data.collect_schema().names()
        )
        self.key_cols = key_cols
        self.entity_name = entity_name
        self.entity = entity
        self.metadata: dict[str, Any] = metadata if metadata is not None else {}

    def df(self) -> pl.DataFrame:
        """Return the underlying ``pl.DataFrame``.

        Returns:
            pl.DataFrame: The materialised data frame.

        """
        return collect_lf(self.data)

    def materialise(self) -> "Result":
        """Evaluate the lazy computation graph and cache it in memory.

        Returns:
            Result: A new Result with the data cached as an in-memory LazyFrame.

        """
        return self._replace(data=self.df().lazy())

    def gdf(self) -> "gpd.GeoDataFrame":
        """Return the data as a ``GeoDataFrame``.

        Only valid when :attr:`has_geometry` is ``True``.  Use
        :meth:`with_geometry` first on non-static results.

        Returns:
            gpd.GeoDataFrame: A ``geopandas.GeoDataFrame`` built from :attr:`data`.

        Raises:
            TypeError: If :attr:`has_geometry` is ``False``.

        """
        if not self.has_geometry:
            raise TypeError(
                f"Result.gdf: This Result (resolution={self.resolution!r}) has no geometry. "
                "Call .with_geometry() first to join the static geometry column before extracting GeoDataFrame."
            )
        import geopandas as gpd
        import shapely

        geometry_col = self.entity.geometry_col
        # Vectorised C-level decode — shapely.from_wkb operates on the whole
        # numpy array at once, avoiding a Python-loop per row.
        df = self.df()
        geometries = shapely.from_wkb(df[geometry_col].to_numpy())
        geo_series = gpd.GeoSeries(geometries, crs="EPSG:4326")
        return gpd.GeoDataFrame(
            df.drop(geometry_col).to_pandas(),
            geometry=geo_series,
        )

    def lazy(self) -> pl.LazyFrame:
        """Return a ``pl.LazyFrame`` for arbitrary further transformations.

        This is an escape hatch for operations not covered by the Result API.
        The resulting ``LazyFrame`` is disconnected from ``Result`` — callers
        are responsible for collecting and wrapping the output themselves.

        Returns:
            pl.LazyFrame: A lazy frame backed by :attr:`data`.

        """
        return self.data

    def with_geometry(self) -> "Result":
        """Return a new ``Result`` with the static geometry column joined in.

        Reads only the key and geometry columns from the entity's static file
        (no full scan), joins on :attr:`key_cols`, and returns a new
        ``Result`` with ``has_geometry=True``.

        This is the intended path for attaching coordinates to annual or
        sub_annual results before calling :meth:`gdf` or
        ``result.plot.choropleth()``.

        Returns:
            Result: A new :class:`Result` with the geometry column merged in and
            ``has_geometry=True``.  If ``has_geometry`` is already ``True``,
            returns ``self`` unchanged.

        """
        if self.has_geometry:
            return self

        key_cols = self.key_cols

        # 1. Evaluate the heavy aggregation graph ONCE to prevent Polars from
        # evaluating it twice during the geometry extraction and join.
        res = self.materialise()
        keys_df = res.df().select(key_cols)

        # 2. Push down the geometry filter into the Parquet reader using is_in
        # so we don't scan the entire static geometry file (which takes ~0.4s).
        geo_lf = self.entity.geometry_lazy
        if len(key_cols) == 1 and keys_df.height < 50000:
            key = key_cols[0]
            geo_lf = geo_lf.filter(pl.col(key).is_in(keys_df[key].to_list()))
        else:
            geo_lf = geo_lf.join(keys_df.lazy(), on=key_cols, how="semi")

        joined = res.data.join(geo_lf, on=key_cols, how="left")

        # 3. Cache the joined geometry so multiple plots are instant.
        return res._replace(data=joined, has_geometry=True).materialise()

    def derive(self, name: str, expr: pl.Expr) -> "Result":
        """Return a new ``Result`` with a computed column appended.

        The derived column is a regular Polars column — no special tagging.
        It is fully chainable::

            result.derive("ndwi", (pl.col("green") - pl.col("nir")) / (pl.col("green") + pl.col("nir")))
                  .derive("drought_flag", pl.when(pl.col("rainfall") < 500).then(1).otherwise(0))

        Args:
            name (str): Name for the new column.
            expr (pl.Expr): A Polars expression that evaluates to the column values.

        Returns:
            Result: A new :class:`Result` with ``name`` appended to :attr:`data`.

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
             - sub_annual
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
            *exprs (pl.Expr): One or more Polars aggregation expressions
                (e.g. ``pl.mean("ndvi")``, ``pl.max("rainfall")``).
            by (str | None, optional): Grouping dimension.  ``None`` collapses all rows to one.
                ``"year"`` groups by entity + year and is valid for both
                annual and sub_annual resolution.
                Other temporal groupings (``"month"``, ``"year_month"``,
                ``"season"``, ``"season_year"``) require
                ``resolution="sub_annual"``.

        Returns:
            Result: A new :class:`Result` whose ``data`` is the aggregated frame.

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

        if by in _SUB_ANNUAL_ONLY_BY and self.resolution != Resolution.SUB_ANNUAL:
            raise ValueError(
                f"Result.aggregate: Grouping by={by!r} requires data at sub_annual resolution, "
                f"but this result has resolution={self.resolution!r}."
            )

        if by is None:
            group_cols = self.key_cols
        else:
            # Temporal grouping columns are expected to already exist on the
            # frame (added by the materialisation layer from the time column).
            group_cols = self.key_cols + [by]

        # Polars group_by throws DuplicateError if an aggregation expression
        # outputs a column name that is already present in the grouping keys.
        # Filter them out to prevent crashes when users iterate over all columns.
        safe_exprs = [
            expr
            for expr in exprs
            if getattr(expr, "meta", None) is None
            or expr.meta.output_name() not in group_cols
        ]

        new_data = self.data.group_by(group_cols).agg(*safe_exprs)

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

        Returns:
            StatsNamespace: The statistical analysis namespace.

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

        Returns:
            PlotNamespace: The visualisation namespace.

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
