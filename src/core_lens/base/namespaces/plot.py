"""Visualisation namespace for Results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import lonboard
    from core_lens.base.result import Result


class PlotNamespace:
    """Visualisation namespace for :class:`~core_lens.base.result.Result`.

    Methods return map or chart objects.

    Attributes:
        result: The parent Result object.
    """

    def __init__(self, result: "Result") -> None:
        self.result = result

    def choropleth(self, column: str, subplot_on: str | None = None) -> "lonboard.Map":
        """Render an interactive choropleth map using Lonboard.

        If the Result does not already have geometry, it will be attached
        automatically via :meth:`~core_lens.base.result.Result.with_geometry`.

        Args:
            column: The column to use for colour mapping.
            subplot_on: Optional temporal column to split data across.

        Returns:
            A Lonboard Map object.
        """
        import lonboard
        from lonboard.colormap import apply_continuous_cmap

        if subplot_on is not None:
            raise NotImplementedError("subplot_on is not yet supported for choropleth.")

        res = self.result if self.result.has_geometry else self.result.with_geometry()
        gdf = res.gdf()

        import matplotlib as mpl
        import numpy as np

        if column not in gdf.columns:
            raise ValueError(f"Column {column!r} not found in Result.")

        values = gdf[column].to_numpy()
        v_min, v_max = np.nanmin(values), np.nanmax(values)
        if v_max > v_min:
            norm_values = (values - v_min) / (v_max - v_min)
        else:
            norm_values = np.zeros_like(values)

        cmap = mpl.colormaps["viridis"]

        layer = lonboard.PolygonLayer.from_geopandas(
            gdf,
            get_fill_color=apply_continuous_cmap(norm_values, cmap),
        )
        return lonboard.Map(layers=[layer])

    def timeseries(
        self,
        x: str | None = None,
        y: str | None = None,
        top_n: int = 10,
        aggregate: bool = False,
    ) -> Any:
        """Render a timeseries line chart using Plotly.

        Args:
            x: The temporal column to plot on the x-axis.
            y: The value column to plot on the y-axis.
            top_n: Maximum number of entities to plot (if aggregate=False).
            aggregate: If True, plots the mean of y across all entities over time.

        Returns:
            A Plotly Figure object.
        """
        import polars as pl
        import plotly.express as px

        if x is None or y is None:
            raise ValueError("Both x and y must be provided for timeseries plot.")

        df = self.result.df()

        if x not in df.columns or y not in df.columns:
            raise ValueError(f"Columns {x!r} and/or {y!r} not found in Result.")

        if aggregate:
            agg_df = df.group_by(x).agg(pl.col(y).mean()).sort(x)
            return px.line(agg_df.to_pandas(), x=x, y=y)

        key_col = self.result.key_cols[0]
        unique_entities = df[key_col].unique()

        if len(unique_entities) > top_n:
            entities = unique_entities.limit(top_n).to_list()
            df = df.filter(pl.col(key_col).is_in(entities))

        df = df.sort(x)
        return px.line(df.to_pandas(), x=x, y=y, color=key_col)

    def scatter(
        self, x: str | None = None, y: str | None = None, top_n: int = 10
    ) -> Any:
        """Render a scatter plot using Plotly.

        Args:
            x: The column to plot on the x-axis.
            y: The column to plot on the y-axis.
            top_n: Maximum number of entities to plot.

        Returns:
            A Plotly Figure object.
        """
        import polars as pl
        import plotly.express as px

        if x is None or y is None:
            raise ValueError("Both x and y must be provided for scatter plot.")

        df = self.result.df()
        if x not in df.columns or y not in df.columns:
            raise ValueError(f"Columns {x!r} and/or {y!r} not found in Result.")

        key_col = self.result.key_cols[0]
        unique_entities = df[key_col].unique()
        if len(unique_entities) > top_n:
            entities = unique_entities.limit(top_n).to_list()
            df = df.filter(pl.col(key_col).is_in(entities))

        return px.scatter(df.to_pandas(), x=x, y=y, color=key_col)

    def distribution(self, x: str | None = None, top_n: int = 10) -> Any:
        """Render a distribution (histogram) plot using Plotly.

        Args:
            x: The column to plot the distribution for.
            top_n: Maximum number of entities to include.

        Returns:
            A Plotly Figure object.
        """
        import polars as pl
        import plotly.express as px

        if x is None:
            raise ValueError("x must be provided for distribution plot.")

        df = self.result.df()
        if x not in df.columns:
            raise ValueError(f"Column {x!r} not found in Result.")

        key_col = self.result.key_cols[0]
        unique_entities = df[key_col].unique()
        if len(unique_entities) > top_n:
            entities = unique_entities.limit(top_n).to_list()
            df = df.filter(pl.col(key_col).is_in(entities))

        return px.histogram(df.to_pandas(), x=x, color=key_col, marginal="box")

    def correlation(self, columns: list[str] | None = None, top_n: int = 10) -> Any:
        """Render a correlation heatmap using Plotly.

        Args:
            columns: Optional list of columns to correlate. Defaults to all numeric.
            top_n: Maximum number of entities to include (currently unused).

        Returns:
            A Plotly Figure object.
        """
        import polars.selectors as cs
        import plotly.express as px

        df = self.result.df()
        if columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in Result.")
            df = df.select(columns)
        else:
            df = df.select(cs.numeric())

        corr = df.to_pandas().corr()
        return px.imshow(corr, text_auto=True, aspect="auto")

    def heatmap(
        self,
        x: str | None = None,
        y: str | None = None,
        value: str | None = None,
        top_n: int = 10,
    ) -> Any:
        """Render a heatmap using Plotly.

        Args:
            x: The column for the x-axis.
            y: The column for the y-axis.
            value: The column for the colour values.
            top_n: Maximum number of entities to include (currently unused).

        Returns:
            A Plotly Figure object.
        """
        import plotly.express as px

        if x is None or y is None or value is None:
            raise ValueError("x, y, and value must be provided for heatmap.")

        df = self.result.df()
        for col in (x, y, value):
            if col not in df.columns:
                raise ValueError(f"Column {col!r} not found in Result.")

        pivot_df = df.pivot(values=value, index=y, on=x, aggregate_function="mean")
        pdf = pivot_df.to_pandas().set_index(y)
        return px.imshow(pdf, aspect="auto")

    def matrix(self) -> Any:
        """Render a scatter matrix (pairs plot) using Plotly.

        Returns:
            A Plotly Figure object.
        """
        import polars.selectors as cs
        import plotly.express as px

        df = self.result.df()
        df = df.select(cs.numeric())

        return px.scatter_matrix(df.to_pandas())
