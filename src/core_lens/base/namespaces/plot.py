"""Visualisation namespace for Results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import lonboard
    from core_lens.base.result import Result


def _apply_theme(fig: Any, result: "Result", title: str) -> None:
    """Apply standard CoreLens theme to a Plotly figure."""
    entity_name = getattr(result.entity, "__name__", "Unknown")
    if hasattr(result, "entity_name"):
        entity_name = result.entity_name

    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        title=title,
        legend_title_text="Entity",
        hovermode="closest",
        margin=dict(b=60),
        annotations=[
            dict(
                text=f"Entity: {entity_name} | Visualised with Plotly",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=1.0,
                y=-0.18,
                xanchor="right",
                yanchor="top",
                font=dict(size=10, color="gray"),
            )
        ],
    )


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

        cmap = mpl.colormaps["plasma"]

        layer = lonboard.PolygonLayer.from_geopandas(
            gdf,
            get_fill_color=apply_continuous_cmap(norm_values, cmap),
        )
        return lonboard.Map(layers=[layer])

    def timeseries(
        self,
        x: str | None = None,
        y: str | list[str] | None = None,
        top_n: int = 10,
        aggregate: bool = False,
    ) -> Any:
        """Render a timeseries line chart using Plotly.

        Args:
            x: The temporal column to plot on the x-axis.
            y: The value column(s) to plot on the y-axis. If None, auto-selects all numeric.
            top_n: Maximum number of entities to plot (if aggregate=False).
            aggregate: If True, plots the mean of y across all entities over time.

        Returns:
            A Plotly Figure object.
        """
        import polars as pl
        import polars.selectors as cs
        import plotly.express as px
        import plotly.graph_objects as go

        if x is None:
            raise ValueError("x must be provided for timeseries plot.")

        df = self.result.df()
        if x not in df.columns:
            raise ValueError(f"Column {x!r} not found in Result.")

        y_cols = [y] if isinstance(y, str) else y
        if y_cols is None:
            y_cols = df.select(cs.numeric()).columns
            if x in y_cols:
                y_cols.remove(x)

        if not y_cols:
            raise ValueError("No numeric columns found for y-axis.")

        key_col = self.result.key_cols[0]
        hover_cols = [c for c in df.columns if c not in ("geometry", "geom")]

        if aggregate:
            agg_df = (
                df.group_by(x)
                .agg([pl.col(c).mean() for c in y_cols])
                .sort(x)
                .to_pandas()
            )
            fig = px.line(agg_df, x=x, y=y_cols, markers=True)
            _apply_theme(fig, self.result, "Timeseries (Aggregated)")
            return fig

        unique_entities = df[key_col].unique()
        if len(unique_entities) > top_n:
            entities = unique_entities.limit(top_n).to_list()
            df = df.filter(pl.col(key_col).is_in(entities))

        df = df.sort(x)
        df_pd = df.to_pandas()

        fig = go.Figure()
        num_entities = len(df_pd[key_col].unique())

        for i, y_col in enumerate(y_cols):
            sub_fig = px.line(
                df_pd, x=x, y=y_col, color=key_col, hover_data=hover_cols, markers=True
            )
            if i == 0:
                fig.update_layout(sub_fig.layout)
            for trace in sub_fig.data:
                trace.visible = i == 0
                fig.add_trace(trace)

        if len(y_cols) > 1:
            buttons = []
            for i, y_col in enumerate(y_cols):
                visibility = [False] * (len(y_cols) * num_entities)
                for j in range(num_entities):
                    visibility[i * num_entities + j] = True
                buttons.append(
                    dict(
                        method="update",
                        label=y_col,
                        args=[{"visible": visibility}, {"yaxis.title.text": y_col}],
                    )
                )
            fig.update_layout(
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons,
                        x=0.0,
                        y=1.15,
                        xanchor="left",
                        yanchor="top",
                    )
                ],
            )

        _apply_theme(fig, self.result, "Timeseries")
        fig.update_layout(yaxis_title=y_cols[0])
        return fig

    def scatter(
        self, x: str | None = None, y: str | list[str] | None = None, top_n: int = 10
    ) -> Any:
        """Render a scatter plot using Plotly.

        Args:
            x: The column to plot on the x-axis.
            y: The column(s) to plot on the y-axis. If None, auto-selects all numeric.
            top_n: Maximum number of entities to plot.

        Returns:
            A Plotly Figure object.
        """
        import polars as pl
        import polars.selectors as cs
        import plotly.express as px
        import plotly.graph_objects as go

        if x is None:
            raise ValueError("x must be provided for scatter plot.")

        df = self.result.df()
        if x not in df.columns:
            raise ValueError(f"Column {x!r} not found in Result.")

        y_cols = [y] if isinstance(y, str) else y
        if y_cols is None:
            y_cols = df.select(cs.numeric()).columns
            if x in y_cols:
                y_cols.remove(x)

        if not y_cols:
            raise ValueError("No numeric columns found for y-axis.")

        key_col = self.result.key_cols[0]
        hover_cols = [c for c in df.columns if c not in ("geometry", "geom")]

        unique_entities = df[key_col].unique()
        if len(unique_entities) > top_n:
            entities = unique_entities.limit(top_n).to_list()
            df = df.filter(pl.col(key_col).is_in(entities))

        df_pd = df.to_pandas()
        fig = go.Figure()
        num_entities = len(df_pd[key_col].unique())

        for i, y_col in enumerate(y_cols):
            sub_fig = px.scatter(
                df_pd,
                x=x,
                y=y_col,
                color=key_col,
                hover_data=hover_cols,
                marginal_x="histogram",
                marginal_y="histogram",
                opacity=0.7,
            )
            if i == 0:
                fig.update_layout(sub_fig.layout)
            for trace in sub_fig.data:
                trace.visible = i == 0
                fig.add_trace(trace)

        if len(y_cols) > 1:
            buttons = []
            traces_per_y = num_entities * 3
            for i, y_col in enumerate(y_cols):
                visibility = [False] * (len(y_cols) * traces_per_y)
                for j in range(traces_per_y):
                    visibility[i * traces_per_y + j] = True
                buttons.append(
                    dict(
                        method="update",
                        label=y_col,
                        args=[{"visible": visibility}, {"yaxis.title.text": y_col}],
                    )
                )
            fig.update_layout(
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons,
                        x=0.0,
                        y=1.15,
                        xanchor="left",
                        yanchor="top",
                    )
                ],
            )

        _apply_theme(fig, self.result, "Scatter Distribution")
        fig.update_layout(yaxis_title=y_cols[0], barmode="overlay")
        return fig

    def distribution(self, x: str | list[str] | None = None, top_n: int = 10) -> Any:
        """Render a distribution (histogram) plot using Plotly.

        Args:
            x: The column(s) to plot the distribution for. If None, auto-selects all numeric.
            top_n: Maximum number of entities to include.

        Returns:
            A Plotly Figure object.
        """
        import polars as pl
        import polars.selectors as cs
        import plotly.express as px
        import plotly.graph_objects as go

        df = self.result.df()

        x_cols = [x] if isinstance(x, str) else x
        if x_cols is None:
            x_cols = df.select(cs.numeric()).columns

        if not x_cols:
            raise ValueError("No numeric columns found for distribution.")

        key_col = self.result.key_cols[0]
        hover_cols = [c for c in df.columns if c not in ("geometry", "geom")]

        unique_entities = df[key_col].unique()
        if len(unique_entities) > top_n:
            entities = unique_entities.limit(top_n).to_list()
            df = df.filter(pl.col(key_col).is_in(entities))

        df_pd = df.to_pandas()
        fig = go.Figure()
        num_entities = len(df_pd[key_col].unique())

        for i, x_col in enumerate(x_cols):
            sub_fig = px.histogram(
                df_pd,
                x=x_col,
                color=key_col,
                marginal="box",
                hover_data=hover_cols,
                opacity=0.7,
                barmode="overlay",
            )
            if i == 0:
                fig.update_layout(sub_fig.layout)
            for trace in sub_fig.data:
                trace.visible = i == 0
                fig.add_trace(trace)

        if len(x_cols) > 1:
            buttons = []
            traces_per_x = num_entities * 2
            for i, x_col in enumerate(x_cols):
                visibility = [False] * (len(x_cols) * traces_per_x)
                for j in range(traces_per_x):
                    visibility[i * traces_per_x + j] = True
                buttons.append(
                    dict(
                        method="update",
                        label=x_col,
                        args=[{"visible": visibility}, {"xaxis.title.text": x_col}],
                    )
                )
            fig.update_layout(
                updatemenus=[
                    dict(
                        active=0,
                        buttons=buttons,
                        x=0.0,
                        y=1.15,
                        xanchor="left",
                        yanchor="top",
                    )
                ],
            )

        _apply_theme(fig, self.result, "Distribution")
        fig.update_layout(xaxis_title=x_cols[0], barmode="overlay")
        return fig

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
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig.update_traces(texttemplate="%{z:.2f}")
        _apply_theme(fig, self.result, "Correlation Matrix")
        return fig

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
        fig = px.imshow(pdf, aspect="auto", color_continuous_scale="Viridis")
        _apply_theme(fig, self.result, f"Heatmap: {value}")
        return fig

    def matrix(self) -> Any:
        """Render a scatter matrix (pairs plot) using Plotly.

        Returns:
            A Plotly Figure object.
        """
        import polars.selectors as cs
        import plotly.express as px

        df = self.result.df()
        key_col = self.result.key_cols[0]
        hover_cols = [c for c in df.columns if c not in ("geometry", "geom")]
        df_num = df.select(cs.numeric() | cs.by_name(key_col))
        hover_cols = df_num.columns

        fig = px.scatter_matrix(
            df_num.to_pandas(), color=key_col, hover_data=hover_cols, opacity=0.7
        )
        _apply_theme(fig, self.result, "Scatter Matrix")
        return fig
