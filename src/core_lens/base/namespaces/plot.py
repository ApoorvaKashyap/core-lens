"""Visualisation namespace for Results."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any


class SubplotOn(Enum):
    """Temporal dimensions to split plot data across.

    Used to specify how data should be partitioned into subplots or map layers
    based on temporal columns added during materialisation.

    Attributes:
        YEAR: Split data by year.
        MONTH: Split data by month.
        SEASON: Split data by meteorological season.
        SEASON_YEAR: Split data by season and year.

    """

    YEAR = "year"
    MONTH = "month"
    SEASON = "season"
    SEASON_YEAR = "season_year"


if TYPE_CHECKING:
    from bokeh.plotting import figure as BokehFigure
    from core_lens.base.result import Result


_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _wkb_to_arrow_table(
    df: Any,
    geom_col: str,
    extra_cols: list[str] | None = None,
) -> Any:
    """Convert a Polars DataFrame with a WKB geometry column to a GeoArrow PyArrow Table.

    Uses ``geoarrow.rust.core.from_wkb`` — a zero-copy C extension bundled
    transitively by lonboard via ``geoarrow-rust-core``.  No Shapely objects
    are ever instantiated for the common case (Polygon / MultiPolygon rows).

    GeometryCollection rows are handled by extracting their polygon parts via
    Shapely, matching the behaviour of the previous GeoPandas code path.  Because
    GeometryCollections are rare in practice the Shapely overhead is bounded.

    WKB type byte encoding (after the 1-byte endian flag):
        1=Point, 2=LineString, 3=Polygon, 4=MultiPoint, 5=MultiLineString,
        6=MultiPolygon, 7=GeometryCollection.

    Args:
        df: A Polars DataFrame containing the WKB binary column.
        geom_col: Name of the WKB binary column in *df*.\
        extra_cols: Additional columns to include in the output table.

    Returns:
        An ``arro3.core.Table`` whose geometry column is GeoArrow-typed.

    """
    import geoarrow.rust.core as ga
    import polars as pl

    keep = [geom_col] + (extra_cols or [])
    df = df.select([c for c in keep if c in df.columns])

    # WKB type detection: slice byte 1 (skip endian flag) — single byte covers
    # all standard types 1-7.  bin.slice returns BinaryView in this Polars build
    # which cannot be cast to UInt8 directly, so we extract the int value via
    # map_elements on the already-sliced 1-byte series (no endian logic, no
    # int.from_bytes on 4 bytes — still much cheaper than the original path).
    geom_type_series = (
        df[geom_col]
        .bin.slice(1, 1)
        .map_elements(lambda b: b[0] if b else 0, return_dtype=pl.UInt8)
    )

    # --- Fast path: pure Polygon / MultiPolygon rows (no Shapely) ---
    poly_mask = geom_type_series.is_in([3, 6])
    df_poly = df.filter(poly_mask)

    # --- Slow path: GeometryCollection rows — extract polygon parts via Shapely ---
    gc_mask = geom_type_series == 7
    df_gc = df.filter(gc_mask)

    if df_gc.height > 0:
        import shapely.wkb as swkb
        from shapely.geometry import MultiPolygon

        repaired_rows: list[dict[str, Any]] = []
        other_cols = [c for c in df_gc.columns if c != geom_col]

        for row in df_gc.iter_rows(named=True):
            raw_wkb = row[geom_col]
            geom = swkb.loads(raw_wkb)
            polys = [
                g
                for g in getattr(geom, "geoms", [])
                if g.geom_type in ("Polygon", "MultiPolygon")
            ]
            if not polys:
                continue
            merged = MultiPolygon(polys) if len(polys) > 1 else polys[0]
            new_row = {geom_col: swkb.dumps(merged)}
            for col in other_cols:
                new_row[col] = row[col]
            repaired_rows.append(new_row)

        if repaired_rows:
            schema = {geom_col: pl.Binary}
            for col in other_cols:
                schema[col] = df_gc[col].dtype
            df_gc_repaired = pl.DataFrame(repaired_rows, schema=schema)
            df_poly = pl.concat([df_poly, df_gc_repaired], how="diagonal")

    import arro3.core as ac

    if df_poly.height == 0:
        # Return an empty table with the correct schema rather than crashing.
        arrow_table = df_poly.to_arrow()
        return ac.Table.from_arrow(arrow_table)

    arrow_table = df_poly.to_arrow()
    del (
        df_poly
    )  # free Polars WKB buffer before PyArrow combine_chunks() allocates its copy
    arrow_table = arrow_table.combine_chunks()
    wkb_col = arrow_table.column(geom_col).combine_chunks()
    geo_col = ga.from_wkb(wkb_col)

    idx = arrow_table.schema.get_field_index(geom_col)
    schema = arrow_table.schema.remove(idx)

    import json

    arr_geo = ac.Array(geo_col)
    ac_geo = ac.ChunkedArray([arr_geo])
    geom_field = arr_geo.field.with_name(geom_col)

    ext_meta = dict(geom_field.metadata or {})
    geo_meta = {}
    if b"ARROW:extension:metadata" in ext_meta:
        try:
            geo_meta = json.loads(ext_meta[b"ARROW:extension:metadata"].decode("utf-8"))
        except Exception:  # noqa: S110
            pass
    geo_meta["crs"] = "EPSG:4326"
    ext_meta[b"ARROW:extension:metadata"] = json.dumps(geo_meta).encode("utf-8")
    new_geom_field = geom_field.with_metadata(ext_meta)

    new_fields = [
        ac.Field.from_arrow(schema.field(i)) for i in range(len(schema.names))
    ]
    new_fields.insert(idx, new_geom_field)
    new_schema = ac.Schema(new_fields)

    tbl_cols = [ac.ChunkedArray(arrow_table.column(c)) for c in schema.names]
    tbl_cols.insert(idx, ac_geo)

    tbl = ac.Table.from_arrays(tbl_cols, schema=new_schema)
    return tbl


def _color_for(index: int) -> str:
    """Get a color from the palette by index.

    Args:
        index: The index of the color.

    Returns:
        The hex color string.

    """
    return _PALETTE[index % len(_PALETTE)]


def _apply_theme(fig: Any, result: "Result", title: str) -> None:
    """Apply standard CoreLens theme to a Bokeh figure.

    Args:
        fig: The Bokeh figure.
        result: The parent Result object.
        title: The title text.

    """
    entity_name = getattr(result.entity, "__name__", "Unknown")
    if hasattr(result, "entity_name"):
        entity_name = result.entity_name

    fig.title.text = title
    fig.title.text_font = "Inter, sans-serif"
    fig.title.text_font_size = "14pt"

    from bokeh.models.annotations import Title

    subtitle = Title(
        text=f"Entity: {entity_name} | Visualised with Bokeh",
        text_font_size="9pt",
        text_color="gray",
        align="right",
        text_font="Inter, sans-serif",
    )
    fig.add_layout(subtitle, "below")


class MapWithLegend:
    """Wrapper around lonboard.Map that adds a legend for Jupyter and HTML exports."""

    def __init__(
        self,
        map_widget: Any,
        cmap: Any,
        v_min: float,
        v_max: float,
        column_name: str,
        clamped_min: bool = False,
        clamped_max: bool = False,
    ) -> None:
        self._map = map_widget
        self._cmap = cmap
        self._v_min = v_min
        self._v_max = v_max
        self._column_name = column_name
        self._clamped_min = clamped_min
        self._clamped_max = clamped_max

    def __getattr__(self, name: str) -> Any:
        return getattr(self._map, name)

    def _generate_legend_div(self, absolute: bool = True) -> str:
        import matplotlib.colors as mcolors
        import numpy as np

        stops = 5
        style = (
            "position:absolute; bottom:35px; left:20px; z-index:999; "
            if absolute
            else ""
        )
        style += (
            "background:white; padding:8px 12px; border-radius:4px; "
            "box-shadow:0 1px 4px rgba(0,0,0,.3); font-family:sans-serif; font-size:12px;"
        )

        html = [
            f'<div style="{style}">',
            f'<b style="margin-bottom:4px;display:inline-block;">{self._column_name}</b><br>',
        ]

        if np.isnan(self._v_min) or np.isnan(self._v_max) or self._v_min >= self._v_max:
            html.append('<span style="color:gray;">No Data</span>')
        else:
            for i in range(stops):
                frac = (stops - 1 - i) / (
                    stops - 1
                )  # Reverse order so highest is on top
                val = self._v_min + frac * (self._v_max - self._v_min)
                rgba = self._cmap(frac)
                hex_color = mcolors.to_hex(rgba)

                prefix = ""
                if i == stops - 1 and self._clamped_min:
                    prefix = "&lt; "
                elif i == 0 and self._clamped_max:
                    prefix = "&gt; "

                html.append(
                    f'<div style="margin-top:2px;">'
                    f'<span style="background:{hex_color};width:12px;height:12px;'
                    f"display:inline-block;margin-right:8px;vertical-align:middle;"
                    f'border:1px solid #ccc;"></span>'
                    f'<span style="vertical-align:middle;">{prefix}{val:.2f}</span></div>'
                )
        html.append("</div>")
        return "".join(html)

    def _ipython_display_(self) -> None:
        import ipywidgets  # type: ignore[import-untyped]
        from IPython.display import display

        legend_div = self._generate_legend_div(absolute=False)
        legend_html = ipywidgets.HTML(legend_div)

        display(ipywidgets.VBox([legend_html, self._map]))  # type: ignore[no-untyped-call]

    def to_html(
        self, filename: str | None = None, title: str | None = None
    ) -> str | None:
        """Export the map to a static HTML string or file, injecting the legend."""
        # lonboard.Map to_html doesn't accept title in all versions,
        # so we conditionally pass it if it's not None or we can just omit title if we aren't sure.
        # But wait, we can just use `self._map.to_html()`
        if title is not None:
            try:
                html_str = self._map.to_html(title=title)
            except TypeError:
                html_str = self._map.to_html()
        else:
            html_str = self._map.to_html()

        legend_div = self._generate_legend_div(absolute=True)

        # Inject right before closing </body> tag
        html_str = html_str.replace("</body>", legend_div + "</body>")

        if filename is not None:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html_str)
            return None
        return str(html_str)


class PlotNamespace:
    """Visualisation namespace for :class:`~core_lens.base.result.Result`.

    Methods return map or chart objects.

    Attributes:
        result: The parent Result object.

    """

    def __init__(self, result: "Result") -> None:
        self.result = result

    def choropleth(
        self,
        column: str,
        subplot_on: SubplotOn | None = None,
        v_min: float | None = None,
        v_max: float | None = None,
    ) -> "MapWithLegend":
        """Render an interactive choropleth map using Lonboard.

        If the Result does not already have geometry, it will be attached
        automatically via :meth:`~core_lens.base.result.Result.with_geometry`.

        Args:
            column (str): The column to use for colour mapping.
            subplot_on (SubplotOn | None, optional): Optional temporal dimension to split data across.
                A :class:`~core_lens.base.namespaces.plot.SubplotOn` enum value. When set, one layer is rendered per unique
                value of ``subplot_on`` in the data.

        Returns:
            lonboard.Map: A Lonboard Map object.

        Raises:
            ValueError: If ``column`` or ``subplot_on`` column not found in data.
            NotImplementedError: Lonboard does not support native subplot grids;
                only single-value ``subplot_on`` is rendered when specified.

        """
        import lonboard
        from lonboard.colormap import apply_continuous_cmap

        if subplot_on is not None and not isinstance(subplot_on, SubplotOn):
            raise ValueError(
                f"PlotNamespace.choropleth: subplot_on must be a SubplotOn enum. "
                f"Valid options: {[e.name for e in SubplotOn]}."
            )

        subplot_col = subplot_on.value if subplot_on is not None else None

        res = self.result if self.result.has_geometry else self.result.with_geometry()

        if column not in res.columns:
            raise ValueError(f"Column {column!r} not found in Result.")

        if subplot_col is not None and subplot_col not in res.columns:
            raise ValueError(
                f"PlotNamespace.choropleth: subplot_on column {subplot_col!r} not found "
                "in Result. Ensure sub_annual data is materialised and temporal columns "
                "are present (they are added automatically by the materialisation layer)."
            )

        import matplotlib as mpl
        import numpy as np
        import polars as pl

        geom_col = res.entity.geometry_col
        extra = [column] + (res.key_cols or [])
        if subplot_col is not None:
            extra.append(subplot_col)

        df = res.df()

        if subplot_col is not None:
            # Render most-recent unique value of subplot_on only.
            # True multi-panel subplots not supported by Lonboard's single-Map API.
            unique_vals = sorted(df[subplot_col].drop_nulls().unique().to_list())
            if unique_vals:
                df = df.filter(pl.col(subplot_col) == unique_vals[-1])

        # --- Direct Polars → GeoArrow → Lonboard path (zero Shapely, zero GeoPandas) ---
        arrow_table = _wkb_to_arrow_table(df, geom_col, extra_cols=extra)

        import warnings

        # Extract colormap values from the already-filtered arrow_table.
        # arrow_table has the same row order as df post-filter, so indices align.
        values = np.asarray(arrow_table.column(column).to_pylist(), dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            actual_min, actual_max = np.nanmin(values), np.nanmax(values)

        calc_v_min = float(v_min) if v_min is not None else float(actual_min)
        calc_v_max = float(v_max) if v_max is not None else float(actual_max)

        clamped_min = v_min is not None and actual_min < v_min
        clamped_max = v_max is not None and actual_max > v_max

        if np.isnan(calc_v_min) or np.isnan(calc_v_max) or calc_v_min >= calc_v_max:
            norm_values = np.where(np.isnan(values), np.nan, 0.0)
        else:
            norm_values = np.clip(
                (values - calc_v_min) / (calc_v_max - calc_v_min), 0.0, 1.0
            )

        cmap = mpl.colormaps["plasma_r"]

        layer = lonboard.PolygonLayer(
            arrow_table,
            get_fill_color=apply_continuous_cmap(norm_values, cmap),
            get_line_color=[255, 255, 255, 180],
            line_width_min_pixels=0.25,
        )
        map_widget = lonboard.Map(layers=[layer])
        return MapWithLegend(
            map_widget, cmap, calc_v_min, calc_v_max, column, clamped_min, clamped_max
        )

    def timeseries(
        self,
        x: str | None = None,
        y: str | list[str] | None = None,
        subplot_on: SubplotOn | str | None = None,
        top_n: int = 10,
        aggregate: bool = False,
    ) -> Any:
        """Render a timeseries line chart using Bokeh.

        Returns a :class:`bokeh.models.Tabs` widget with two panels:
        *Per Entity* (up to ``top_n`` entities) and *Aggregated Mean*.
        When ``aggregate=True``, only the aggregated panel is returned as a
        plain :class:`bokeh.plotting.figure`.

        Args:
            x (str | None, optional): The temporal column to plot on the x-axis.
            y (str | list[str] | None, optional): The value column(s) to plot on the y-axis.
                If ``None``, all numeric columns (except ``x``) are used.
            subplot_on (SubplotOn | str | None, optional): Optional temporal dimension to split data across.
                A :class:`~core_lens.base.namespaces.plot.SubplotOn` enum value or string.
            top_n (int, optional): Maximum number of entities rendered in per-entity view.
            aggregate (bool, optional): If ``True``, renders only the aggregated mean view.

        Returns:
            Any: A Bokeh Tabs object (or a single Figure when ``aggregate=True``).

        """
        import polars as pl
        import polars.selectors as cs
        from bokeh.models import Tabs, TabPanel
        from bokeh.plotting import figure

        if x is None:
            raise ValueError("x must be provided for timeseries plot.")

        df = self.result.df()
        if x not in df.columns:
            raise ValueError(f"Column {x!r} not found in Result.")

        y_cols = [y] if isinstance(y, str) else y
        if y_cols is None:
            y_cols = [c for c in df.select(cs.numeric()).columns if c != x]

        if not y_cols:
            raise ValueError("No numeric columns found for y-axis.")

        subplot_col = None
        if subplot_on is not None:
            if isinstance(subplot_on, SubplotOn):
                subplot_col = subplot_on.value
            else:
                subplot_col = str(subplot_on)
            if subplot_col not in df.columns:
                raise ValueError(
                    f"PlotNamespace.timeseries: subplot_on column {subplot_col!r} not found "
                    "in Result."
                )

        key_col = self.result.key_cols[0]
        y_col = y_cols[0]

        def _make_fig(pdf: Any, title: str) -> "BokehFigure":
            hover_cols = [c for c in pdf.columns if c not in ("geometry", "geom")]
            tooltips = [(c, f"@{{{c}}}") for c in hover_cols]
            fig = figure(  # type: ignore[call-arg]
                width=900,
                height=400,
                x_axis_label=x,
                y_axis_label=y_col,
                tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                tooltips=tooltips,
            )
            fig.xaxis.axis_label_text_font = "Inter, sans-serif"
            fig.yaxis.axis_label_text_font = "Inter, sans-serif"

            from bokeh.models import ColumnDataSource

            def _src(row: Any) -> ColumnDataSource:
                return ColumnDataSource(
                    {c: row[c].tolist() for c in hover_cols if c in row.columns}
                )

            if subplot_col and subplot_col in pdf.columns:
                # One sub-panel per unique subplot_col value via colour coding.
                unique_vals = sorted(pdf[subplot_col].dropna().unique())
                for vi, val in enumerate(unique_vals):
                    sub = pdf[pdf[subplot_col] == val]
                    entities = (
                        sub[key_col].unique() if key_col in sub.columns else [None]
                    )
                    for ei, entity in enumerate(entities):
                        row = sub[sub[key_col] == entity] if entity is not None else sub
                        color = _color_for(vi * len(entities) + ei)
                        label = f"{entity} ({val})" if entity is not None else str(val)
                        source = _src(row)
                        fig.line(
                            x=x,
                            y=y_col,
                            source=source,
                            color=color,
                            legend_label=label,
                            line_width=2,
                        )
                        fig.scatter(
                            x=x,
                            y=y_col,
                            source=source,
                            color=color,
                            size=5,
                        )
            elif key_col in df.columns:
                entities = df.get_column(key_col).unique().to_list()
                for i, entity in enumerate(entities):
                    entity_df = df.filter(pl.col(key_col) == entity)
                    source_data = {}
                    for c in hover_cols:
                        if c in entity_df.columns:
                            lst = entity_df.get_column(c).to_list()
                            if entity_df.schema[c].is_numeric():
                                lst = [float("nan") if v is None else v for v in lst]
                            source_data[c] = lst
                    source = ColumnDataSource(source_data)  # type: ignore[arg-type]
                    color = _color_for(i)
                    fig.line(
                        x=x,
                        y=y_col,
                        source=source,
                        color=color,
                        legend_label=str(entity),
                        line_width=2,
                    )
                    fig.scatter(
                        x=x,
                        y=y_col,
                        source=source,
                        color=color,
                        size=5,
                    )
            else:
                source = _src(pdf)
                fig.line(
                    x=x,
                    y=y_col,
                    source=source,
                    color=_color_for(0),
                    line_width=2,
                )
                fig.scatter(
                    x=x,
                    y=y_col,
                    source=source,
                    color=_color_for(0),
                    size=5,
                )

            if fig.legend:
                fig.legend.click_policy = "hide"
                fig.add_layout(fig.legend[0], "right")
            _apply_theme(fig, self.result, title)
            return fig

        group_cols = [x, subplot_col] if subplot_col else [x]
        agg_pdf = (
            df.group_by(group_cols)
            .agg([pl.col(c).mean() for c in y_cols])
            .sort(x)
            .to_pandas()
        )

        if aggregate:
            return _make_fig(agg_pdf, "Timeseries (Aggregated Mean)")

        entity_df = df.sort(x)
        unique_entities = entity_df[key_col].unique()
        if len(unique_entities) > top_n:
            entities = unique_entities.limit(top_n).to_list()
            entity_df = entity_df.join(
                pl.DataFrame({key_col: entities}), on=key_col, how="inner"
            )
        entity_pdf = entity_df.to_pandas()

        fig_entity = _make_fig(entity_pdf, f"Timeseries — {y_col} (Per Entity)")
        fig_agg = _make_fig(agg_pdf, f"Timeseries — {y_col} (Aggregated Mean)")

        tabs = Tabs(
            tabs=[
                TabPanel(child=fig_entity, title="Per Entity"),
                TabPanel(child=fig_agg, title="Aggregated Mean"),
            ]
        )
        return tabs

    def scatter(
        self, x: str | None = None, y: str | list[str] | None = None, top_n: int = 10
    ) -> Any:
        """Render a scatter plot using Bokeh.

        Args:
            x (str | None, optional): The column to plot on the x-axis.
            y (str | list[str] | None, optional): The column(s) to plot on the y-axis. If None, auto-selects all numeric.
            top_n (int, optional): Maximum number of entities to plot.

        Returns:
            Any: A Bokeh Figure or Tabs object.

        """
        import polars as pl
        import polars.selectors as cs
        from bokeh.models import Tabs, TabPanel
        from bokeh.plotting import figure

        if x is None:
            raise ValueError("x must be provided for scatter plot.")

        df = self.result.df()
        if x not in df.columns:
            raise ValueError(f"Column {x!r} not found in Result.")

        y_cols = list([y] if isinstance(y, str) else (y if y is not None else []))
        if not y_cols:
            y_cols = [c for c in df.select(cs.numeric()).columns if c != x]

        if not y_cols:
            raise ValueError("No numeric columns found for y-axis.")

        key_col = self.result.key_cols[0]
        hover_cols = [c for c in df.columns if c not in ("geometry", "geom")]

        unique_entities = df[key_col].unique()
        if len(unique_entities) > top_n:
            entities = unique_entities.limit(top_n).to_list()
            df = df.join(pl.DataFrame({key_col: entities}), on=key_col, how="inner")

        def _make_scatter(y_col: str) -> "BokehFigure":
            from bokeh.models import ColumnDataSource

            fig = figure(  # type: ignore[call-arg]
                width=800,
                height=450,
                x_axis_label=x,
                y_axis_label=y_col,
                tools="pan,wheel_zoom,box_zoom,reset,save,hover",
                tooltips=[(c, f"@{{{c}}}") for c in hover_cols],
            )

            entities = df.get_column(key_col).unique().to_list()
            for i, entity in enumerate(entities):
                row = df.filter(pl.col(key_col) == entity)

                # Replace None with NaN for numeric columns to prevent Bokeh JS rendering crashes
                source_data = {}
                for c in hover_cols:
                    if c in row.columns:
                        lst = row.get_column(c).to_list()
                        if row.schema[c].is_numeric():
                            lst = [float("nan") if v is None else v for v in lst]
                        source_data[c] = lst

                source = ColumnDataSource(source_data)  # type: ignore[arg-type]
                fig.scatter(
                    x=x,
                    y=y_col,
                    source=source,
                    color=_color_for(i),
                    legend_label=str(entity),
                    size=8,
                    alpha=0.7,
                )
            if fig.legend:
                fig.legend.click_policy = "hide"
                fig.add_layout(fig.legend[0], "right")
            _apply_theme(fig, self.result, "Scatter Distribution")
            return fig

        if len(y_cols) == 1:
            return _make_scatter(y_cols[0])

        tabs = Tabs(tabs=[TabPanel(child=_make_scatter(yc), title=yc) for yc in y_cols])
        return tabs

    def distribution(self, x: str | list[str] | None = None, top_n: int = 10) -> Any:
        """Render a distribution (histogram) plot using Bokeh.

        Args:
            x (str | list[str] | None, optional): The column(s) to plot the distribution for. If None, auto-selects all numeric.
            top_n (int, optional): Maximum number of entities to include.

        Returns:
            Any: A Bokeh Figure or Tabs object.

        """
        import numpy as np
        import pandas as pd
        import polars as pl
        import polars.selectors as cs
        from bokeh.models import Tabs, TabPanel
        from bokeh.plotting import figure

        df = self.result.df()

        x_cols = list([x] if isinstance(x, str) else (x if x is not None else []))
        if not x_cols:
            x_cols = df.select(cs.numeric()).columns

        if not x_cols:
            raise ValueError("No numeric columns found for distribution.")

        key_col = self.result.key_cols[0]

        unique_entities = df[key_col].unique()
        if len(unique_entities) > top_n:
            entities = unique_entities.limit(top_n).to_list()
            df = df.join(pl.DataFrame({key_col: entities}), on=key_col, how="inner")

        df_pd = df.to_pandas()

        def _make_hist(x_col: str) -> "BokehFigure":
            fig = figure(  # type: ignore[call-arg]
                width=800,
                height=400,
                x_axis_label=x_col,
                y_axis_label="Count",
                tools="pan,wheel_zoom,box_zoom,reset,save",
            )
            all_vals = df_pd[x_col].dropna().to_numpy()
            bins = np.histogram_bin_edges(all_vals, bins=30)
            for i, entity in enumerate(df_pd[key_col].unique()):
                subset = pd.Series(df_pd.loc[df_pd[key_col] == entity, x_col])
                vals = np.asarray(subset.dropna())
                hist, _ = np.histogram(vals, bins=bins)
                fig.quad(
                    top=hist,
                    bottom=0,
                    left=bins[:-1],
                    right=bins[1:],
                    color=_color_for(i),
                    alpha=0.6,
                    legend_label=str(entity),
                )
            if fig.legend:
                fig.legend.click_policy = "hide"
                fig.add_layout(fig.legend[0], "right")
            _apply_theme(fig, self.result, "Distribution")
            return fig

        if len(x_cols) == 1:
            return _make_hist(x_cols[0])

        tabs = Tabs(tabs=[TabPanel(child=_make_hist(xc), title=xc) for xc in x_cols])
        return tabs

    def correlation(self, columns: list[str] | None = None, top_n: int = 10) -> Any:
        """Render a correlation heatmap using Bokeh.

        Args:
            columns (list[str] | None, optional): Optional list of columns to correlate. Defaults to all numeric.
            top_n (int, optional): Maximum number of entities to include (currently unused).

        Returns:
            Any: A Bokeh Figure object.

        """
        import polars.selectors as cs
        from bokeh.models import BasicTicker, ColumnDataSource, LinearColorMapper
        from bokeh.models.annotations import ColorBar
        from bokeh.plotting import figure
        from bokeh.transform import transform

        df = self.result.df()
        if columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns {missing} not found in Result.")
            df = df.select(columns)
        else:
            df = df.select(cs.numeric())

        corr = df.to_pandas().corr()
        cols = list(corr.columns)

        xs: list[str] = []
        ys: list[str] = []
        vals: list[float] = []
        for row_col in cols:
            for col_col in cols:
                xs.append(col_col)
                ys.append(row_col)
                vals.append(float(corr.loc[row_col, col_col]))  # type: ignore[arg-type]

        source = ColumnDataSource({"x": xs, "y": ys, "value": vals})

        mapper = LinearColorMapper(palette="RdBu11", low=-1, high=1)

        from bokeh.models import FactorRange

        fig = figure(  # type: ignore[call-arg]
            width=600,
            height=550,
            x_range=FactorRange(factors=cols),
            y_range=FactorRange(factors=list(reversed(cols))),
            tools="hover,save",
            tooltips=[("pair", "@x vs @y"), ("r", "@value{0.00}")],
        )
        fig.rect(
            x="x",
            y="y",
            width=1,
            height=1,
            source=source,
            fill_color=transform("value", mapper),
            line_color=None,
        )
        color_bar = ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(desired_num_ticks=11),
            width=12,
        )
        fig.add_layout(color_bar, "right")
        fig.xaxis.major_label_orientation = 1.0
        _apply_theme(fig, self.result, "Correlation Matrix")
        return fig

    def heatmap(
        self,
        x: str | None = None,
        y: str | None = None,
        value: str | None = None,
        top_n: int = 10,
    ) -> Any:
        """Render a heatmap using Bokeh.

        Args:
            x (str | None, optional): The column for the x-axis.
            y (str | None, optional): The column for the y-axis.
            value (str | None, optional): The column for the colour values.
            top_n (int, optional): Maximum number of entities to include (currently unused).

        Returns:
            Any: A Bokeh Figure object.

        """
        from bokeh.models import (
            BasicTicker,
            ColumnDataSource,
            FactorRange,
            LinearColorMapper,
        )
        from bokeh.models.annotations import ColorBar
        from bokeh.plotting import figure
        from bokeh.transform import transform

        if x is None or y is None or value is None:
            raise ValueError("x, y, and value must be provided for heatmap.")

        df = self.result.df()
        for col in (x, y, value):
            if col not in df.columns:
                raise ValueError(f"Column {col!r} not found in Result.")

        pivot_df = df.pivot(values=value, index=y, on=x, aggregate_function="mean")
        pdf = pivot_df.to_pandas().set_index(y)

        x_vals = [str(v) for v in pdf.columns.tolist()]
        y_vals = [str(v) for v in pdf.index.tolist()]

        xs2: list[str] = []
        ys2: list[str] = []
        vals2: list[float] = []
        for yv in y_vals:
            for xv in x_vals:
                xs2.append(xv)
                ys2.append(yv)
                raw = pdf.loc[yv, xv]
                vals2.append(float(raw) if raw is not None else float("nan"))  # type: ignore[arg-type]

        source = ColumnDataSource({"x": xs2, "y": ys2, "value": vals2})

        low = min(v for v in vals2 if v == v)  # NaN-safe min
        high = max(v for v in vals2 if v == v)

        mapper = LinearColorMapper(palette="Viridis256", low=low, high=high)

        fig = figure(  # type: ignore[call-arg]
            width=900,
            height=500,
            x_range=FactorRange(factors=x_vals),
            y_range=FactorRange(factors=list(reversed(y_vals))),
            tools="hover,save",
            tooltips=[(x, "@x"), (y, "@y"), (value, "@value{0.00}")],
        )
        fig.rect(
            x="x",
            y="y",
            width=1,
            height=1,
            source=source,
            fill_color=transform("value", mapper),
            line_color=None,
        )
        color_bar = ColorBar(
            color_mapper=mapper,
            ticker=BasicTicker(desired_num_ticks=10),
            width=12,
        )
        fig.add_layout(color_bar, "right")
        fig.xaxis.major_label_orientation = 1.0
        _apply_theme(fig, self.result, f"Heatmap: {value}")
        return fig

    def matrix(self) -> Any:
        """Render a scatter matrix (pairs plot) using Bokeh.

        Returns:
            Any: A Bokeh ``gridplot`` object.

        """
        import polars.selectors as cs
        from bokeh.layouts import gridplot
        from bokeh.plotting import figure

        df = self.result.df()
        key_col = self.result.key_cols[0]
        df_num = df.select(cs.numeric() | cs.by_name(key_col)).to_pandas()
        num_cols = [c for c in df_num.columns if c != key_col]

        entities = df_num[key_col].unique()
        figs: list[list[Any]] = []

        for row_col in num_cols:
            row: list[Any] = []
            for col_col in num_cols:
                fig = figure(  # type: ignore[call-arg]
                    width=200,
                    height=200,
                    x_axis_label=col_col if row_col == num_cols[-1] else "",
                    y_axis_label=row_col if col_col == num_cols[0] else "",
                    tools="pan,wheel_zoom,reset",
                )
                for i, entity in enumerate(entities):
                    sub = df_num[df_num[key_col] == entity]
                    fig.scatter(
                        x=sub[col_col].tolist(),
                        y=sub[row_col].tolist(),
                        color=_color_for(i),
                        size=4,
                        alpha=0.6,
                        legend_label=str(entity)
                        if row_col == num_cols[0] and col_col == num_cols[0]
                        else "",
                    )
                row.append(fig)
            figs.append(row)

        grid = gridplot(figs)
        return grid
