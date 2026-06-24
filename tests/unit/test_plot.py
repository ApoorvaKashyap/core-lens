"""Unit tests for the PlotNamespace."""

import lonboard
import polars as pl
import pytest
import shapely.geometry as sgeom
import shapely.wkb as swkb
from typing import Any
from core_lens.base.entity import BaseEntity
from core_lens.base.result import Result
from core_lens.schema.profile import Resolution, SchemaProfile
from core_lens.base.namespaces.plot import SubplotOn


class DummyEntity(BaseEntity):
    @property
    def key_cols(self) -> list[str]:
        return ["dummy_id"]

    @property
    def geometry_col(self) -> str:
        return "geometry"

    @property
    def static_path(self) -> str:
        return "dummy_static.geoparquet"

    @property
    def schema_profile(self) -> SchemaProfile:
        return SchemaProfile(
            key_cols=["dummy_id"],
            geometry_col="geometry",
            geometry_type="wkb",
            annual_time_col=None,
            fortnightly_time_col=None,
            bbox_cols=None,
        )

    def where(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def spatial_filter(
        self,
        geometry: Any | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        relationship: str = "centroid",
        threshold: float = 0.5,
    ) -> Any:
        raise NotImplementedError

    def spatial_join(self, other: Any, agg: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def dummy_result() -> Result:
    geom1 = sgeom.box(0, 0, 1, 1)
    geom2 = sgeom.box(1, 1, 2, 2)

    df = pl.DataFrame(
        {
            "dummy_id": ["A", "B"],
            "value": [10.5, 20.1],
            "geometry": [swkb.dumps(geom1), swkb.dumps(geom2)],
        }
    )

    return Result(
        data=df,
        resolution=Resolution.STATIC,
        has_geometry=True,
        key_cols=["dummy_id"],
        entity_name="dummy",
        entity=DummyEntity(),
    )


def test_choropleth_basic(dummy_result: Result) -> None:
    """Test choropleth creates a lonboard Map correctly."""
    plot_map = dummy_result.plot.choropleth("value")

    assert isinstance(plot_map, lonboard.Map)
    assert len(plot_map.layers) == 1
    layer = plot_map.layers[0]
    assert isinstance(layer, lonboard.PolygonLayer)


def test_choropleth_missing_column(dummy_result: Result) -> None:
    """Test choropleth raises error if column is missing."""
    with pytest.raises(ValueError, match="Column 'missing' not found in Result"):
        dummy_result.plot.choropleth("missing")


def test_timeseries_basic(dummy_result: Result) -> None:
    """Test timeseries creates a plotly figure."""
    # Add a time column to dummy result for testing
    df = dummy_result.data.with_columns(pl.Series("year", [2020, 2021]))
    res = dummy_result._replace(data=df)

    fig = res.plot.timeseries(x="year", y="value")
    assert fig is not None
    from bokeh.models import Tabs

    assert isinstance(fig, Tabs)
    entity_fig = fig.tabs[0].child  # type: ignore[union-attr]
    assert entity_fig.xaxis.axis_label == "year"  # type: ignore[union-attr]
    assert entity_fig.yaxis.axis_label == "value"  # type: ignore[union-attr]


def test_timeseries_aggregate(dummy_result: Result) -> None:
    """Test timeseries aggregate creates a plotly figure."""
    df = dummy_result.data.with_columns(pl.Series("year", [2020, 2021]))
    res = dummy_result._replace(data=df)

    fig = res.plot.timeseries(x="year", y="value", aggregate=True)
    assert fig is not None
    assert len(fig.renderers) > 0


def test_timeseries_missing_xy(dummy_result: Result) -> None:
    """Test timeseries raises error if x missing."""
    with pytest.raises(ValueError, match="x must be provided"):
        dummy_result.plot.timeseries()


def test_timeseries_multiple_y(dummy_result: Result) -> None:
    """Test timeseries with multiple y columns creates a dropdown menu."""
    df = dummy_result.data.with_columns(
        [pl.Series("year", [2020, 2021]), pl.Series("value2", [5.0, 10.0])]
    )
    res = dummy_result._replace(data=df)
    fig = res.plot.timeseries(x="year", y=["value", "value2"])
    assert fig is not None
    from bokeh.models import Tabs

    assert isinstance(fig, Tabs)


def test_scatter_basic(dummy_result: Result) -> None:
    """Test scatter creates a plotly figure."""
    df = dummy_result.data.with_columns(pl.Series("other_val", [1.1, 2.2]))
    res = dummy_result._replace(data=df)

    fig = res.plot.scatter(x="value", y="other_val")
    assert fig is not None
    assert fig.xaxis.axis_label == "value"
    assert fig.yaxis.axis_label == "other_val"


def test_scatter_missing_xy(dummy_result: Result) -> None:
    """Test scatter raises error if x missing."""
    with pytest.raises(ValueError, match="x must be provided"):
        dummy_result.plot.scatter()


def test_scatter_multiple_y(dummy_result: Result) -> None:
    """Test scatter with multiple y columns creates a dropdown menu."""
    df = dummy_result.data.with_columns(
        [pl.Series("other_val", [1.1, 2.2]), pl.Series("third_val", [3.3, 4.4])]
    )
    res = dummy_result._replace(data=df)
    fig = res.plot.scatter(x="value", y=["other_val", "third_val"])
    assert fig is not None
    assert hasattr(fig, "tabs")
    assert len(fig.tabs) > 0


def test_distribution_basic(dummy_result: Result) -> None:
    """Test distribution creates a plotly figure."""
    fig = dummy_result.plot.distribution(x="value")
    assert fig is not None
    assert fig.xaxis.axis_label == "value"


def test_distribution_missing_x(dummy_result: Result) -> None:
    """Test distribution does not raise error if x missing (it auto-selects)."""
    fig = dummy_result.plot.distribution()
    assert fig is not None


def test_distribution_multiple_x(dummy_result: Result) -> None:
    """Test distribution with multiple x columns creates a dropdown menu."""
    df = dummy_result.data.with_columns(pl.Series("other_val", [1.1, 2.2]))
    res = dummy_result._replace(data=df)
    fig = res.plot.distribution(x=["value", "other_val"])
    assert fig is not None
    assert hasattr(fig, "tabs")
    assert len(fig.tabs) > 0


def test_correlation_basic(dummy_result: Result) -> None:
    """Test correlation creates a plotly figure."""
    df = dummy_result.data.with_columns(pl.Series("other_val", [1.1, 2.2]))
    res = dummy_result._replace(data=df)

    fig = res.plot.correlation()
    assert fig is not None


def test_correlation_multiple_columns(dummy_result: Result) -> None:
    """Test correlation creates a figure with multiple specified columns."""
    df = dummy_result.data.with_columns(
        [pl.Series("other_val", [1.1, 2.2]), pl.Series("third_val", [3.3, 4.4])]
    )
    res = dummy_result._replace(data=df)
    fig = res.plot.correlation(["value", "other_val", "third_val"])
    assert fig is not None


def test_correlation_missing_columns(dummy_result: Result) -> None:
    """Test correlation raises error if columns missing."""
    with pytest.raises(ValueError, match="not found in Result"):
        dummy_result.plot.correlation(columns=["missing"])


def test_heatmap_basic(dummy_result: Result) -> None:
    """Test heatmap creates a plotly figure."""
    df = dummy_result.data.with_columns(pl.Series("other_val", [1.1, 2.2]))
    df = df.with_columns(pl.Series("cat", ["A", "B"]))
    res = dummy_result._replace(data=df)

    fig = res.plot.heatmap(x="dummy_id", y="cat", value="other_val")
    assert fig is not None


def test_heatmap_missing_args(dummy_result: Result) -> None:
    """Test heatmap raises error if args missing."""
    with pytest.raises(ValueError, match="must be provided"):
        dummy_result.plot.heatmap(x="dummy_id", y="cat")


def test_matrix_basic(dummy_result: Result) -> None:
    """Test matrix creates a plotly figure."""
    df = dummy_result.data.with_columns(pl.Series("other_val", [1.1, 2.2]))
    res = dummy_result._replace(data=df)

    fig = res.plot.matrix()
    assert fig is not None


def test_choropleth_geometry_collection(dummy_result: Result) -> None:
    """Test choropleth handles GeometryCollection."""
    import shapely.geometry as sgeom

    poly = sgeom.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    gc = sgeom.GeometryCollection([poly, sgeom.Point(0.5, 0.5)])
    df = dummy_result.data.with_columns(pl.Series("geometry", [gc.wkb, gc.wkb]))
    res = dummy_result._replace(data=df, has_geometry=True)
    m = res.plot.choropleth("value")
    assert m is not None


def test_choropleth_subplot_on(dummy_result: Result) -> None:
    """Test choropleth handles subplot_on argument."""
    df = dummy_result.data.with_columns(pl.Series("year", [2020, 2021]))
    res = dummy_result._replace(data=df)
    m = res.plot.choropleth("value", subplot_on=SubplotOn.YEAR)
    assert m is not None


def test_choropleth_subplot_on_missing(dummy_result: Result) -> None:
    with pytest.raises(ValueError, match="not found in Result"):
        dummy_result.plot.choropleth("value", subplot_on=SubplotOn.YEAR)


def test_choropleth_subplot_invalid(dummy_result: Result) -> None:
    with pytest.raises(ValueError, match="must be a SubplotOn enum"):
        dummy_result.plot.choropleth("value", subplot_on="year")  # type: ignore


def test_timeseries_subplot_on(dummy_result: Result) -> None:
    df = dummy_result.data.with_columns(
        [pl.Series("year", [2020, 2021]), pl.Series("month", [1, 2])]
    )
    res = dummy_result._replace(data=df)
    fig = res.plot.timeseries(x="year", y="value", subplot_on=SubplotOn.MONTH)
    assert fig is not None


def test_timeseries_subplot_on_str(dummy_result: Result) -> None:
    df = dummy_result.data.with_columns(
        [pl.Series("year", [2020, 2021]), pl.Series("month", [1, 2])]
    )
    res = dummy_result._replace(data=df)
    fig = res.plot.timeseries(x="year", y="value", subplot_on="month")
    assert fig is not None


def test_timeseries_subplot_on_missing(dummy_result: Result) -> None:
    df = dummy_result.data.with_columns([pl.Series("year", [2020, 2021])])
    res = dummy_result._replace(data=df)
    with pytest.raises(ValueError, match="not found in Result"):
        res.plot.timeseries(x="year", y="value", subplot_on="missing")
