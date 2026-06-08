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
    assert fig.layout.xaxis.title.text == "year"
    assert fig.layout.yaxis.title.text == "value"


def test_timeseries_aggregate(dummy_result: Result) -> None:
    """Test timeseries aggregate creates a plotly figure."""
    df = dummy_result.data.with_columns(pl.Series("year", [2020, 2020]))
    res = dummy_result._replace(data=df)

    fig = res.plot.timeseries(x="year", y="value", aggregate=True)
    assert fig is not None
    # one point in aggregate
    assert len(fig.data[0].x) == 1


def test_timeseries_missing_xy(dummy_result: Result) -> None:
    """Test timeseries raises error if x or y missing."""
    with pytest.raises(ValueError, match="Both x and y must be provided"):
        dummy_result.plot.timeseries(x="year")


def test_scatter_basic(dummy_result: Result) -> None:
    """Test scatter creates a plotly figure."""
    df = dummy_result.data.with_columns(pl.Series("other_val", [1.1, 2.2]))
    res = dummy_result._replace(data=df)

    fig = res.plot.scatter(x="value", y="other_val")
    assert fig is not None
    assert fig.layout.xaxis.title.text == "value"
    assert fig.layout.yaxis.title.text == "other_val"


def test_scatter_missing_xy(dummy_result: Result) -> None:
    """Test scatter raises error if x or y missing."""
    with pytest.raises(ValueError, match="Both x and y must be provided"):
        dummy_result.plot.scatter(x="value")


def test_distribution_basic(dummy_result: Result) -> None:
    """Test distribution creates a plotly figure."""
    fig = dummy_result.plot.distribution(x="value")
    assert fig is not None
    assert fig.layout.xaxis.title.text == "value"


def test_distribution_missing_x(dummy_result: Result) -> None:
    """Test distribution raises error if x missing."""
    with pytest.raises(ValueError, match="x must be provided"):
        dummy_result.plot.distribution()


def test_correlation_basic(dummy_result: Result) -> None:
    """Test correlation creates a plotly figure."""
    df = dummy_result.data.with_columns(pl.Series("other_val", [1.1, 2.2]))
    res = dummy_result._replace(data=df)

    fig = res.plot.correlation()
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
