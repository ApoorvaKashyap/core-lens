"""Tests for ``core_lens.base.result.Result``."""

from __future__ import annotations

import polars as pl
import pytest

from core_lens.base.result import Result


def _make_result(
    entity,
    resolution: str = "annual",
    has_geometry: bool = False,
    data: pl.DataFrame | None = None,
    metadata: dict | None = None,
) -> Result:
    """Build a ``Result`` for testing without touching Parquet files.

    Args:
        entity: A ``BaseEntity`` instance.
        resolution: One of ``"static"``, ``"annual"``, or ``"fortnightly"``.
        has_geometry: Whether the result carries a geometry column.
        data: Custom DataFrame. Defaults to a two-row NDVI frame.
        metadata: Optional metadata dict.

    Returns:
        A ``Result`` ready for assertion.
    """
    if data is None:
        data = pl.DataFrame({"mws_id": ["13_001", "13_002"], "ndvi_mean": [0.45, 0.50]})
    return Result(
        data=data,
        resolution=resolution,
        has_geometry=has_geometry,
        key_cols=["mws_id"],
        entity_name="minimalmws",
        entity=entity,
        metadata=metadata,
    )


class TestResultConstruction:
    def test_attributes_set_correctly(self, entity_cls):
        entity = entity_cls()
        df = pl.DataFrame({"mws_id": ["1"], "ndvi": [0.4]})
        result = Result(
            data=df,
            resolution="annual",
            has_geometry=False,
            key_cols=["mws_id"],
            entity_name="mws",
            entity=entity,
        )

        assert result.resolution == "annual"
        assert result.has_geometry is False
        assert result.key_cols == ["mws_id"]
        assert result.entity_name == "mws"
        assert result.entity is entity
        assert result.columns == ["mws_id", "ndvi"]

    def test_metadata_defaults_to_empty_dict(self, entity_cls):
        result = _make_result(entity_cls())

        assert result.metadata == {}

    def test_metadata_stored_when_provided(self, entity_cls):
        meta = {"method": "pearson", "p_value": 0.003}
        result = _make_result(entity_cls(), metadata=meta)

        assert result.metadata == meta


class TestResultDf:
    def test_df_returns_underlying_frame(self, entity_cls):
        entity = entity_cls()
        df = pl.DataFrame({"mws_id": ["1"], "val": [42]})
        result = _make_result(entity, data=df)

        assert result.df() is df


class TestResultLazy:
    def test_lazy_returns_lazyframe(self, entity_cls):
        result = _make_result(entity_cls())

        lazy = result.lazy()

        assert isinstance(lazy, pl.LazyFrame)

    def test_lazy_data_matches_df(self, entity_cls):
        result = _make_result(entity_cls())

        collected = result.lazy().collect()

        assert collected.equals(result.df())


class TestResultGdf:
    def test_gdf_raises_when_no_geometry(self, entity_cls):
        result = _make_result(entity_cls(), has_geometry=False)

        with pytest.raises(TypeError, match="no geometry"):
            result.gdf()


class TestResultWithGeometry:
    def test_with_geometry_raises_not_implemented(self, entity_cls):
        result = _make_result(entity_cls())

        with pytest.raises(NotImplementedError):
            result.with_geometry()


class TestResultDerive:
    def test_derive_appends_column(self, entity_cls):
        result = _make_result(entity_cls())
        derived = result.derive("double_ndvi", pl.col("ndvi_mean") * 2)

        assert "double_ndvi" in derived.columns
        assert "double_ndvi" not in result.columns

    def test_derive_values_correct(self, entity_cls):
        result = _make_result(entity_cls())
        derived = result.derive("double_ndvi", pl.col("ndvi_mean") * 2)

        expected = result.df()["ndvi_mean"] * 2
        assert derived.df()["double_ndvi"].to_list() == expected.to_list()

    def test_derive_original_unchanged(self, entity_cls):
        result = _make_result(entity_cls())
        result.derive("x", pl.col("ndvi_mean") + 1)

        assert "x" not in result.columns

    def test_derive_is_chainable(self, entity_cls):
        result = _make_result(entity_cls())
        chained = result.derive("a", pl.col("ndvi_mean") * 2).derive(
            "b", pl.col("a") + 1
        )

        assert "a" in chained.columns
        assert "b" in chained.columns

    def test_derive_preserves_metadata(self, entity_cls):
        meta = {"source": "test"}
        result = _make_result(entity_cls(), metadata=meta)
        derived = result.derive("x", pl.col("ndvi_mean") * 1)

        assert derived.metadata == meta


class TestResultAggregate:
    def test_aggregate_without_by_collapses_rows(self, entity_cls):
        result = _make_result(entity_cls())
        agg = result.aggregate(pl.mean("ndvi_mean"))

        assert len(agg.df()) == 1

    def test_aggregate_with_temporal_by_on_fortnightly(self, entity_cls):
        df = pl.DataFrame(
            {
                "mws_id": ["1", "1", "2"],
                "year": [2021, 2022, 2021],
                "ndvi": [0.4, 0.5, 0.3],
            }
        )
        result = _make_result(entity_cls(), resolution="fortnightly", data=df)
        agg = result.aggregate(pl.mean("ndvi"), by="year")

        assert "year" in agg.df().columns

    def test_aggregate_on_static_raises(self, entity_cls):
        result = _make_result(entity_cls(), resolution="static")

        with pytest.raises(ValueError, match="not supported on static"):
            result.aggregate(pl.mean("ndvi_mean"))

    def test_aggregate_temporal_by_on_annual_raises(self, entity_cls):
        result = _make_result(entity_cls(), resolution="annual")

        with pytest.raises(ValueError, match="resolution='fortnightly'"):
            result.aggregate(pl.mean("ndvi_mean"), by="year")

    def test_aggregate_unknown_by_raises(self, entity_cls):
        result = _make_result(entity_cls())

        with pytest.raises(ValueError, match="Unknown grouping"):
            result.aggregate(pl.mean("ndvi_mean"), by="quarter")

    @pytest.mark.parametrize(
        "by", ["year", "month", "year_month", "season", "season_year"]
    )
    def test_all_fortnightly_by_keys_accepted(self, entity_cls, by):
        df = pl.DataFrame(
            {
                "mws_id": ["1"],
                "year": [2022],
                "month": [1],
                "year_month": ["2022-01"],
                "season": ["rabi"],
                "season_year": ["rabi_2022"],
                "ndvi": [0.4],
            }
        )
        result = _make_result(entity_cls(), resolution="fortnightly", data=df)
        agg = result.aggregate(pl.mean("ndvi"), by=by)

        assert by in agg.df().columns


class TestResultReplace:
    def test_replace_carries_forward_unchanged_fields(self, entity_cls):
        entity = entity_cls()
        result = _make_result(entity, resolution="annual")
        replaced = result._replace(resolution="fortnightly")

        assert replaced.resolution == "fortnightly"
        assert replaced.entity is entity
        assert replaced.key_cols == result.key_cols
        assert replaced.entity_name == result.entity_name

    def test_replace_metadata_carried_forward(self, entity_cls):
        meta = {"method": "pearson"}
        result = _make_result(entity_cls(), metadata=meta)
        replaced = result._replace(has_geometry=True)

        assert replaced.metadata == meta
