"""Shared pytest fixtures for core_lens unit tests."""

from __future__ import annotations

import datetime
import pathlib

import polars as pl
import pytest
import shapely.geometry as sgeom
import shapely.wkb as swkb

from core_lens.aoi import AoI, _REGISTRY
from core_lens.base.entity import BaseEntity
from core_lens.base.result import Result
from core_lens.base.view import View
from core_lens.schema.profile import SchemaProfile


def _make_static_parquet(path: pathlib.Path, *, extra_cols: bool = False) -> None:
    """Write a minimal static GeoParquet with WKB geometry to *path*.

    Args:
        path: Destination file path.
        extra_cols: When True, appends an ``area_ha`` float column.
    """
    wkb_bytes = swkb.dumps(sgeom.box(73.0, 15.0, 74.0, 16.0))
    data: dict = {
        "mws_id": ["13_001", "13_002"],
        "district": ["TestDistrict", "TestDistrict"],
        "geometry": [wkb_bytes, wkb_bytes],
    }
    if extra_cols:
        data["area_ha"] = [100.0, 200.0]
    pl.DataFrame(data).write_parquet(path)


def _make_annual_parquet(path: pathlib.Path) -> None:
    """Write a minimal annual Parquet to *path*.

    Args:
        path: Destination file path.
    """
    pl.DataFrame(
        {
            "mws_id": ["13_001", "13_001", "13_002"],
            "year": [2021, 2022, 2021],
            "ndvi_mean": [0.45, 0.50, 0.40],
        }
    ).write_parquet(path)


def _make_fortnightly_parquet(path: pathlib.Path) -> None:
    """Write a minimal fortnightly Parquet to *path*.

    Args:
        path: Destination file path.
    """
    pl.DataFrame(
        {
            "mws_id": ["13_001", "13_001"],
            "fortnightly_date": [
                datetime.date(2022, 1, 1),
                datetime.date(2022, 1, 15),
            ],
            "ndvi": [0.42, 0.47],
        }
    ).write_parquet(path)


def _make_entity_cls(
    static: pathlib.Path,
    annual: pathlib.Path | None = None,
    fortnightly: pathlib.Path | None = None,
) -> type[BaseEntity]:
    """Dynamically create a concrete ``BaseEntity`` subclass for testing.

    The returned class is fully concrete — it implements all abstract
    methods and is backed by the supplied Parquet files.

    Args:
        static: Path to the static GeoParquet file.
        annual: Path to the annual Parquet file, or ``None``.
        fortnightly: Path to the fortnightly Parquet file, or ``None``.

    Returns:
        A ``BaseEntity`` subclass ready for ``AoI.register``.
    """
    profile = SchemaProfile(
        key_cols=["mws_id"],
        geometry_col="geometry",
        geometry_type="wkb",
        annual_time_col="year" if annual else None,
        fortnightly_time_col="fortnightly_date" if fortnightly else None,
        bbox_cols=None,
        extra_static_cols=["district"],
    )

    # Capture paths as locals so the inner class closure is self-contained.
    _static = str(static)
    _annual = str(annual) if annual else None
    _fortnightly = str(fortnightly) if fortnightly else None
    _profile = profile

    class MinimalMWSEntity(BaseEntity):
        @property
        def key_cols(self) -> list[str]:
            return ["mws_id"]

        @property
        def geometry_col(self) -> str:
            return "geometry"

        @property
        def static_path(self) -> str:
            return _static

        @property
        def annual_path(self) -> str | None:
            return _annual

        @property
        def fortnightly_path(self) -> str | None:
            return _fortnightly

        @property
        def schema_profile(self) -> SchemaProfile:
            return _profile

        def where(self, **kwargs) -> View:  # pragma: no cover
            raise NotImplementedError

        def spatial_filter(self, **kwargs) -> View:
            # Return an empty-keyed View so AoI.__init__ can scope entities
            # without performing any real spatial computation.
            keys = pl.DataFrame({"mws_id": []}, schema={"mws_id": pl.String})
            return View(keys=keys, entity=self, entity_name="minimalmws")

        def spatial_join(self, other, agg) -> View:  # pragma: no cover
            raise NotImplementedError

    return MinimalMWSEntity


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the global AoI entity registry before and after every test."""
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()


@pytest.fixture()
def static_parquet(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the path to a freshly written static GeoParquet file."""
    p = tmp_path / "static.parquet"
    _make_static_parquet(p)
    return p


@pytest.fixture()
def static_parquet_extra(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the path to a static GeoParquet that includes extra columns."""
    p = tmp_path / "static_extra.parquet"
    _make_static_parquet(p, extra_cols=True)
    return p


@pytest.fixture()
def annual_parquet(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the path to a freshly written annual Parquet file."""
    p = tmp_path / "annual.parquet"
    _make_annual_parquet(p)
    return p


@pytest.fixture()
def fortnightly_parquet(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return the path to a freshly written fortnightly Parquet file."""
    p = tmp_path / "fortnightly.parquet"
    _make_fortnightly_parquet(p)
    return p


@pytest.fixture()
def entity_cls(static_parquet: pathlib.Path) -> type[BaseEntity]:
    """Return a concrete entity class backed by only a static file."""
    return _make_entity_cls(static_parquet)


@pytest.fixture()
def entity_cls_full(
    static_parquet: pathlib.Path,
    annual_parquet: pathlib.Path,
    fortnightly_parquet: pathlib.Path,
) -> type[BaseEntity]:
    """Return a concrete entity class backed by all three Parquet files."""
    return _make_entity_cls(static_parquet, annual_parquet, fortnightly_parquet)


@pytest.fixture()
def registered_entity_cls(entity_cls: type[BaseEntity]) -> type[BaseEntity]:
    """Register *entity_cls* with ``AoI`` and return it."""
    AoI.register(entity_cls)
    return entity_cls


@pytest.fixture()
def bbox_aoi(registered_entity_cls) -> AoI:
    """Return an ``AoI`` constructed from a bounding box."""
    return AoI(".", bbox=(73.0, 15.0, 74.0, 16.0))


@pytest.fixture()
def geom_aoi(registered_entity_cls) -> AoI:
    """Return an ``AoI`` constructed from a Shapely geometry."""
    return AoI(".", geometry=sgeom.box(73.0, 15.0, 74.0, 16.0))


@pytest.fixture()
def minimal_schema() -> SchemaProfile:
    """Return a ``SchemaProfile`` with no optional temporal or bbox fields."""
    return SchemaProfile(
        key_cols=["mws_id"],
        geometry_col="geometry",
        geometry_type="wkb",
        annual_time_col=None,
        fortnightly_time_col=None,
        bbox_cols=None,
    )


@pytest.fixture()
def sample_result(entity_cls):
    """Return a ``Result`` with an in-memory DataFrame at annual resolution."""

    df = pl.DataFrame({"mws_id": ["13_001", "13_002"], "ndvi_mean": [0.45, 0.50]})
    return Result(
        data=df,
        resolution="annual",
        has_geometry=False,
        key_cols=["mws_id"],
        entity_name="minimalmws",
        entity=entity_cls(),
    )
