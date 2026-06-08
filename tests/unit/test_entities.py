"""Tests for built-in entities."""

from core_lens.entities.tehsil import TehsilEntity
from core_lens.entities.mws import MWSEntity
from core_lens.schema.profile import SchemaProfile


def test_tehsil_entity() -> None:
    entity = TehsilEntity()
    assert entity.key_cols == ["id"]
    assert entity.geometry_col == "geom"
    assert entity.static_path == "data/tehsil/static/tehsil.parquet"
    assert entity.annual_path is None
    assert entity.fortnightly_path is None

    # Check schema profile override
    profile = entity.schema_profile
    assert isinstance(profile, SchemaProfile)
    assert profile.key_cols == ["id"]
    assert profile.geometry_col == "geom"
    assert "TEHSIL" in profile.extra_static_cols


def test_mws_entity() -> None:
    entity = MWSEntity()
    assert entity.key_cols == ["mws_id"]
    assert entity.geometry_col == "geometry"
    assert entity.static_path == "data/mws/static"
    assert entity.annual_path == "data/mws/annual"
    assert entity.fortnightly_path == "data/mws/fortnightly"
