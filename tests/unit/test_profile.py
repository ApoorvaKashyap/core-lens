"""Tests for ``core_lens.schema.profile.SchemaProfile``."""

import json
import pathlib
import pytest
from core_lens.schema.profile import SchemaProfile


def test_schema_profile_from_file(tmp_path: pathlib.Path) -> None:
    schema_data = {
        "key_cols": ["mws_id"],
        "geometry_col": "geometry",
        "geometry_type": "wkb",
        "annual_time_col": "year",
        "fortnightly_time_col": "date",
        "bbox_cols": None,
    }

    file_path = tmp_path / "schema.json"
    file_path.write_text(json.dumps(schema_data))

    profile = SchemaProfile.from_file(file_path)
    assert profile.key_cols == ["mws_id"]
    assert profile.geometry_col == "geometry"
    assert profile.geometry_type == "wkb"
    assert profile.annual_time_col == "year"


def test_schema_profile_from_file_invalid(tmp_path: pathlib.Path) -> None:
    file_path = tmp_path / "schema.json"
    file_path.write_text("invalid json")

    with pytest.raises(json.JSONDecodeError):
        SchemaProfile.from_file(file_path)
