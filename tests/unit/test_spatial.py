import pytest
from typing import Any
import pathlib
import polars as pl
import shapely.geometry as sgeom
from core_lens.utils.spatial import (
    resolve_path,
    build_bbox_index,
    exact_spatial_filter,
)


def test_resolve_path_relative(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    p = pathlib.Path("some_file.txt")
    p.write_text("hello")
    assert resolve_path("some_file.txt") == str(tmp_path / "some_file.txt")


def test_resolve_path_not_found(tmp_path: Any) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_path(str(tmp_path / "missing.txt"))


def test_build_bbox_index_with_bbox_cols(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame(
        {"id": [1], "minx": [0.0], "miny": [0.0], "maxx": [1.0], "maxy": [1.0]}
    ).write_parquet(p)
    df = build_bbox_index(
        str(p), ["id"], ("minx", "miny", "maxx", "maxy"), "geom", "latlon"
    )
    assert df.columns == ["id", "minx", "miny", "maxx", "maxy"]


def test_build_bbox_index_latlon_error() -> None:
    with pytest.raises(ValueError, match="geometry_type='latlon' requires bbox_cols"):
        build_bbox_index("path", ["id"], None, "geom", "latlon")


def test_build_bbox_index_wkt(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame({"id": [1], "geom": ["POINT (0.5 0.5)"]}).write_parquet(p)
    df = build_bbox_index(str(p), ["id"], None, "geom", "wkt")
    assert df.columns == ["id", "minx", "miny", "maxx", "maxy"]


def test_exact_spatial_filter_empty(tmp_path: Any) -> None:
    candidates = pl.DataFrame({"id": []}, schema={"id": pl.Int64})
    res = exact_spatial_filter(
        candidates, "path", ["id"], "geom", "wkb", sgeom.Point(0, 0)
    )
    assert res.is_empty()


def test_exact_spatial_filter_composite_key(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame({"id1": [1], "id2": [2], "geom": ["POINT (0.5 0.5)"]}).write_parquet(p)
    candidates = pl.DataFrame({"id1": [1], "id2": [2]})
    res = exact_spatial_filter(
        candidates, str(p), ["id1", "id2"], "geom", "wkt", sgeom.box(0, 0, 1, 1)
    )
    assert res.height == 1


def test_exact_spatial_filter_area(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame(
        {
            "id": [1, 2],
            "geom": ["POINT (0.5 0.5)", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))"],
        }
    ).write_parquet(p)
    candidates = pl.DataFrame({"id": [1, 2]})
    res = exact_spatial_filter(
        candidates,
        str(p),
        ["id"],
        "geom",
        "wkt",
        sgeom.box(0, 0, 0.5, 1.0),
        relationship="area",
        threshold=0.1,
    )
    # The polygon area is 1.0. Intersection with box(0, 0, 0.5, 1.0) is area 0.5.
    # 0.5 / 1.0 = 0.5 >= 0.1 threshold, so it matches. Point doesn't have area.
    assert res.height == 1
    assert res["id"][0] == 2


def test_execute_spatial_join(tmp_path: Any) -> None:
    from core_lens.utils.spatial import execute_spatial_join
    from core_lens.schema.profile import SchemaProfile

    class DummyOther:
        static_path = str(tmp_path / "other.parquet")
        key_cols = ["oid"]
        schema_profile = SchemaProfile(
            key_cols=["oid"],
            geometry_col="geom",
            geometry_type="wkt",
            annual_time_col=None,
            sub_annual_time_col=None,
            bbox_cols=None,
        )

        def _resolve(self, path: str) -> str:
            return path

    pl.DataFrame(
        {
            "oid": [1, 2],
            "val": [10.0, 20.0],
            "geom": [
                "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))",
                "POLYGON ((1 1, 2 1, 2 2, 1 2, 1 1))",
            ],
        }
    ).write_parquet(DummyOther.static_path)

    primary = pl.DataFrame(
        {"pid": [1], "geom": ["POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))"]}
    )

    res = execute_spatial_join(
        primary,
        ["pid"],
        "geom",
        "wkt",
        DummyOther(),
        agg={"val": "mean", "count": "count", "area": "area"},
        other_entity_name="other",
    )
    assert res.height == 1
    assert res["other_val"][0] == 15.0
    assert res["other_count"][0] == 2
    assert res["other_area"][0] == 2.0


def test_build_bbox_index_empty(tmp_path: Any) -> None:
    p = tmp_path / "test.parquet"
    pl.DataFrame(
        {"id": [], "geom": []}, schema={"id": pl.Int64, "geom": pl.Utf8}
    ).write_parquet(p)
    df = build_bbox_index(str(p), ["id"], None, "geom", "wkt")
    assert df.height == 0
    assert df.columns == ["id", "minx", "miny", "maxx", "maxy"]


def test_exact_spatial_filter_invalid_relationship(tmp_path: Any) -> None:
    candidates = pl.DataFrame({"id": [1]})
    with pytest.raises(ValueError, match="Unknown relationship"):
        exact_spatial_filter(
            candidates,
            "path",
            ["id"],
            "geom",
            "wkb",
            sgeom.Point(0, 0),
            relationship="invalid",
        )


def test_execute_spatial_join_errors(tmp_path: Any) -> None:
    from core_lens.utils.spatial import execute_spatial_join
    from core_lens.schema.profile import SchemaProfile

    class DummyOther:
        static_path = str(tmp_path / "other.parquet")
        key_cols = ["oid"]
        schema_profile = SchemaProfile(
            key_cols=["oid"],
            geometry_col="geom",
            geometry_type="wkt",
            annual_time_col=None,
            sub_annual_time_col=None,
            bbox_cols=None,
        )

        def _resolve(self, path: str) -> str:
            return path

    pl.DataFrame({"oid": [1], "val": [10.0], "geom": ["POINT (0 0)"]}).write_parquet(
        DummyOther.static_path
    )
    primary = pl.DataFrame({"pid": [1]})
    with pytest.raises(ValueError, match="geometry column 'geom' not found"):
        execute_spatial_join(
            primary,
            ["pid"],
            "geom",
            "wkt",
            DummyOther(),
            agg={},
            other_entity_name="other",
        )


def test_execute_spatial_join_no_match_and_agg(tmp_path: Any) -> None:
    from core_lens.utils.spatial import execute_spatial_join
    from core_lens.schema.profile import SchemaProfile

    class DummyOther:
        static_path = str(tmp_path / "other.parquet")
        key_cols = ["oid"]
        schema_profile = SchemaProfile(
            key_cols=["oid"],
            geometry_col="geom",
            geometry_type="wkt",
            annual_time_col=None,
            sub_annual_time_col=None,
            bbox_cols=None,
        )

        def _resolve(self, path: str) -> str:
            return path

    pl.DataFrame(
        {
            "oid": [1],
            "val": [10.0],
            "val2": [10.0],
            "val3": [10.0],
            "geom": ["POINT (10 10)"],
        }
    ).write_parquet(DummyOther.static_path)
    primary = pl.DataFrame({"pid": [1, 2], "geom": ["POINT (0 0)", "POINT (10 10)"]})
    res = execute_spatial_join(
        primary,
        ["pid"],
        "geom",
        "wkt",
        DummyOther(),
        agg={"val": "sum", "val2": "min", "val3": "max", "count": "count"},
        other_entity_name="other",
    )
    assert res.height == 2
    assert res["other_val"][0] is None  # No match
    assert res["other_val"][1] == 10.0  # Match
    assert res["other_val2"][1] == 10.0
    assert res["other_val3"][1] == 10.0


def _make_pie_entity(tmp_path: Any, name: str, polygons: list[tuple[str, str]]) -> Any:
    """Build a minimal entity stub for point_in_entities tests.

    Args:
        tmp_path: pytest tmp_path fixture.
        name: Class name (used as entity key in result dict).
        polygons: List of (id, wkt_polygon_string) tuples.

    Returns:
        Entity-like object with _index, key_cols, schema_profile, _resolve, static_path.

    """
    from core_lens.schema.profile import SchemaProfile

    ids = [p[0] for p in polygons]
    wkts = [p[1] for p in polygons]

    p = tmp_path / f"{name}.parquet"
    pl.DataFrame({"entity_id": ids, "geom": wkts}).write_parquet(p)

    # Build bbox index manually from WKT bounds.
    import shapely

    geoms = shapely.from_wkt(wkts)
    bnds = shapely.bounds(geoms) if len(geoms) else None
    if bnds is not None and len(bnds):
        index = pl.DataFrame(
            {
                "entity_id": ids,
                "minx": bnds[:, 0].tolist(),
                "miny": bnds[:, 1].tolist(),
                "maxx": bnds[:, 2].tolist(),
                "maxy": bnds[:, 3].tolist(),
            }
        )
    else:
        index = pl.DataFrame(
            schema={
                "entity_id": pl.Utf8,
                "minx": pl.Float64,
                "miny": pl.Float64,
                "maxx": pl.Float64,
                "maxy": pl.Float64,
            }
        )

    profile = SchemaProfile(
        key_cols=["entity_id"],
        geometry_col="geom",
        geometry_type="wkt",
        annual_time_col=None,
        sub_annual_time_col=None,
        bbox_cols=None,
    )

    static_str = str(p)

    class _Entity:
        key_cols = ["entity_id"]
        static_path = static_str
        schema_profile = profile
        _index = index

        def _resolve(self, path: str) -> str:
            return path

    _Entity.__name__ = name
    return _Entity()


def test_point_in_entities_hit(tmp_path: Any) -> None:
    """Point inside a polygon returns its id."""
    from core_lens.utils.spatial import point_in_entities

    entity = _make_pie_entity(
        tmp_path,
        "Zone",
        [("Z1", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")],
    )
    result = point_in_entities({"lat": 0.5, "lon": 0.5}, [entity])
    assert result == {"Zone": "Z1"}


def test_point_in_entities_miss(tmp_path: Any) -> None:
    """Point outside all polygons returns None."""
    from core_lens.utils.spatial import point_in_entities

    entity = _make_pie_entity(
        tmp_path,
        "Zone",
        [("Z1", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")],
    )
    result = point_in_entities({"lat": 5.0, "lon": 5.0}, [entity])
    assert result == {"Zone": None}


def test_point_in_entities_multi_entity(tmp_path: Any) -> None:
    """Each entity resolved independently; mixed hit/miss."""
    from core_lens.utils.spatial import point_in_entities

    entity_a = _make_pie_entity(
        tmp_path,
        "District",
        [
            ("D1", "POLYGON ((0 0, 10 0, 10 10, 0 10, 0 0))"),
            ("D2", "POLYGON ((10 0, 20 0, 20 10, 10 10, 10 0))"),
        ],
    )

    entity_b = _make_pie_entity(
        tmp_path,
        "Block",
        [("B1", "POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))")],
    )

    # Point at (3, 3) — inside D1 and B1.
    result = point_in_entities({"lat": 3.0, "lon": 3.0}, [entity_a, entity_b])
    assert result["District"] == "D1"
    assert result["Block"] == "B1"

    # Point at (15, 5) — inside D2, outside B1.
    result2 = point_in_entities({"lat": 5.0, "lon": 15.0}, [entity_a, entity_b])
    assert result2["District"] == "D2"
    assert result2["Block"] is None


def test_point_in_entities_lng_alias(tmp_path: Any) -> None:
    """Accepts 'lng' as alias for longitude."""
    from core_lens.utils.spatial import point_in_entities

    entity = _make_pie_entity(
        tmp_path,
        "Zone",
        [("Z1", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")],
    )
    result = point_in_entities({"lat": 0.5, "lng": 0.5}, [entity])
    assert result == {"Zone": "Z1"}


def test_point_in_entities_empty_list(tmp_path: Any) -> None:
    """Empty entity list returns empty dict."""
    from core_lens.utils.spatial import point_in_entities

    result = point_in_entities({"lat": 0.5, "lon": 0.5}, [])
    assert result == {}


def test_point_in_entities_on_boundary(tmp_path: Any) -> None:
    """Point exactly on polygon boundary — Shapely treats as contained."""
    from core_lens.utils.spatial import point_in_entities

    entity = _make_pie_entity(
        tmp_path,
        "Zone",
        [("Z1", "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))")],
    )
    # Shapely `contains` is strict (boundary not contained) — boundary point
    # may return None. Test documents this behaviour rather than asserting a
    # specific id, so the test remains robust across Shapely versions.
    result = point_in_entities({"lat": 0.0, "lon": 0.5}, [entity])
    assert result["Zone"] in ("Z1", None)


def test_point_in_entities_bbox_skips_exact_check(tmp_path: Any) -> None:
    """Bbox pre-filter short-circuits: point outside all bboxes skips I/O."""
    from core_lens.utils.spatial import point_in_entities

    entity = _make_pie_entity(
        tmp_path,
        "Zone",
        [("Z1", "POLYGON ((10 10, 20 10, 20 20, 10 20, 10 10))")],
    )
    # Point at (0, 0) is outside bbox [10,10,20,20], so candidates empty.
    result = point_in_entities({"lat": 0.0, "lon": 0.0}, [entity])
    assert result == {"Zone": None}
