"""
demo_mws.py — core_lens MWSEntity capability demonstration
=========================================================
Walks through every meaningful operation available on a spatiotemporal
entity: registration, AoI construction, filtering, materialisation,
geometry, derive, stats, plot, temporal queries, and export.

Prerequisites
-------------
    pip install core-lens geopandas shapely polars scipy plotly lonboard

Data layout expected
--------------------
    data/
    └── mws/
        ├── static/
        │   └── mws.parquet   ← GeoParquet with WKB geometry in "geom"
        ├── annual/
        │   └── ...           ← Parquet partitioned by year
        └── fortnightly/
            └── ...           ← Parquet partitioned by basin and year

Columns available:
- Static: mws_id, tehsil, district, state, area_in_ha, ba_name, geometry, etc.
- Annual: mws_id, year, dw_precipitation, dw_et, ci_cropping_intensity, etc.
- Fortnightly: mws_id, date, df_precipitation, df_et, season, etc.
"""

from __future__ import annotations

import polars as pl
import shapely.geometry as sgeom

from core_lens import AoI, SeasonConfig
from core_lens.entities import MWSEntity
from core_lens.export import geoparquet

DATA_ROOT = "/home/akashyap/Finals"

# ── 1. Registration ──────────────────────────────────────────────────────────
# Entities must be explicitly registered before any AoI is created.
# Only registered entities are scoped and accessible as aoi.<name>.

AoI.register(MWSEntity)

print("Registered entities:", AoI.registered_entities())
# → ['mws']


# ── 2. AoI construction — bounding box ───────────────────────────────────────
# Construct an AoI using a bounding box.

aoi = AoI(DATA_ROOT, bbox=(68.0, 6.0, 98.0, 38.0))

print("AoI geometry type :", type(aoi.geometry).__name__)
print("AoI current season:", aoi.current_season)
print("AoI current year  :", aoi.current_year)


# ── 3. AoI construction — alternative boundary modes ─────────────────────────

# From a raw bounding box
aoi_bbox = AoI(DATA_ROOT, bbox=(71.0, 25.0, 75.0, 28.0))

# From a pre-built Shapely polygon
poly = sgeom.box(71.0, 25.0, 75.0, 28.0)
aoi_geom = AoI(DATA_ROOT, geometry=poly)


# ── 4. AoI construction — custom SeasonConfig ────────────────────────────────
# Override default season date ranges.

aoi_custom = AoI(
    DATA_ROOT,
    ba_name="Barmer",
    seasons=SeasonConfig(
        kharif=("06-15", "10-15"),
        rabi=("10-16", "02-28"),
        zaid=("03-01", "06-14"),
    ),
)


# ── 5. Accessing the scoped entity ───────────────────────────────────────────
# aoi.mws is a lazy View — no Parquet I/O yet.
# It contains only the mws instances whose geometry intersects aoi.geometry.

view = aoi.mws
print("\nView keys (first 5):")
print(view.keys.head(5))


# ── 6. Attribute filtering with .where() ─────────────────────────────────────
# Filter by any static column. Multiple kwargs are AND-ed.
# Returns a new View — still lazy.

view_barmer = aoi.mws.where(ba_name="Barmer")
view_narrow = aoi.mws.where(ba_name="Barmer", state="Rajasthan")


# ── 7. Materialisation — .static ─────────────────────────────────────────────
# Triggers the actual Parquet read. Returns a Result with has_geometry=True
# because static files always carry geometry.

result_static = aoi.mws.static
print("\nStatic result columns (first 10):", result_static.columns[:10])
print("Resolution            :", result_static.resolution)  # Resolution.STATIC
print("Has geometry          :", result_static.has_geometry)  # True


# ── 8. Result accessors ──────────────────────────────────────────────────────
# df(), gdf(), lazy()
df_static = result_static.df()
print("\nDataFrame head (static):")
print(df_static.select(["mws_id", "tehsil", "district", "state", "area_in_ha"]).head(3))


# ── 9. Temporal querying & Materialisation — .annual & .fortnightly ──────────
# Use .between() to specify a time filter (year range, or season)
# Then use .annual or .fortnightly to materialise

# Annual data for a range of years
view_annual = view_barmer.between("2018-01-01", "2023-12-31")
result_annual = view_annual.annual
print("\nAnnual result columns (first 10):", result_annual.columns[:10])
print("Resolution            :", result_annual.resolution)
print("Has geometry          :", result_annual.has_geometry)  # False by default

# Fortnightly data for a specific season and year
result_fn = view_barmer.between(season="kharif", year=2021).fortnightly
print("\nFortnightly columns (first 10):", result_fn.columns[:10])


# ── 10. with_geometry() ──────────────────────────────────────────────────────
# .annual and .fortnightly lack geometry to save memory. Use with_geometry()
# to join geometry from the static table.

result_annual_geo = result_annual.with_geometry()
print("\nAnnual (with geometry) has_geometry:", result_annual_geo.has_geometry)


# ── 11. Derive — computed columns ────────────────────────────────────────────
# Append Polars expressions as a regular column. Fully chainable.

result_static_derived = result_static.derive(
    "area_km2",
    (pl.col("area_in_ha") / 100).round(2),
).derive(
    "dryspell_score",
    (pl.col("di_avg_dryspe") / pl.col("area_in_ha")).round(4),
)


# ── 12. Stats — describe ─────────────────────────────────────────────────────
# Descriptive statistics. For temporal data, you can group by entity.

desc_annual = result_annual.stats.describe(columns=["dw_precipitation", "dw_et"])
print("\nDescribe annual (overall):")
print(desc_annual.df())

desc_annual_by_entity = result_annual.stats.describe(
    columns=["dw_precipitation"],
    by="entity",  # Groups by entity across the years
)
print("\nDescribe annual (by entity):")
print(desc_annual_by_entity.df().head(5))


# ── 13. Stats — correlate ────────────────────────────────────────────────────
# Pairwise correlations on fortnightly data
corr = result_fn.stats.correlate(
    columns=["df_precipitation", "df_et"],
    method="pearson",
    across="entity",
)
print("\nCorrelation (Precipitation vs ET):")
print(corr.df())


# ── 14. Stats — test (group-based) ───────────────────────────────────────────
# Compare average dryspell distribution across districts.

test_result = result_static_derived.stats.test(
    column="di_avg_dryspe",
    groups="district",
    method="mann-whitney",
)
print("\nHypothesis test — dryspell by district:")
print(test_result.df())


# ── 15. Stats — change ───────────────────────────────────────────────────────
# Computes change over time. Perfect for annual/fortnightly results!

change_result = result_annual.stats.change(
    column="dw_et",
    from_period=2018,
    to_period=2023,
)
print("\nEvapotranspiration change (2018 vs 2023):")
print(change_result.df().head(5))


# ── 16. Stats — similarity ───────────────────────────────────────────────────
# Find the 5 MWS most similar to a target based on static attributes.

target_id = result_static_derived.df()["mws_id"][0]

sim = result_static_derived.stats.similarity(
    target=target_id,
    columns={
        "area_in_ha": None,
        "di_avg_dryspe": None,
    },
    method="euclidean",
    top_n=5,
)
print(f"\nMost similar MWS to {target_id}:")
print(sim.df())


# ── 17. Spatial filter ───────────────────────────────────────────────────────
# Filter by geometry directly — bbox or Shapely polygon.

view_poly = aoi.mws.spatial_filter(geometry=poly)
print("\nPolygon-filtered MWS count:", len(view_poly.static.df()))


# ── 18. Spatial Join ─────────────────────────────────────────────────────────
# Join attributes from another entity (e.g., if we had `DistrictEntity`).
# Below is a hypothetical example commented out.
#
# from core_lens.entities import DistrictEntity
# AoI.register(DistrictEntity)
# view_joined = aoi.mws.spatial_join(aoi.district, agg={"population": "sum"})
# result_joined = view_joined.static


# ── 19. Plot — choropleth ────────────────────────────────────────────────────
# Run in a Jupyter notebook for an interactive view.
# On a temporal result, you must use .with_geometry() first, or specify a subset.

try:
    map_html = change_result.with_geometry().plot.choropleth("dw_et_change")
    map_html.to_html("mws_change_map.html")
    print("Saved choropleth map to mws_change_map.html")
except Exception:
    pass


# ── 20. Plot — line (temporal trend) ─────────────────────────────────────────
# Works on temporal Result objects

try:
    line_fig = result_annual.plot.timeseries(x="year", y="dw_et")
    line_fig.write_html("mws_et_trend.html")
    print("Saved line plot to mws_et_trend.html")
except Exception:
    pass


# ── 21. Plot — scatter ───────────────────────────────────────────────────────
try:
    scatter_fig = result_static_derived.plot.scatter(x="area_km2", y="dryspell_score")
    scatter_fig.write_html("mws_scatter.html")
except Exception:
    pass


# ── 22. AoI map ──────────────────────────────────────────────────────────────
# Map with a specific subset as an overlay.

try:
    aoi_map = aoi.plot(overlay=result_static_derived)
    aoi_map.to_html("aoi_mws_map.html")
except Exception:
    pass


# ── 23. Export ───────────────────────────────────────────────────────────────

# GeoParquet — preserves geometry
geoparquet(result_static_derived, "output_mws.geoparquet")
print("GeoParquet written: output_mws.geoparquet")


# ── 24. Chained pipeline — full example ──────────────────────────────────────
# Combine filtering, temporal scoping, and materialisation.

final = (
    aoi.mws.where(ba_name="Barmer")
    .between(season="kharif", year=2021)
    .fortnightly.derive(
        "et_ratio", (pl.col("df_et") / pl.col("df_precipitation")).round(3)
    )
)

print("\nFull pipeline (fortnightly) result shape:", final.data.shape)
print("Columns:", final.columns[:10])

top_et = final.df().sort("et_ratio", descending=True).head(5)
print("\nTop 5 ET ratios in Barmer (Kharif 2021):")
print(top_et.select(["mws_id", "date", "df_et", "df_precipitation", "et_ratio"]))
