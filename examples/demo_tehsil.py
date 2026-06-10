"""
demo_tehsil.py — core_lens TehsilEntity capability demonstration
=================================================================
Walks through every meaningful operation available on a static-only
entity: registration, AoI construction, filtering, materialisation,
geometry, derive, stats, plot, and export.

Prerequisites
-------------
    pip install core-lens geopandas shapely polars scipy plotly lonboard

Data layout expected
--------------------
    data/
    └── tehsil/
        └── static/
            └── tehsil.parquet   ← GeoParquet with WKB geometry in "geom"

Columns present: OGC_FID, id, District, STATE, Shape_Area, Shape_Leng,
                 TEHSIL, geom
"""

from __future__ import annotations

import polars as pl

from core_lens import AoI, SeasonConfig
from core_lens.entities import TehsilEntity
from core_lens.export import geoparquet
import shapely.geometry as sgeom


DATA_ROOT = "data/"

# ── 1. Registration ──────────────────────────────────────────────────────────
# Entities must be explicitly registered before any AoI is created.
# Only registered entities are scoped and accessible as aoi.<name>.

AoI.register(TehsilEntity)

print("Registered entities:", AoI.registered_entities())
# → ['tehsil']


# ── 2. AoI construction — bounding box ───────────────────────────────────────
# Construct an AoI using a bounding box that covers all of India to
# plot all tehsils in the dataset.

aoi = AoI(DATA_ROOT, bbox=(68.0, 6.0, 98.0, 38.0))

print("AoI geometry type :", type(aoi.geometry).__name__)  # Polygon / MultiPolygon
print("AoI current season:", aoi.current_season)  # kharif | rabi | zaid
print("AoI current year  :", aoi.current_year)


# ── 3. AoI construction — alternative boundary modes ─────────────────────────

# From a raw bounding box (near BIHAR)
aoi_bbox = AoI(DATA_ROOT, bbox=(86.0, 21.0, 87.5, 22.0))

# From a pre-built Shapely polygon
poly = sgeom.box(86.0, 21.0, 87.5, 22.0)
aoi_geom = AoI(DATA_ROOT, geometry=poly)


# ── 4. AoI construction — custom SeasonConfig ────────────────────────────────
# Override default season date ranges for a different agro-climatic zone.

aoi_custom = AoI(
    DATA_ROOT,
    TEHSIL="DANAPUR",
    District="PATNA",
    seasons=SeasonConfig(
        kharif=("06-15", "10-15"),
        rabi=("10-16", "02-28"),
        zaid=("03-01", "06-14"),
    ),
)


# ── 5. Accessing the scoped entity ───────────────────────────────────────────
# aoi.tehsil is a lazy View — no Parquet I/O yet.
# It contains only the tehsil instances whose geometry intersects aoi.geometry.

view = aoi.tehsil
print("\nView keys (first 5):")
print(view.keys.head(5))  # pl.DataFrame with 'id' column


# ── 6. Attribute filtering with .where() ─────────────────────────────────────
# Filter by any static column.  Multiple kwargs are AND-ed.
# Returns a new View — still lazy.

view_chamba = aoi.tehsil.where(District="PATNA")
view_hp = aoi.tehsil.where(STATE="BIHAR")
view_narrow = aoi.tehsil.where(District="PATNA", STATE="BIHAR")


# ── 7. Materialisation — .static ─────────────────────────────────────────────
# Triggers the actual Parquet read.  Returns a Result with has_geometry=True
# because static files always carry geometry.

result = aoi.tehsil.static
print("\nResult columns  :", result.columns)
print("Result resolution:", result.resolution)  # Resolution.STATIC
print("Has geometry    :", result.has_geometry)  # True
print("Entity name     :", result.entity_name)  # 'tehsil'
print("Shape           :", result.data.shape)


# ── 8. Result accessors ───────────────────────────────────────────────────────

# Polars DataFrame — fast, no geometry overhead
df: pl.DataFrame = result.df()
print("\nDataFrame head:")
print(df.head(3))

# GeoDataFrame — use when you need geometry operations
gdf = result.gdf()
print("\nGeoDataFrame CRS :", gdf.crs)
print("GeoDataFrame head:")
print(gdf.head(3))

# Polars LazyFrame — escape hatch for arbitrary custom transforms
lf: pl.LazyFrame = result.lazy()


# ── 9. Derive — computed columns ─────────────────────────────────────────────
# .derive() appends a Polars expression as a regular column.
# Fully chainable.  Shape_Area is in square degrees; scale to approx km².

result_with_area = aoi.tehsil.static.derive(
    "area_km2",
    (pl.col("Shape_Area") * 12_308).round(2),  # rough WGS-84 → km² factor
).derive(
    "compactness",
    (pl.col("Shape_Area") / (pl.col("Shape_Leng") ** 2)).round(4),
)

print("\nDerived columns:", result_with_area.columns)


# ── 10. Stats — describe ──────────────────────────────────────────────────────
# Descriptive statistics, by column (default) or by entity.

desc_by_col = result_with_area.stats.describe(
    columns=["area_km2", "compactness", "Shape_Leng"]
)
print("\nDescribe by column:")
print(desc_by_col.df())

desc_by_entity = result_with_area.stats.describe(
    columns=["area_km2", "Shape_Leng"],
    by="entity",
)
print("\nDescribe by entity (first 5):")
print(desc_by_entity.df().head(5))


# ── 11. Stats — correlate ─────────────────────────────────────────────────────
# Pairwise Pearson / Spearman / Kendall correlations.
# data: column_a | column_b | correlation | p_value

corr = result_with_area.stats.correlate(
    columns=["area_km2", "Shape_Leng", "compactness"],
    method="spearman",
    across="mws",
)
print("\nCorrelation (Spearman):")
print(corr.df())
print("metadata:", corr.metadata)


# ── 12. Stats — test (group-based) ───────────────────────────────────────────
# Compare area distributions across states.
# data: group | n | mean | std | median
# metadata: statistic, p_value, significant, significance_level

try:
    test_result = result_with_area.stats.test(
        column="area_km2",
        groups="STATE",
        method="mann-whitney",
    )
    print("\nHypothesis test — area by STATE:")
    print(test_result.df())
    print("p_value:", test_result.metadata["p_value"])
    print("significant:", test_result.metadata["significant"])
except IndexError:
    print("\nHypothesis test skipped: not enough groups in the sample dataset.")


# ── 13. Stats — test (single-sample) ─────────────────────────────────────────
# Test whether mean area differs from a reference value.

test_vs_ref = result_with_area.stats.test(
    column="area_km2",
    against=500.0,  # reference area in km²
    method="t-test",
)
print("\nOne-sample t-test vs 500 km²:")
print(test_vs_ref.metadata)


# ── 14. Stats — change ────────────────────────────────────────────────────────
# Note: change() is most meaningful on annual/fortnightly data.
# For static entities it still works if the data carries a numeric column
# that can be compared between two named groups or derived states.
# Here we illustrate absolute change between two derived values stored
# in separate filtered views joined externally.
#
# (Skipped for pure static entity — included as reference)
#
# result.stats.change(column="tree_cover", from_period=2018, to_period=2023)


# ── 15. Stats — similarity ────────────────────────────────────────────────────
# Find the 5 tehsils most geometrically similar to a target tehsil
# based on area and boundary length.

target_id = result_with_area.df()["id"][0]

sim = result_with_area.stats.similarity(
    target=target_id,  # an id value that exists in data
    columns={
        "area_km2": None,
        "Shape_Leng": None,
        "compactness": None,
    },
    method="euclidean",
    top_n=5,
)
print(f"\nMost similar tehsils to {target_id}:")
print(sim.df())
print("metadata:", sim.metadata)


# ── 16. Spatial filter ────────────────────────────────────────────────────────
# Filter by geometry directly — bbox or Shapely polygon.
# Uses the in-memory bbox index for fast pre-filter, then STRtree exact check.

view_bbox = aoi.tehsil.spatial_filter(bbox=(86.5, 21.2, 87.0, 21.8))
result_bbox = view_bbox.static
print("\nSpatially filtered count:", len(result_bbox.df()))

view_poly = aoi.tehsil.spatial_filter(geometry=poly)
result_poly = view_poly.static
print("Polygon-filtered count  :", len(result_poly.df()))


# ── 17. with_geometry() ───────────────────────────────────────────────────────
# .static already has geometry.  On annual/fortnightly results (other entities)
# you'd call this to join geometry back in.  Demonstrated here for completeness.

result_geo = result.with_geometry()  # no-op since already has_geometry=True
print("\nwith_geometry is_noop:", result_geo is result)


# ── 18. Plot — choropleth ─────────────────────────────────────────────────────
# Returns a Lonboard Map object.  Run in a Jupyter notebook for interactive view.

choropleth_map = result_with_area.plot.choropleth("area_km2")
choropleth_map.to_html("choropleth_map.html")
print("Saved choropleth map to choropleth_map.html")


# ── Plot subset — BIHAR only (used for Plotly demos below) ───────────────────
# Avoids materialising all-India geometries for every plot call.
# Scatter/dist/corr/heatmap don't need the full dataset.

result_plot = (
    aoi.tehsil.where(STATE="BIHAR")
    .static.derive(
        "area_km2",
        (pl.col("Shape_Area") * 12_308).round(2),
    )
    .derive(
        "compactness",
        (pl.col("Shape_Area") / (pl.col("Shape_Leng") ** 2)).round(4),
    )
)


# ── 19. Plot — scatter ────────────────────────────────────────────────────────
scatter_fig = result_plot.plot.scatter(x="area_km2", y="Shape_Leng")
scatter_fig.show()


# ── 20. Plot — distribution ───────────────────────────────────────────────────
dist_fig = result_plot.plot.distribution(x="area_km2")
dist_fig.show()


# ── 21. Plot — correlation matrix ────────────────────────────────────────────
corr_fig = result_plot.plot.correlation(
    columns=["area_km2", "Shape_Leng", "compactness"]
)
corr_fig.show()


# ── 22. Plot — heatmap ────────────────────────────────────────────────────────
# Useful when you have a pivot-able structure.  Here: District vs STATE
# with mean area_km2 as the value.

heatmap_fig = result_plot.plot.heatmap(
    x="District",
    y="STATE",
    value="area_km2",
)
heatmap_fig.show()


# ── 23. AoI map ───────────────────────────────────────────────────────────────


# Use result_plot (BIHAR subset) as overlay — avoids decoding all-India geometries.
aoi_map_with_overlay = aoi.plot(overlay=result_plot)
aoi_map_with_overlay.to_html("aoi_map_with_overlay.html")
print("Saved AoI map with overlay to aoi_map_with_overlay.html")


# ── 24. Export ────────────────────────────────────────────────────────────────
# All four formats from core_lens.export.

# Standard Parquet (no geometry)
result_no_geom = result_with_area.derive(
    "_drop_geom_marker",
    pl.lit(True),  # geometry is already in data; strip it
)
# parquet() raises if has_geometry=True — drop geometry first via .df()
pl_df = result_with_area.df().drop("geom")
pl_df.write_parquet("output_tehsil.parquet")
print("\nParquet written: output_tehsil.parquet")

# JSON (tabular, no geometry)
pl_df.write_json("output_tehsil.json")
print("JSON written   : output_tehsil.json")

# GeoParquet — uses DuckDB spatial extension, preserves WKB geometry
geoparquet(result_with_area, "output_tehsil.geoparquet")
print("GeoParquet written: output_tehsil.geoparquet")

# ── 25. Chained pipeline — full example ──────────────────────────────────────
# Combine everything in one readable chain.

final = (
    aoi.tehsil.where(STATE="BIHAR")  # scoped View  # attribute filter
    .static.derive(  # materialise  # add computed column
        "area_km2",
        (pl.col("Shape_Area") * 12_308).round(2),
    )
    .derive(
        "compactness",
        (pl.col("Shape_Area") / (pl.col("Shape_Leng") ** 2)).round(4),
    )
)

print("\nFull pipeline result shape:", final.data.shape)
print("Columns:", final.columns)

top_largest = final.df().sort("area_km2", descending=True).head(5)
print("\nTop 5 largest tehsils in BIHAR:")
print(top_largest.select(["TEHSIL", "District", "area_km2", "compactness"]))
