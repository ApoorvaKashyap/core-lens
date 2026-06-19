# Visualisation and Plots

The `result.plot` namespace provides integrated plotting capabilities. Spatial plots use **Lonboard** for high-performance interactive maps, while charts and timeseries use **Bokeh**.

Plot methods return plot objects (like a Bokeh Figure or Lonboard Map) which can be displayed using `bokeh.io.show` or exported via `bokeh.io.save`.

## Choropleth Maps

Generate interactive maps colored by a specific column. If the `Result` doesn't have geometry attached, it will automatically attach it by calling `.with_geometry()`.

```python
res = aoi.mws.annual
map_view = res.plot.choropleth(column="ndvi")
```

## Timeseries Plots

Plot timeseries data. You can aggregate the data or plot the top N entities individually.

```python
from bokeh.io import show

# Aggregate timeseries
fig_agg = res.plot.timeseries(x="year", y="ndvi", aggregate=True)
show(fig_agg)

# Timeseries for top entities
fig_top = res.plot.timeseries(x="year", y=["ndvi", "rainfall"], top_n=5)
```

## Scatter & Distribution Plots

Explore relationships and distributions in the data.

```python
# Scatter plot
scatter_fig = res.plot.scatter(x="rainfall", y="ndvi", top_n=10)

# Histogram / Distribution
dist_fig = res.plot.distribution(x=["ndvi", "rainfall"])
```

## Correlation Heatmaps & Scatter Matrices

Visualise correlations across variables.

```python
# Heatmap of correlation matrix
corr_fig = res.plot.correlation(columns=["ndvi", "rainfall", "temperature"])

# Scatter matrix (pairs plot)
matrix_fig = res.plot.matrix()
```
