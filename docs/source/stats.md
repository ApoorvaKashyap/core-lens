# Statistical Analysis

The `result.stats` namespace provides powerful analytical tools for geospatial and timeseries data. All statistical methods return a new `Result` object with computed `data` and populated `metadata`.

## Descriptive Statistics

```python
# Describe specific columns
desc = res.stats.describe(columns=["ndvi", "rainfall"])

# Breakdown per entity
desc_entity = res.stats.describe(by="entity")
```

## Correlation

Find temporal or spatial correlations between variables:

```python
corr = res.stats.correlate(
    columns=["ndvi", "rainfall", "temperature"],
    method="pearson", # or "spearman", "kendall"
    across="entity"      # correlate across entities or time
)
```

## Hypothesis Testing

Test for significant differences between groups or periods:

```python
# Group-based testing
test_res = res.stats.test(
    column="cropping_intensity",
    groups="temperature_zone",
    method="mann-whitney"
)

# Period-based testing
test_period = res.stats.test(
    column="ndvi",
    periods=[(2010, 2015), (2016, 2023)],
    method="t-test"
)
```

## Change Detection

Analyse absolute, percentage, or trend changes over time:

```python
# Trend over time
trend = res.stats.change(
    column="ndvi",
    from_period=2010,
    to_period=2023,
    method="trend"
)

# Absolute or percentage change
pct_change = res.stats.change(
    column="tree_cover",
    from_period=2018,
    to_period=2023,
    method="percentage"
)
```

## Anomaly Detection

Identify anomalies against a historical baseline or cross-sectionally:

```python
# Timeseries anomaly against its own history
ts_anomalies = res.stats.anomaly(
    column="ndvi",
    mode="timeseries",
    method="stl",
    baseline=(2010, 2018)
)

# Cross-sectional anomaly against other entities
cross_anomalies = res.stats.anomaly(
    column="ndvi",
    mode="cross_sectional",
    method="zscore",
    baseline=(2010, 2020)
)
```

## Similarity Search

Find entities similar to a target entity across multiple dimensions:

```python
similar = res.stats.similarity(
    target="13_551",
    columns={
        "rainfall": ("annual", {"year": 2018}),
        "ndvi": ("fortnightly", {"season": "kharif", "year": 2020})
    },
    method="euclidean",
    top_n=10
)
```
