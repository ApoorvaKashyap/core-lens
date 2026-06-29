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
from core_lens.base.namespaces.stats import CorrelateMethod

corr = res.stats.correlate(
    columns=["ndvi", "rainfall", "temperature"],
    method=CorrelateMethod.PEARSON, # or SPEARMAN, KENDALL
    across="entity"      # correlate across entities or time
)
```

## Hypothesis Testing

Test for significant differences between groups or periods:

```python
from core_lens.base.namespaces.stats import TestMethod

# Group-based testing
test_res = res.stats.test(
    column="cropping_intensity",
    groups="temperature_zone",
    method=TestMethod.MANN_WHITNEY
)

# Period-based testing
test_period = res.stats.test(
    column="ndvi",
    periods=[(2010, 2015), (2016, 2023)],
    method=TestMethod.T_TEST
)
```

## Change Detection

Analyse absolute, percentage, or trend changes over time:

```python
from core_lens.base.namespaces.stats import ChangeMethod

# Trend over time
trend = res.stats.change(
    column="ndvi",
    from_period=2010,
    to_period=2023,
    method=ChangeMethod.TREND
)

# Absolute or percentage change
pct_change = res.stats.change(
    column="tree_cover",
    from_period=2018,
    to_period=2023,
    method=ChangeMethod.PERCENTAGE
)
```

## Anomaly Detection

Identify anomalies against a historical baseline or cross-sectionally:

```python
from core_lens.base.namespaces.stats import AnomalyTsMethod, AnomalyCrossMethod

# Timeseries anomaly against its own history
ts_anomalies = res.stats.anomaly(
    column="ndvi",
    mode="timeseries",
    method=AnomalyTsMethod.STL,
    baseline=(2010, 2018)
)

# Cross-sectional anomaly against other entities
cross_anomalies = res.stats.anomaly(
    column="ndvi",
    mode="cross_sectional",
    method=AnomalyCrossMethod.ZSCORE,
    baseline=(2010, 2020)
)
```

## Similarity Search

Find entities similar to a target entity across multiple dimensions:

```python
from core_lens.base.namespaces.stats import SimilarityMethod

similar = res.stats.similarity(
    target="13_551",
    columns={
        "rainfall": ("annual", {"year": 2018}),
        "ndvi": ("fortnightly", {"season": "kharif", "year": 2020})
    },
    method=SimilarityMethod.EUCLIDEAN,
    top_n=10
)
```
