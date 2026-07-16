from . import __version__


def hello() -> str:
    hello_str = """Hello from core-lens!

The following primary capabilities are available:
- AoI: Construct Area of Interest queries (e.g., `AoI(district="Godda")`)
- Result: Lazy evaluation and computation graph materialisation

Analytics (via `result.stats`):
- anomaly(): Timeseries and cross-sectional anomaly detection
- change(): Absolute, percentage, and trend changes over time
- similarity(): Multi-feature similarity search across entities
- test(): Statistical hypothesis testing
- correlate(): Feature correlation analysis

Visualisation (via `result.plot`):
- choropleth(): Interactive Lonboard spatial mapping
- timeseries(): Temporal data visualisation
- scatter(): Scatter plots for variable comparison
- distribution(): Histograms and density plots
- correlation(): Correlation heatmaps between variables
- heatmap(): 2D density/binned heatmaps
- matrix(): Scatter matrix/pair plots
"""
    return hello_str


def version() -> str:
    return str(__version__)


if __name__ == "__main__":
    print(hello())
    print(f"Version: {version()}")
