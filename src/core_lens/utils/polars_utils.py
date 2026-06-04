"""Polars scan helpers with predicate pushdown for entity materialisation."""

from __future__ import annotations

import polars as pl


def scan_with_key_filter(
    path: str,
    key_cols: list[str],
    key_values: pl.DataFrame,
    time_expr: pl.Expr | None = None,
) -> pl.LazyFrame:
    """Return a ``pl.LazyFrame`` filtered to the given keys and optional time range.

    Uses ``pl.scan_parquet`` with two predicate-pushdown layers:

    1. **Key filter** — restricts to entity instances whose key column(s) are
       in ``key_values``.  For a single-column key this is an ``is_in``
       predicate pushed down to the Parquet reader.  For composite keys each
       column is filtered independently (over-selects slightly, then pruned
       by the join at collect time).

    2. **Time filter** — an optional Polars expression appended with ``&``,
       also pushed down if the Parquet file carries column statistics.

    Args:
        path: Absolute path to a Parquet file.
        key_cols: Column name(s) that form the entity's unique key.
        key_values: A narrow ``pl.DataFrame`` containing only the key
            column(s) with the exact values to retain.
        time_expr: An optional Polars filter expression for the time column,
            as produced by :func:`~core_lens.utils.season.resolve_time_filter`.

    Returns:
        A ``pl.LazyFrame`` ready to be ``.collect()``-ed.
    """
    lf = pl.scan_parquet(path)

    if len(key_cols) == 1:
        key = key_cols[0]
        values = key_values[key].to_list()
        lf = lf.filter(pl.col(key).is_in(values))
    else:
        # Composite key: filter each column independently. A small over-selection
        # is acceptable because the subsequent join at collect time is exact.
        for key in key_cols:
            values = key_values[key].to_list()
            lf = lf.filter(pl.col(key).is_in(values))

    if time_expr is not None:
        lf = lf.filter(time_expr)

    return lf
