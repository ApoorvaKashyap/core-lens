import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core_lens.base.result import Result


def parquet(result: "Result", path: str | pathlib.Path, **kwargs: Any) -> None:
    """Export the result data to a standard Parquet file.

    This function uses Polars to write the underlying data frame.

    Args:
        result: The Result object to export.
        path: Destination path for the Parquet file.
        **kwargs: Additional options to pass to Polars `write_parquet`.

    Example:
        >>> from core_lens.export import parquet
        >>> parquet(result, "output.parquet", compression="zstd", compression_level=3)
    """
    result.df().write_parquet(path, **kwargs)


def json(result: "Result", path: str | pathlib.Path, **kwargs: Any) -> None:
    """Export the result data to a standard JSON file.

    This function uses Polars to write the underlying data frame.

    Args:
        result: The Result object to export.
        path: Destination path for the JSON file.
        **kwargs: Additional options to pass to Polars `write_json`.

    Example:
        >>> from core_lens.export import json
        >>> json(result, "output.json", pretty=True)
    """
    result.df().write_json(path, **kwargs)


def geoparquet(result: "Result", path: str | pathlib.Path, **kwargs: Any) -> None:
    """Export the result data to a GeoParquet file.

    This function uses DuckDB to write the data frame with spatial extensions.

    Args:
        result: The Result object to export.
        path: Destination path for the GeoParquet file.
        **kwargs: Additional options to pass to DuckDB's COPY statement.

    Example:
        >>> from core_lens.export import geoparquet
        >>> geoparquet(
        ...     result.with_geometry(),
        ...     "output.geoparquet",
        ...     compression="ZSTD",
        ...     partition_by="year"
        ... )

    Raises:
        TypeError: If the Result object does not have geometry.
    """
    if not result.has_geometry:
        raise TypeError(
            "This Result has no geometry. "
            "Call .with_geometry() first to join the static geometry column before exporting to geoparquet."
        )

    import duckdb

    df = result.df()
    geom_col = result.entity.geometry_col

    cols = []
    for c in df.columns:
        if c == geom_col:
            cols.append(f'ST_GeomFromWKB("{c}") AS "{c}"')
        else:
            cols.append(f'"{c}"')

    select_clause = ", ".join(cols)

    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")

    # We must escape single quotes in path if any exist.
    path_str = str(path).replace("'", "''")

    options = ["FORMAT PARQUET"]
    for k, v in kwargs.items():
        if k.lower() == "partition_by":
            if isinstance(v, str):
                options.append(f"PARTITION_BY ({v})")
            else:
                options.append(f"PARTITION_BY ({', '.join(v)})")
        elif isinstance(v, bool):
            options.append(f"{k.upper()} {'TRUE' if v else 'FALSE'}")
        elif isinstance(v, str):
            options.append(f"{k.upper()} '{v}'")
        else:
            options.append(f"{k.upper()} {v}")

    options_str = ", ".join(options)

    query = f"""
    COPY (
        SELECT {select_clause}
        FROM df
    ) TO '{path_str}' ({options_str});
    """
    conn.execute(query)


def geojson(result: "Result", path: str | pathlib.Path, **kwargs: Any) -> None:
    """Export the result data to a GeoJSON file.

    This function uses DuckDB to write the data frame with spatial extensions.

    Args:
        result: The Result object to export.
        path: Destination path for the GeoJSON file.
        **kwargs: Additional options to pass to DuckDB's COPY statement.

    Example:
        >>> from core_lens.export import geojson
        >>> geojson(result.with_geometry(), "output.json")

    Raises:
        TypeError: If the Result object does not have geometry.
    """
    if not result.has_geometry:
        raise TypeError(
            "This Result has no geometry. "
            "Call .with_geometry() first to join the static geometry column before exporting to geojson."
        )

    import duckdb

    df = result.df()
    geom_col = result.entity.geometry_col

    cols = []
    for c in df.columns:
        if c == geom_col:
            cols.append(f'ST_GeomFromWKB("{c}") AS "{c}"')
        else:
            cols.append(f'"{c}"')

    select_clause = ", ".join(cols)

    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")

    path_str = str(path).replace("'", "''")

    options = ["FORMAT GDAL", "DRIVER 'GeoJSON'"]
    for k, v in kwargs.items():
        if isinstance(v, bool):
            options.append(f"{k.upper()} {'TRUE' if v else 'FALSE'}")
        elif isinstance(v, str):
            options.append(f"{k.upper()} '{v}'")
        else:
            options.append(f"{k.upper()} {v}")

    options_str = ", ".join(options)

    query = f"""
    COPY (
        SELECT {select_clause}
        FROM df
    ) TO '{path_str}' ({options_str});
    """
    conn.execute(query)
