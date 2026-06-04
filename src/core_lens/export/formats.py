import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core_lens.base.result import Result


def parquet(result: "Result", path: str | pathlib.Path) -> None:
    """Export the result data to a standard Parquet file.

    This function uses Polars to write the underlying data frame.

    Args:
        result: The Result object to export.
        path: Destination path for the Parquet file.
    """
    result.df().write_parquet(path)


def json(result: "Result", path: str | pathlib.Path) -> None:
    """Export the result data to a standard JSON file.

    This function uses Polars to write the underlying data frame.

    Args:
        result: The Result object to export.
        path: Destination path for the JSON file.
    """
    result.df().write_json(path)


def geoparquet(result: "Result", path: str | pathlib.Path) -> None:
    """Export the result data to a GeoParquet file.

    This function uses DuckDB to write the data frame with spatial extensions.

    Args:
        result: The Result object to export.
        path: Destination path for the GeoParquet file.

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

    query = f"""
    COPY (
        SELECT {select_clause}
        FROM df
    ) TO '{path_str}' (FORMAT PARQUET);
    """
    conn.execute(query)


def geojson(result: "Result", path: str | pathlib.Path) -> None:
    """Export the result data to a GeoJSON file.

    This function uses DuckDB to write the data frame with spatial extensions.

    Args:
        result: The Result object to export.
        path: Destination path for the GeoJSON file.

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

    query = f"""
    COPY (
        SELECT {select_clause}
        FROM df
    ) TO '{path_str}' (FORMAT GDAL, DRIVER 'GeoJSON');
    """
    conn.execute(query)
