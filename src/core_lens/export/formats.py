"""Module for Core Lens."""

import pathlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core_lens.base.result import Result

_DUCKDB_CONN = None


def _get_duckdb_conn() -> Any:
    global _DUCKDB_CONN
    if _DUCKDB_CONN is None:
        import duckdb

        _DUCKDB_CONN = duckdb.connect()
        _DUCKDB_CONN.execute("INSTALL spatial; LOAD spatial;")
    return _DUCKDB_CONN


def parquet(result: "Result", path: str | pathlib.Path, **kwargs: Any) -> None:
    """Export the result data to a standard Parquet file.

    This function uses Polars to write the underlying data frame.

    Args:
        result (Result): The Result object to export.
        path (str | pathlib.Path): Destination path for the Parquet file.
        **kwargs (Any): Additional options to pass to Polars `write_parquet`.

    Raises:
        TypeError: If the Result object has geometry.

    Example:
        >>> from core_lens.export import parquet
        >>> parquet(result, "output.parquet", compression="zstd", compression_level=3)

    """
    if result.has_geometry:
        raise TypeError(
            "This Result has geometry. Use geoparquet() to export geospatial data."
        )
    result.df().write_parquet(path, **kwargs)


def json(result: "Result", path: str | pathlib.Path, **kwargs: Any) -> None:
    """Export the result data to a standard JSON file.

    This function uses Polars to write the underlying data frame.

    Args:
        result (Result): The Result object to export.
        path (str | pathlib.Path): Destination path for the JSON file.
        **kwargs (Any): Additional options to pass to Polars `write_json`.

    Raises:
        TypeError: If the Result object has geometry.

    Example:
        >>> from core_lens.export import json
        >>> json(result, "output.json", pretty=True)

    """
    if result.has_geometry:
        raise TypeError(
            "This Result has geometry. Use geojson() to export geospatial data."
        )
    result.df().write_json(path, **kwargs)


def csv(result: "Result", path: str | pathlib.Path, **kwargs: Any) -> None:
    """Export the result data to a standard CSV file.

    This function uses Polars to write the underlying data frame.

    Args:
        result (Result): The Result object to export.
        path (str | pathlib.Path): Destination path for the CSV file.
        **kwargs (Any): Additional options to pass to Polars `write_csv`.

    Raises:
        TypeError: If the Result object has geometry.

    Example:
        >>> from core_lens.export import csv
        >>> csv(result, "output.csv", separator=",")

    """
    if result.has_geometry:
        raise TypeError(
            "This Result has geometry. Exporting geospatial data to CSV is not supported."
        )
    result.df().write_csv(path, **kwargs)


def geoparquet(result: "Result", path: str | pathlib.Path, **kwargs: Any) -> None:
    """Export the result data to a GeoParquet file.

    This function uses DuckDB to write the data frame with spatial extensions.

    Args:
        result (Result): The Result object to export.
        path (str | pathlib.Path): Destination path for the GeoParquet file.
        **kwargs (Any): Additional options to pass to DuckDB's COPY statement.

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

    df = result.df()
    geom_col = result.entity.geometry_col

    cols = []
    for c in df.columns:
        if c == geom_col:
            cols.append(f'ST_GeomFromWKB("{c}") AS "{c}"')
        else:
            cols.append(f'"{c}"')

    select_clause = ", ".join(cols)

    conn = _get_duckdb_conn()

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
    """  # noqa: S608
    conn.execute(query)


def geojson(result: "Result", path: str | pathlib.Path, **kwargs: Any) -> None:
    """Export the result data to a GeoJSON file.

    This function natively streams the data directly from Polars and Shapely
    without invoking GDAL or DuckDB, making it fast for large datasets.

    Args:
        result (Result): The Result object to export.
        path (str | pathlib.Path): Destination path for the GeoJSON file.
        **kwargs (Any): Driver-specific kwargs are currently ignored.
            The `driver` keyword argument supports `GeoJSONSeq` for newline-delimited GeoJSON.

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

    geom_col = result.entity.geometry_col
    data = result.df()

    driver = kwargs.pop("driver", "GeoJSON")
    is_seq = driver == "GeoJSONSeq"

    if kwargs:
        from loguru import logger

        logger.warning(
            "geojson() now uses a custom streaming Polars writer. "
            "Driver-specific kwargs ({}) are ignored.",
            list(kwargs.keys()),
        )

    import shapely
    import tempfile

    wkb_arr = data[geom_col].to_numpy()
    geoms = shapely.from_wkb(wkb_arr)
    geom_strs = shapely.to_geojson(geoms)

    with tempfile.NamedTemporaryFile(mode="w+") as tmp:
        data.drop(geom_col).write_ndjson(tmp.name)
        tmp.seek(0)

        with open(path, "w") as out:
            if not is_seq:
                out.write('{"type": "FeatureCollection", "features": [\n')

            first = True
            for i, prop_json_str in enumerate(tmp):
                if not first:
                    out.write("\n" if is_seq else ",\n")
                else:
                    first = False

                # construct the Feature JSON manually to avoid parsing/re-serialising
                out.write(
                    f'{{"type": "Feature", "geometry": {geom_strs[i]}, '
                    f'"properties": {prop_json_str.strip()}}}'
                )

            if not is_seq:
                out.write("\n]}\n")
            else:
                out.write("\n")
