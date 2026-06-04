"""Season-to-date-range resolution for time filtering."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from core_lens.aoi import SeasonConfig


def resolve_time_filter(
    time_filter: dict[str, str | int | tuple[int, int] | None],
    time_col: str,
    season_config: "SeasonConfig",
) -> pl.Expr:
    """Convert a ``View.time_filter`` dict to a Polars filter expression.

    Two modes are supported, matching the dict shapes produced by
    :meth:`~core_lens.base.view.View.between`:

    **Date range mode** — ``{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}``.
    Translated directly to a ``between`` predicate on ``time_col``.

    **Season mode** — ``{"season": name, "year": int | (int, int)}``.
    The season name is resolved to a ``(MM-DD, MM-DD)`` range from
    ``season_config``.  Year-crossing seasons (e.g. rabi: Nov–Mar) produce an
    ``OR`` expression.  When ``year`` is absent, all years are matched.  When
    ``year`` is a single integer, only that year (or the transition year for
    year-crossing seasons) is matched.

    Args:
        time_filter: The dict stored on :attr:`~core_lens.base.view.View.time_filter`.
        time_col: Name of the time column in the Parquet file.
        season_config: The :class:`~core_lens.aoi.SeasonConfig` in effect.

    Returns:
        A Polars expression that can be passed to ``.filter()``.

    Raises:
        ValueError: If ``time_filter`` has an unrecognised structure.
    """
    if "start" in time_filter and "end" in time_filter:
        start = time_filter["start"]
        end = time_filter["end"]
        assert isinstance(start, str)
        assert isinstance(end, str)
        return _date_range_expr(time_col, start, end)

    if "season" in time_filter:
        season_name = time_filter["season"]
        assert isinstance(season_name, str)
        if season_name == "current":
            season_name = season_config.season_for(datetime.date.today())
        year = time_filter.get("year")
        assert year is None or isinstance(year, int) or isinstance(year, tuple)
        return _season_expr(time_col, season_name, year, season_config)

    raise ValueError(
        f"Unrecognised time_filter structure: {time_filter!r}. "
        "Expected {'start': ..., 'end': ...} or {'season': ..., 'year': ...}."
    )


def _date_range_expr(time_col: str, start: str, end: str) -> pl.Expr:
    """Return a Polars expression for an inclusive ISO-8601 date range.

    Handles both ``pl.Date`` and ``pl.Int``/``pl.Int64`` time columns (year
    integers stored as integers rather than dates).

    Args:
        time_col: Name of the time column.
        start: ISO-8601 start date string (``"YYYY-MM-DD"``).
        end: ISO-8601 end date string (``"YYYY-MM-DD"``).

    Returns:
        A Polars expression.
    """
    start_date = datetime.date.fromisoformat(start)
    end_date = datetime.date.fromisoformat(end)
    col = pl.col(time_col)
    return col.is_between(pl.lit(start_date), pl.lit(end_date))


def _season_expr(
    time_col: str,
    season_name: str,
    year: int | tuple[int, int] | None,
    season_config: "SeasonConfig",
) -> pl.Expr:
    """Build a Polars expression for a season (and optional year range).

    For non-year-crossing seasons the expression is a single ``between`` call.
    For year-crossing seasons (e.g. rabi: Nov–Mar) the expression uses an
    ``OR`` of two half-range predicates so that rows from both calendar years
    are included.

    Args:
        time_col: Name of the time column.
        season_name: One of ``"kharif"``, ``"rabi"``, ``"zaid"``.
        year: Single year, ``(from_year, to_year)`` inclusive tuple, or ``None``
            to match all years.
        season_config: The :class:`~core_lens.aoi.SeasonConfig` in effect.

    Returns:
        A Polars expression.
    """
    start_md, end_md = getattr(season_config, season_name)
    year_crossing = start_md > end_md

    year_from, year_to = _year_bounds(year)

    col = pl.col(time_col)

    if not year_crossing:
        # Simple case: season stays within a single calendar year.
        # Build concrete date literals for the year range.
        start_date = datetime.date(year_from, *_parse_md(start_md))
        end_date = datetime.date(year_to, *_parse_md(end_md))
        return col.is_between(pl.lit(start_date), pl.lit(end_date))

    # Year-crossing season: split into two halves.
    # For rabi (11-01 → 03-31) in years [y_from, y_to]:
    #   - Nov–Dec of year in [y_from, y_to]
    #   - Jan–Mar of year in [y_from+1, y_to+1]
    start_m, start_d = _parse_md(start_md)
    end_m, end_d = _parse_md(end_md)

    late_start = datetime.date(year_from, start_m, start_d)
    late_end = datetime.date(year_to, 12, 31)

    early_start = datetime.date(year_from + 1, 1, 1)
    early_end = datetime.date(year_to + 1, end_m, end_d)

    return col.is_between(pl.lit(late_start), pl.lit(late_end)) | col.is_between(
        pl.lit(early_start), pl.lit(early_end)
    )


def _parse_md(md: str) -> tuple[int, int]:
    month, day = md.split("-")
    return int(month), int(day)


def _year_bounds(year: int | tuple[int, int] | None) -> tuple[int, int]:
    # When year is None use a wide range covering past and future data.
    if year is None:
        return 1900, 2100
    if isinstance(year, int):
        return year, year
    return year[0], year[1]
