"""Seasonal and monsoon-oriented aggregation utilities."""

from __future__ import annotations

import xarray as xr


def seasonal_mean(data: xr.DataArray, season: str, dim: str = "time") -> xr.DataArray:
    """Calculate a climatological mean for a named meteorological season.

    Supported seasons are DJF, MAM, JJA and SON. The season is represented
    by its conventional three calendar months.
    """
    months = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
    key = season.upper()
    if key not in months:
        raise ValueError("season must be one of DJF, MAM, JJA or SON.")
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(f"{dim!r} must be a dimension with a coordinate.")
    return data.where(data[dim].dt.month.isin(months[key]), drop=True).mean(dim=dim, skipna=True)


def monsoon_total(data: xr.DataArray, start_month: int = 6, end_month: int = 9, dim: str = "time") -> xr.DataArray:
    """Calculate annual June-September-style monsoon precipitation totals.

    ``start_month`` and ``end_month`` are inclusive and must define a
    non-wrapping calendar interval. For Indian summer monsoon studies the
    defaults are June through September.
    """
    if not 1 <= start_month <= 12 or not 1 <= end_month <= 12 or start_month > end_month:
        raise ValueError("start_month and end_month must define a valid non-wrapping interval.")
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(f"{dim!r} must be a dimension with a coordinate.")
    selected = data.where(data[dim].dt.month.isin(range(start_month, end_month + 1)), drop=True)
    return selected.groupby(f"{dim}.year").sum(dim=dim, skipna=True)
