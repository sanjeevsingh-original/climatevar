"""ETCCDI-inspired daily precipitation indices.

All functions expect a daily precipitation ``xarray.DataArray`` with a time
coordinate. Values are assumed to be precipitation amounts for each day,
not rates. Units are preserved; callers should ensure the input units are
consistent (for example, mm/day).
"""

from __future__ import annotations

import xarray as xr


def _validate_daily(data: xr.DataArray, dim: str) -> None:
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(f"{dim!r} must be a dimension with a coordinate.")


def _year_group(data: xr.DataArray, dim: str) -> str:
    _validate_daily(data, dim)
    return f"{dim}.year"


def rx1day(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Annual maximum 1-day precipitation (Rx1day)."""
    return data.groupby(_year_group(data, dim)).max(dim=dim, skipna=True)


def rx5day(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Annual maximum consecutive 5-day precipitation (Rx5day)."""
    _validate_daily(data, dim)
    rolling = data.rolling({dim: 5}, min_periods=5).sum()
    return rolling.groupby(f"{dim}.year").max(dim=dim, skipna=True)


def prcptot(
    data: xr.DataArray, dim: str = "time", wet_day_threshold: float = 1.0
) -> xr.DataArray:
    """Annual precipitation total on wet days.

    By default, wet days are days with precipitation >= 1.0 in the input
    units. Missing values are excluded from the sum.
    """
    _validate_daily(data, dim)
    wet = data.where(data >= wet_day_threshold)
    return wet.groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def r10mm(data: xr.DataArray, dim: str = "time", threshold: float = 10.0) -> xr.DataArray:
    """Annual count of days with precipitation >= 10 mm (R10mm)."""
    _validate_daily(data, dim)
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def r20mm(data: xr.DataArray, dim: str = "time", threshold: float = 20.0) -> xr.DataArray:
    """Annual count of days with precipitation >= 20 mm (R20mm)."""
    _validate_daily(data, dim)
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def _percentile_total(
    data: xr.DataArray,
    reference: xr.DataArray,
    percentile: float,
    dim: str,
) -> xr.DataArray:
    _validate_daily(data, dim)
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1.")
    if reference.ndim != data.ndim - 1 and reference.dims != tuple(d for d in data.dims if d != dim):
        raise ValueError("reference must be a spatial/other-dimensional threshold without the time dimension.")
    threshold = data.quantile(percentile, dim=dim, skipna=True) if reference is None else reference
    wet = data.where(data > threshold)
    return wet.groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def r95p(
    data: xr.DataArray,
    reference: xr.DataArray | None = None,
    dim: str = "time",
    wet_day_threshold: float = 1.0,
) -> xr.DataArray:
    """Annual precipitation from very wet days (above the 95th percentile).

    ``reference`` should be a fixed 95th-percentile threshold calculated from
    an independently selected baseline period. If omitted, the threshold is
    calculated from the supplied data itself, which is convenient for
    exploratory analysis but should be documented in research applications.
    """
    _validate_daily(data, dim)
    threshold = data.where(data >= wet_day_threshold).quantile(0.95, dim=dim, skipna=True) if reference is None else reference
    return data.where(data > threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def r99p(
    data: xr.DataArray,
    reference: xr.DataArray | None = None,
    dim: str = "time",
    wet_day_threshold: float = 1.0,
) -> xr.DataArray:
    """Annual precipitation from extremely wet days (above the 99th percentile)."""
    _validate_daily(data, dim)
    threshold = data.where(data >= wet_day_threshold).quantile(0.99, dim=dim, skipna=True) if reference is None else reference
    return data.where(data > threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def _max_run(data: xr.DataArray, condition: xr.DataArray, dim: str) -> xr.DataArray:
    """Return the maximum consecutive True run along ``dim`` within each year."""
    _validate_daily(data, dim)
    values = condition.fillna(False).astype(int)
    groups = values.groupby(f"{dim}.year")

    def run_length(block: xr.DataArray) -> xr.DataArray:
        reset = block.where(block == 0).ffill(dim=dim).fillna(-1)
        return block.groupby(reset).cumsum(dim=dim).max(dim=dim, skipna=True)

    return groups.map(run_length)


def cwd(data: xr.DataArray, dim: str = "time", wet_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual maximum consecutive wet days (CWD)."""
    return _max_run(data, data >= wet_day_threshold, dim)


def cdd(data: xr.DataArray, dim: str = "time", dry_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual maximum consecutive dry days (CDD).

    Dry days are precipitation values < ``dry_day_threshold``.
    """
    return _max_run(data, data < dry_day_threshold, dim)
