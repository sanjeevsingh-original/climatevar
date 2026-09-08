"""ETCCDI-inspired daily precipitation indices.

All functions expect a daily precipitation ``xarray.DataArray`` with a time
coordinate. Values are assumed to be precipitation amounts for each day,
not rates. Units are preserved; callers should ensure the input units are
consistent (for example, mm/day).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _validate_daily(data: xr.DataArray, dim: str) -> None:
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(f"{dim!r} must be a dimension with a coordinate.")


def rx1day(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Annual maximum 1-day precipitation (Rx1day)."""
    _validate_daily(data, dim)
    return data.groupby(f"{dim}.year").max(dim=dim, skipna=True)


def rx5day(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Annual maximum consecutive 5-day precipitation (Rx5day)."""
    _validate_daily(data, dim)
    rolling = data.rolling({dim: 5}, min_periods=5).sum()
    return rolling.groupby(f"{dim}.year").max(dim=dim, skipna=True)


def prcptot(
    data: xr.DataArray, dim: str = "time", wet_day_threshold: float = 1.0
) -> xr.DataArray:
    """Annual precipitation total on wet days (default threshold: 1.0)."""
    _validate_daily(data, dim)
    wet = data.where(data >= wet_day_threshold)
    return wet.groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def r10mm(data: xr.DataArray, dim: str = "time", threshold: float = 10.0) -> xr.DataArray:
    """Annual count of days with precipitation >= 10 mm."""
    _validate_daily(data, dim)
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def r20mm(data: xr.DataArray, dim: str = "time", threshold: float = 20.0) -> xr.DataArray:
    """Annual count of days with precipitation >= 20 mm."""
    _validate_daily(data, dim)
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def _percentile_total(
    data: xr.DataArray, reference: xr.DataArray | None, percentile: float, dim: str
) -> xr.DataArray:
    _validate_daily(data, dim)
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1.")
    threshold = (
        data.where(data >= 1.0).quantile(percentile, dim=dim, skipna=True)
        if reference is None
        else reference
    )
    return data.where(data > threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def r95p(
    data: xr.DataArray,
    reference: xr.DataArray | None = None,
    dim: str = "time",
) -> xr.DataArray:
    """Annual precipitation from days above a 95th-percentile threshold.

    ``reference`` should normally be a fixed threshold calculated from an
    independently selected baseline period. If omitted, the threshold is
    calculated from the supplied data for exploratory analysis.
    """
    return _percentile_total(data, reference, 0.95, dim)


def r99p(
    data: xr.DataArray,
    reference: xr.DataArray | None = None,
    dim: str = "time",
) -> xr.DataArray:
    """Annual precipitation from days above a 99th-percentile threshold."""
    return _percentile_total(data, reference, 0.99, dim)


def _max_consecutive_1d(values: np.ndarray) -> int:
    """Maximum consecutive True values in one 1-D array."""
    best = current = 0
    for value in values:
        if np.isfinite(value) and bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _max_run(data: xr.DataArray, condition: xr.DataArray, dim: str) -> xr.DataArray:
    """Maximum consecutive condition-true days in each calendar year."""
    _validate_daily(data, dim)

    def run_length(block: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            _max_consecutive_1d,
            block,
            input_core_dims=[[dim]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[int],
        )

    return condition.groupby(f"{dim}.year").map(run_length)


def cwd(data: xr.DataArray, dim: str = "time", wet_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual maximum consecutive wet days (CWD)."""
    return _max_run(data, data >= wet_day_threshold, dim)


def cdd(data: xr.DataArray, dim: str = "time", dry_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual maximum consecutive dry days (CDD)."""
    return _max_run(data, data < dry_day_threshold, dim)
