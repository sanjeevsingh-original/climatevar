"""ETCCDI-inspired precipitation indices with implicit time-aware units."""

from __future__ import annotations

import numpy as np
import xarray as xr

from .normalize import daily_amount


def _validate_daily(data: xr.DataArray, dim: str) -> xr.DataArray:
    """Prepare precipitation as daily interval totals in mm."""
    return daily_amount(data, dim)


def _validate_threshold(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative value in mm.")
    return value


def _with_mm_units(result: xr.DataArray) -> xr.DataArray:
    result.attrs = dict(result.attrs)
    result.attrs["units"] = "mm"
    result.attrs["climatevar:precipitation_unit"] = "mm"
    return result


def rx1day(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Annual maximum 1-day precipitation (Rx1day), in mm."""
    data = _validate_daily(data, dim)
    return _with_mm_units(data.groupby(f"{dim}.year").max(dim=dim, skipna=True))


def rx5day(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Annual maximum consecutive 5-day precipitation (Rx5day), in mm."""
    data = _validate_daily(data, dim)
    rolling = data.rolling({dim: 5}, min_periods=5).sum()
    return _with_mm_units(rolling.groupby(f"{dim}.year").max(dim=dim, skipna=True))


def prcptot(data: xr.DataArray, dim: str = "time", wet_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual precipitation total on wet days, in mm."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold")
    return _with_mm_units(data.where(data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True))


def r10mm(data: xr.DataArray, dim: str = "time", threshold: float = 10.0) -> xr.DataArray:
    """Annual count of days with precipitation >= threshold mm."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(threshold, "threshold")
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def r20mm(data: xr.DataArray, dim: str = "time", threshold: float = 20.0) -> xr.DataArray:
    """Annual count of days with precipitation >= threshold mm."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(threshold, "threshold")
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)


def _reference_mm(reference: xr.DataArray | float) -> xr.DataArray:
    if not isinstance(reference, xr.DataArray):
        return xr.DataArray(float(reference), attrs={"units": "mm"})
    units = str(reference.attrs.get("units", "mm")).strip().lower()
    factors = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    if units not in factors:
        raise ValueError("Percentile reference must use mm, cm, or m.")
    out = reference * factors[units]
    out.attrs = dict(reference.attrs)
    out.attrs["units"] = "mm"
    return out


def _percentile_total(
    data: xr.DataArray,
    reference: xr.DataArray | float | None,
    percentile: float,
    dim: str,
) -> xr.DataArray:
    data = _validate_daily(data, dim)
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1.")
    threshold = data.where(data >= 1.0).quantile(percentile, dim=dim, skipna=True) if reference is None else _reference_mm(reference)
    return _with_mm_units(data.where(data > threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True))


def r95p(data: xr.DataArray, reference: xr.DataArray | float | None = None, dim: str = "time") -> xr.DataArray:
    """Annual precipitation from days above a 95th-percentile threshold, in mm."""
    return _percentile_total(data, reference, 0.95, dim)


def r99p(data: xr.DataArray, reference: xr.DataArray | float | None = None, dim: str = "time") -> xr.DataArray:
    """Annual precipitation from days above a 99th-percentile threshold, in mm."""
    return _percentile_total(data, reference, 0.99, dim)


def _max_consecutive_1d(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if np.isfinite(value) and bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _max_run(condition: xr.DataArray, dim: str) -> xr.DataArray:
    return condition.groupby(f"{dim}.year").map(
        lambda block: xr.apply_ufunc(
            _max_consecutive_1d,
            block,
            input_core_dims=[[dim]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[int],
        )
    )


def cwd(data: xr.DataArray, dim: str = "time", wet_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual maximum consecutive wet days (CWD)."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold")
    return _max_run(data >= threshold, dim)


def cdd(data: xr.DataArray, dim: str = "time", dry_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual maximum consecutive dry days (CDD)."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(dry_day_threshold, "dry_day_threshold")
    return _max_run(data < threshold, dim)
