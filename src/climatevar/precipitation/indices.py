"""ETCCDI-inspired precipitation indices with implicit time-aware units.

The public functions accept common rainfall products directly. Sub-daily
precipitation is aggregated to daily totals internally, while index outputs
use the conventional units of the index (mm or days/counts).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .normalize import daily_amount


def _validate_daily(data: xr.DataArray, dim: str) -> xr.DataArray:
    return daily_amount(data, dim)


def _validate_threshold(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative value in mm.")
    return value


def _complete_year_mask(data: xr.DataArray, dim: str, min_valid_fraction: float) -> xr.DataArray:
    if not 0 < min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be greater than 0 and at most 1.")
    return data.notnull().groupby(f"{dim}.year").mean(dim=dim) >= min_valid_fraction


def _apply_completeness(result: xr.DataArray, complete: xr.DataArray) -> xr.DataArray:
    year_dim = next((d for d in complete.dims if d not in result.dims), None)
    if year_dim is None:
        return result
    return result.where(complete)


def _with_mm_units(result: xr.DataArray) -> xr.DataArray:
    result.attrs = dict(result.attrs)
    result.attrs["units"] = "mm"
    result.attrs["climatevar:precipitation_unit"] = "mm"
    return result


def rx1day(data: xr.DataArray, dim: str = "time", min_valid_fraction: float = 0.9) -> xr.DataArray:
    """Annual maximum 1-day precipitation (Rx1day), in mm.

    ``min_valid_fraction`` prevents incomplete years from being interpreted as
    valid annual extremes.
    """
    data = _validate_daily(data, dim)
    complete = _complete_year_mask(data, dim, min_valid_fraction)
    result = data.groupby(f"{dim}.year").max(dim=dim, skipna=True)
    result = result.where(complete)
    return _with_mm_units(result)


def rx5day(data: xr.DataArray, dim: str = "time", min_valid_fraction: float = 0.9) -> xr.DataArray:
    """Annual maximum consecutive 5-day precipitation (Rx5day), in mm."""
    data = _validate_daily(data, dim)
    complete = _complete_year_mask(data, dim, min_valid_fraction)
    rolling = data.rolling({dim: 5}, min_periods=5).sum()
    result = rolling.groupby(f"{dim}.year").max(dim=dim, skipna=True)
    return _with_mm_units(result.where(complete))


def prcptot(
    data: xr.DataArray,
    dim: str = "time",
    wet_day_threshold: float = 1.0,
    min_valid_fraction: float = 0.9,
) -> xr.DataArray:
    """Annual wet-day precipitation total (PRCPTOT), in mm."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold")
    complete = _complete_year_mask(data, dim, min_valid_fraction)
    result = data.where(data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)
    return _with_mm_units(result.where(complete))


def r10mm(data: xr.DataArray, dim: str = "time", threshold: float = 10.0, min_valid_fraction: float = 0.9) -> xr.DataArray:
    """Annual count of days with precipitation >= threshold mm."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(threshold, "threshold")
    complete = _complete_year_mask(data, dim, min_valid_fraction)
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True).where(complete)


def r20mm(data: xr.DataArray, dim: str = "time", threshold: float = 20.0, min_valid_fraction: float = 0.9) -> xr.DataArray:
    """Annual count of days with precipitation >= threshold mm."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(threshold, "threshold")
    complete = _complete_year_mask(data, dim, min_valid_fraction)
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True).where(complete)


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
    baseline_start: str | None,
    baseline_end: str | None,
    wet_day_threshold: float,
    min_valid_fraction: float,
) -> xr.DataArray:
    data = _validate_daily(data, dim)
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1.")
    wet_day_threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold")
    complete = _complete_year_mask(data, dim, min_valid_fraction)

    if reference is not None:
        threshold = _reference_mm(reference)
    else:
        baseline = data
        if baseline_start is not None:
            baseline = baseline.sel({dim: slice(baseline_start, baseline_end)})
        elif baseline_end is not None:
            baseline = baseline.sel({dim: slice(None, baseline_end)})
        wet = baseline.where(baseline >= wet_day_threshold)
        threshold = wet.quantile(percentile, dim=dim, skipna=True)

    result = data.where(data > threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)
    return _with_mm_units(result.where(complete))


def r95p(
    data: xr.DataArray,
    reference: xr.DataArray | float | None = None,
    dim: str = "time",
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    wet_day_threshold: float = 1.0,
    min_valid_fraction: float = 0.9,
) -> xr.DataArray:
    """Annual R95p precipitation above the 95th-percentile wet-day threshold."""
    return _percentile_total(data, reference, 0.95, dim, baseline_start, baseline_end, wet_day_threshold, min_valid_fraction)


def r99p(
    data: xr.DataArray,
    reference: xr.DataArray | float | None = None,
    dim: str = "time",
    baseline_start: str | None = None,
    baseline_end: str | None = None,
    wet_day_threshold: float = 1.0,
    min_valid_fraction: float = 0.9,
) -> xr.DataArray:
    """Annual R99p precipitation above the 99th-percentile wet-day threshold."""
    return _percentile_total(data, reference, 0.99, dim, baseline_start, baseline_end, wet_day_threshold, min_valid_fraction)


def _max_consecutive_1d(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if np.isfinite(value) and bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _max_run(condition: xr.DataArray, dim: str, complete: xr.DataArray) -> xr.DataArray:
    result = condition.groupby(f"{dim}.year").map(
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
    return result.where(complete)


def cwd(data: xr.DataArray, dim: str = "time", wet_day_threshold: float = 1.0, min_valid_fraction: float = 0.9) -> xr.DataArray:
    """Annual maximum consecutive wet days (CWD)."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold")
    return _max_run(data >= threshold, dim, _complete_year_mask(data, dim, min_valid_fraction))


def cdd(data: xr.DataArray, dim: str = "time", dry_day_threshold: float = 1.0, min_valid_fraction: float = 0.9) -> xr.DataArray:
    """Annual maximum consecutive dry days (CDD)."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(dry_day_threshold, "dry_day_threshold")
    return _max_run(data < threshold, dim, _complete_year_mask(data, dim, min_valid_fraction))
