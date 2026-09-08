"""ETCCDI-inspired daily precipitation indices.

All precipitation inputs are validated and normalized to millimetres (mm).
Daily rainfall indices never accept precipitation rates/fluxes because a rate
requires an explicit temporal integration before it can be used as a daily
amount.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _validate_daily(data: xr.DataArray, dim: str) -> xr.DataArray:
    """Validate a daily precipitation amount and return it in mm."""
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(f"{dim!r} must be a dimension with a coordinate.")
    units = str(data.attrs.get("units", "")).strip().lower().replace("°", "deg")
    if not units:
        raise ValueError("Precipitation indices require explicit precipitation units. Use units='mm' or normalize the dataset first.")
    if units in {"kg m-2 s-1", "kg/m2/s", "kg m-2 s^-1", "mm s-1", "mm/s", "mm day-1", "mm/day", "mm d-1", "mm h-1", "mm/hour", "mm hr-1", "mm min-1", "mm/min"}:
        raise ValueError("Precipitation rates/fluxes cannot be used directly by daily indices. Convert/integrate them to daily precipitation amounts in mm first.")
    factors = {"mm": 1.0, "millimeter": 1.0, "millimetre": 1.0, "m": 1000.0, "meter": 1000.0, "metre": 1000.0, "cm": 10.0}
    if units not in factors:
        raise ValueError(f"Unsupported precipitation amount unit {data.attrs.get('units')!r}; expected mm, cm, or m.")
    out = data * factors[units]
    out.attrs = dict(data.attrs)
    out.attrs["units"] = "mm"
    out.attrs["climatevar:precipitation_unit"] = "mm"
    return out


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
    wet = data.where(data >= threshold)
    return _with_mm_units(wet.groupby(f"{dim}.year").sum(dim=dim, skipna=True))


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


def _percentile_total(data: xr.DataArray, reference: xr.DataArray | None, percentile: float, dim: str) -> xr.DataArray:
    data = _validate_daily(data, dim)
    if not 0 <= percentile <= 1:
        raise ValueError("percentile must be between 0 and 1.")
    if reference is None:
        threshold = data.where(data >= 1.0).quantile(percentile, dim=dim, skipna=True)
    else:
        ref_units = str(reference.attrs.get("units", "")).strip().lower()
        if ref_units != "mm":
            raise ValueError("Percentile reference thresholds must have explicit units='mm'.")
        threshold = reference
    result = data.where(data > threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True)
    return _with_mm_units(result)


def r95p(data: xr.DataArray, reference: xr.DataArray | None = None, dim: str = "time") -> xr.DataArray:
    """Annual precipitation from days above a 95th-percentile threshold, in mm."""
    return _percentile_total(data, reference, 0.95, dim)


def r99p(data: xr.DataArray, reference: xr.DataArray | None = None, dim: str = "time") -> xr.DataArray:
    """Annual precipitation from days above a 99th-percentile threshold, in mm."""
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
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold")
    return _max_run(data, data >= threshold, dim)


def cdd(data: xr.DataArray, dim: str = "time", dry_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual maximum consecutive dry days (CDD)."""
    data = _validate_daily(data, dim)
    threshold = _validate_threshold(dry_day_threshold, "dry_day_threshold")
    return _max_run(data, data < threshold, dim)
