"""ETCCDI-inspired daily precipitation indices.

Precipitation inputs are normalized internally to daily amounts in millimetres
(mm). Common amounts and precipitation rates are accepted automatically, so
users do not need to manually convert datasets before calculating indices.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _seconds_per_step(data: xr.DataArray, dim: str) -> xr.DataArray:
    values = data[dim].values
    if len(values) <= 1:
        return xr.DataArray(86400.0)
    seconds = np.empty(len(values), dtype=float)
    for i in range(len(values) - 1):
        delta = values[i + 1] - values[i]
        try:
            seconds[i] = float(delta / np.timedelta64(1, "s"))
        except (TypeError, ValueError):
            seconds[i] = float(delta.total_seconds())
    seconds[-1] = seconds[-2]
    return xr.DataArray(seconds, coords={dim: data[dim]}, dims=dim)


def _validate_daily(data: xr.DataArray, dim: str) -> xr.DataArray:
    """Normalize precipitation to daily amount in mm, inferring common rates."""
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(f"{dim!r} must be a dimension with a coordinate.")
    units = str(data.attrs.get("units", "")).strip().lower().replace("°", "deg").replace("*", "")
    if not units:
        # Maintain compatibility with synthetic/legacy rainfall arrays.
        units = "mm"
    out = data.copy()
    attrs = dict(data.attrs)

    amount_factor = {"mm": 1.0, "millimeter": 1.0, "millimetre": 1.0,
                     "cm": 10.0, "m": 1000.0, "meter": 1000.0, "metre": 1000.0}
    if units in amount_factor:
        out = data * amount_factor[units]
    elif units in {"mm/day", "mm day-1", "mm d-1", "mm/day-1"}:
        out = data * (_seconds_per_step(data, dim) / 86400.0)
    elif units in {"cm/day", "cm day-1", "cm d-1"}:
        out = data * 10.0 * (_seconds_per_step(data, dim) / 86400.0)
    elif units in {"m/day", "m day-1", "m d-1"}:
        out = data * 1000.0 * (_seconds_per_step(data, dim) / 86400.0)
    elif units in {"mm/hour", "mm hr-1", "mm h-1"}:
        out = data * 24.0
    elif units in {"mm/min", "mm min-1"}:
        out = data * 1440.0
    elif units in {"mm/s", "mm s-1"}:
        out = data * 86400.0
    elif units in {"kg m-2 s-1", "kg/m2/s", "kg m-2 s^-1"}:
        out = data * 86400.0
    elif units in {"kg m-2 day-1", "kg m-2 d-1", "kg/m2/day"}:
        out = data
    elif units in {"kg m-2 h-1", "kg/m2/hour"}:
        out = data * 24.0
    elif units in {"kg m-2 min-1", "kg/m2/min"}:
        out = data * 1440.0
    else:
        raise ValueError(f"Unsupported precipitation units {data.attrs.get('units')!r}.")

    out.attrs = attrs
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


def _percentile_total(data: xr.DataArray, reference: xr.DataArray | float | None, percentile: float, dim: str) -> xr.DataArray:
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


def _max_run(data: xr.DataArray, condition: xr.DataArray, dim: str) -> xr.DataArray:
    _validate_daily(data, dim)
    return condition.groupby(f"{dim}.year").map(
        lambda block: xr.apply_ufunc(_max_consecutive_1d, block,
                                     input_core_dims=[[dim]], output_core_dims=[[]],
                                     vectorize=True, dask="parallelized", output_dtypes=[int])
    )


def cwd(data: xr.DataArray, dim: str = "time", wet_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual maximum consecutive wet days (CWD)."""
    data = _validate_daily(data, dim)
    return _max_run(data, data >= _validate_threshold(wet_day_threshold, "wet_day_threshold"), dim)


def cdd(data: xr.DataArray, dim: str = "time", dry_day_threshold: float = 1.0) -> xr.DataArray:
    """Annual maximum consecutive dry days (CDD)."""
    data = _validate_daily(data, dim)
    return _max_run(data, data < _validate_threshold(dry_day_threshold, "dry_day_threshold"), dim)
