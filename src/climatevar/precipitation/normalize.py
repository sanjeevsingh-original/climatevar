"""Implicit precipitation normalization for common climate datasets."""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..time import infer_frequency, time_step_seconds

_AMOUNT_FACTORS = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimetre": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "metre": 1000.0,
}
_RATE_TO_MM_PER_SECOND = {
    "mm/s": 1.0,
    "mm s-1": 1.0,
    "mm s^-1": 1.0,
    "mm/min": 1.0 / 60.0,
    "mm min-1": 1.0 / 60.0,
    "mm/hour": 1.0 / 3600.0,
    "mm hr-1": 1.0 / 3600.0,
    "mm h-1": 1.0 / 3600.0,
    "mm/day": 1.0 / 86400.0,
    "mm day-1": 1.0 / 86400.0,
    "mm d-1": 1.0 / 86400.0,
    "kg m-2 s-1": 1.0,
    "kg/m2/s": 1.0,
    "kg m-2 s^-1": 1.0,
    "kg m-2 min-1": 1.0 / 60.0,
    "kg/m2/min": 1.0 / 60.0,
    "kg m-2 h-1": 1.0 / 3600.0,
    "kg/m2/hour": 1.0 / 3600.0,
    "kg m-2 day-1": 1.0 / 86400.0,
    "kg m-2 d-1": 1.0 / 86400.0,
    "kg/m2/day": 1.0 / 86400.0,
}


def _units(data: xr.DataArray) -> str:
    return str(data.attrs.get("units", "mm")).strip().lower().replace("°", "deg").replace("*", "")


def _step_seconds(data: xr.DataArray, dim: str) -> xr.DataArray:
    values = time_step_seconds(data.rename({dim: "time"}) if dim != "time" else data).values
    finite = values[np.isfinite(values) & (values > 0)]
    if not finite.size:
        raise ValueError("At least two valid time coordinates are required for rate precipitation.")
    # time_step_seconds returns NaN for the first sample; use the next interval
    # for it and the final observed interval for the last sample.
    if np.isnan(values[0]):
        values[0] = np.median(finite)
    for i in range(1, len(values)):
        if not np.isfinite(values[i]) or values[i] <= 0:
            values[i] = np.median(finite)
    return xr.DataArray(values, coords={dim: data[dim]}, dims=dim)


def interval_amount_mm(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Return precipitation represented by each sample as an interval amount in mm.

    Amount data (mm, cm, m) are treated as interval accumulations. Rate/flux
    data are integrated over the observed sampling interval. Missing units are
    treated as mm for compatibility with simple research arrays.
    """
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(f"{dim!r} must be a dimension with a coordinate.")
    unit = _units(data)
    if unit in _AMOUNT_FACTORS:
        out = data * _AMOUNT_FACTORS[unit]
    elif unit in _RATE_TO_MM_PER_SECOND:
        out = data * _RATE_TO_MM_PER_SECOND[unit] * _step_seconds(data, dim)
    else:
        raise ValueError(f"Unsupported precipitation units {data.attrs.get('units')!r}.")
    out.attrs = dict(data.attrs)
    out.attrs["units"] = "mm"
    out.attrs["climatevar:quantity"] = "precipitation_amount"
    out.attrs["climatevar:source_units"] = data.attrs.get("units", "mm")
    return out


def normalize_precipitation(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Normalize precipitation to a time-resolution-aware rate/amount.

    Hourly/subdaily observations become a rate such as ``mm/hr`` or
    ``mm/3hr``; daily data become ``mm/day``. Monthly or irregular period
    accumulations remain ``mm``. The conversion is implicit for common
    precipitation amount and flux units.
    """
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(f"{dim!r} must be a dimension with a coordinate.")
    unit = _units(data)
    amount = interval_amount_mm(data, dim)
    try:
        frequency = infer_frequency(data.rename({dim: "time"}) if dim != "time" else data)
    except ValueError:
        frequency = "irregular"

    if frequency == "hourly":
        denominator = "hr"
        seconds = 3600.0
    elif frequency == "3-hourly":
        denominator = "3hr"
        seconds = 10800.0
    elif frequency == "6-hourly":
        denominator = "6hr"
        seconds = 21600.0
    elif frequency == "daily":
        denominator = "day"
        seconds = 86400.0
    else:
        out = amount
        out.attrs["climatevar:frequency"] = frequency
        out.attrs["climatevar:precipitation_representation"] = "period_amount"
        return out

    out = amount / seconds
    # Express the rate in the natural interval denominator rather than SI flux.
    out = out * seconds
    out.attrs = dict(data.attrs)
    out.attrs["units"] = f"mm/{denominator}"
    out.attrs["climatevar:quantity"] = "precipitation_rate"
    out.attrs["climatevar:frequency"] = frequency
    out.attrs["climatevar:source_units"] = data.attrs.get("units", "mm")
    out.attrs["climatevar:precipitation_representation"] = "interval_rate"
    return out


def daily_amount(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Aggregate common hourly/subdaily precipitation to daily totals in mm."""
    amount = interval_amount_mm(data, dim)
    try:
        frequency = infer_frequency(data.rename({dim: "time"}) if dim != "time" else data)
    except ValueError:
        frequency = "irregular"
    if frequency == "daily":
        amount.attrs["units"] = "mm"
        amount.attrs["climatevar:frequency"] = "daily"
        return amount
    if frequency in {"hourly", "3-hourly", "6-hourly", "irregular"}:
        result = amount.resample({dim: "1D"}).sum(skipna=True)
        result.attrs = dict(data.attrs)
        result.attrs["units"] = "mm"
        result.attrs["climatevar:quantity"] = "precipitation_amount"
        result.attrs["climatevar:frequency"] = "daily"
        result.attrs["climatevar:source_units"] = data.attrs.get("units", "mm")
        return result
    return amount
