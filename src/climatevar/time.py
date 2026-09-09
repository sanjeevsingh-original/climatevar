"""Calendar-aware temporal diagnostics for climate datasets."""
from __future__ import annotations

import calendar as _calendar
import numpy as np
import xarray as xr

_CALENDAR_ALIASES = {"gregorian": "standard", "proleptic_gregorian": "proleptic_gregorian", "julian": "julian", "standard": "standard", "noleap": "noleap", "365_day": "noleap", "all_leap": "all_leap", "366_day": "all_leap", "360_day": "360_day"}

def calendar_name(data):
    time = data["time"]
    declared = time.attrs.get("calendar") or time.encoding.get("calendar")
    if declared: return _CALENDAR_ALIASES.get(str(declared).lower(), str(declared).lower())
    index = time.to_index()
    if hasattr(index, "calendar"): return _CALENDAR_ALIASES.get(str(index.calendar).lower(), str(index.calendar).lower())
    return "standard"

def _seconds_between(a, b):
    delta = a - b
    if hasattr(delta, "total_seconds"): return float(delta.total_seconds())
    return float(delta / np.timedelta64(1, "s"))

def _time_bounds(data):
    time = data["time"]
    candidates = []
    bounds_name = time.attrs.get("bounds")
    if bounds_name: candidates.append(str(bounds_name))
    candidates.extend(["time_bnds", "time_bounds"])
    variables = data.variables if isinstance(data, xr.Dataset) else data.coords
    for name in candidates:
        if name in variables:
            bounds = data[name]
            if bounds.ndim == 2 and bounds.sizes.get("time") == time.size and bounds.shape[1] == 2:
                return bounds
    return None

def time_step_seconds(data):
    t = data["time"]
    bounds = _time_bounds(data)
    if bounds is not None:
        values = np.asarray([_seconds_between(row[1], row[0]) for row in bounds.values], dtype=float)
        return xr.DataArray(values, coords={"time": t}, dims="time", name="time_step_seconds")
    if t.size < 2: return xr.full_like(t, np.nan, dtype=float).rename("time_step_seconds")
    values = t.values
    seconds = np.full(values.shape, np.nan, dtype=float)
    for i in range(1, len(values)): seconds[i] = _seconds_between(values[i], values[i - 1])
    if len(seconds) > 1: seconds[0] = seconds[1]
    return xr.DataArray(seconds, coords={"time": t}, dims="time", name="time_step_seconds")

def infer_frequency(data, *, tolerance=0.02):
    dt = time_step_seconds(data).values
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt): raise ValueError("At least two valid time coordinates or valid time bounds are required.")
    median = float(np.median(dt))
    candidates = {"hourly": 3600.0, "3-hourly": 10800.0, "6-hourly": 21600.0, "daily": 86400.0, "monthly": 30.4375 * 86400.0}
    for name, target in candidates.items():
        if abs(median - target) / target <= tolerance: return name
    return "irregular"

def expected_days_in_year(year, calendar="standard"):
    cal = _CALENDAR_ALIASES.get(str(calendar).lower(), str(calendar).lower())
    if cal == "360_day": return 360
    if cal == "noleap": return 365
    if cal == "all_leap": return 366
    if cal in {"standard", "gregorian", "proleptic_gregorian", "julian"}:
        if cal == "julian": return 366 if year % 4 == 0 else 365
        return 366 if _calendar.isleap(year) else 365
    raise ValueError(f"Unsupported CF calendar {calendar!r}.")

def expected_days(data, dim="time"):
    if dim not in data.coords: raise ValueError(f"{dim!r} must be a coordinate.")
    years = data[dim].dt.year
    unique = np.unique(np.asarray(years.values, dtype=int))
    values = [expected_days_in_year(int(year), calendar_name(data)) for year in unique]
    return xr.DataArray(values, coords={"year": unique}, dims="year", name="expected_days")

def year_complete_mask(data, dim="time", *, min_valid_fraction=0.9):
    if not 0 < min_valid_fraction <= 1: raise ValueError("min_valid_fraction must be greater than 0 and at most 1.")
    valid = data.notnull().groupby(f"{dim}.year").sum(dim=dim)
    expected = expected_days(data, dim).reindex(year=valid["year"])
    return (valid / expected) >= min_valid_fraction

def season_year(data, dim="time", *, season="DJF"):
    season = season.upper()
    if season not in {"DJF", "MAM", "JJA", "SON"}: raise ValueError("season must be one of DJF, MAM, JJA, or SON.")
    years = data[dim].dt.year.astype(int)
    if season == "DJF": return xr.where(data[dim].dt.month == 12, years + 1, years).rename("season_year")
    return years.rename("season_year")

def drop_leap_day(data, dim="time"):
    return data.where(~((data[dim].dt.month == 2) & (data[dim].dt.day == 29)), drop=True)

def validate_time(data, *, require_sorted=True):
    t = data["time"]
    if t.ndim != 1: raise ValueError("The time coordinate must be one-dimensional.")
    if t.size < 1: raise ValueError("The time coordinate is empty.")
    if t.to_index().duplicated().any(): raise ValueError("Duplicate timestamps detected.")
    if require_sorted and not t.to_index().is_monotonic_increasing: raise ValueError("Time coordinate is not monotonically increasing.")
