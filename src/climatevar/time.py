"""Calendar and temporal-frequency diagnostics for climate datasets."""
from __future__ import annotations

import numpy as np
import xarray as xr


def calendar_name(data: xr.DataArray | xr.Dataset) -> str:
    """Return the calendar declared by CF metadata or inferred from the time index."""
    time = data["time"]
    declared = time.attrs.get("calendar") or time.encoding.get("calendar")
    if declared:
        return str(declared)
    index = time.to_index()
    if hasattr(index, "calendar"):
        return str(index.calendar)
    return "standard"


def time_step_seconds(data: xr.DataArray | xr.Dataset) -> xr.DataArray:
    """Return successive time-step durations in seconds.

    Works with pandas and cftime-compatible xarray time coordinates. The first
    element is NaN because no preceding timestamp exists.
    """
    t = data["time"]
    if t.size < 2:
        return xr.full_like(t, np.nan, dtype=float)
    values = t.values
    seconds = np.full(values.shape, np.nan, dtype=float)
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        seconds[i] = float(delta.total_seconds())
    return xr.DataArray(seconds, coords={"time": t}, dims="time", name="time_step_seconds")


def infer_frequency(data: xr.DataArray | xr.Dataset, *, tolerance: float = 0.02) -> str:
    """Infer a practical sampling frequency from median time spacing."""
    dt = time_step_seconds(data).values
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(dt):
        raise ValueError("At least two valid time coordinates are required.")
    median = float(np.median(dt))
    candidates = {
        "hourly": 3600.0,
        "3-hourly": 10800.0,
        "6-hourly": 21600.0,
        "daily": 86400.0,
        "monthly": 30.4375 * 86400.0,
    }
    for name, target in candidates.items():
        if abs(median - target) / target <= tolerance:
            return name
    return "irregular"


def validate_time(data: xr.DataArray | xr.Dataset, *, require_sorted: bool = True) -> None:
    """Validate time coordinates for duplicates and monotonic ordering."""
    t = data["time"]
    if t.ndim != 1:
        raise ValueError("The time coordinate must be one-dimensional.")
    if t.size < 1:
        raise ValueError("The time coordinate is empty.")
    if t.to_index().duplicated().any():
        raise ValueError("Duplicate timestamps detected.")
    if require_sorted and not t.to_index().is_monotonic_increasing:
        raise ValueError("Time coordinate is not monotonically increasing.")
