"""Temporal normalization and precipitation accumulation helpers."""
from __future__ import annotations
import numpy as np
import xarray as xr

def sort_and_validate_time(ds: xr.Dataset) -> xr.Dataset:
    """Sort by time and reject duplicate timestamps."""
    if "time" not in ds.coords: raise ValueError("Dataset requires a canonical 'time' coordinate.")
    out = ds.sortby("time")
    values = out.time.values
    if len(values) > 1 and np.any(values[1:] == values[:-1]): raise ValueError("Duplicate timestamps found in time coordinate.")
    return out

def resample_precipitation(data: xr.DataArray, *, frequency: str = "1D", how: str = "sum") -> xr.DataArray:
    """Aggregate precipitation to a target frequency."""
    if "time" not in data.dims: raise ValueError("Precipitation data must have a time dimension.")
    if how == "sum": return data.resample(time=frequency).sum(skipna=True, min_count=1)
    if how == "max": return data.resample(time=frequency).max(skipna=True)
    if how == "mean": return data.resample(time=frequency).mean(skipna=True)
    raise ValueError("how must be 'sum', 'max', or 'mean'.")

def accumulated_to_increment(data: xr.DataArray, *, dim: str = "time", reset_tolerance: float = 0.0) -> xr.DataArray:
    """Convert a monotonically accumulated field to increments; negative changes are resets."""
    if dim not in data.dims: raise ValueError(f"Missing dimension {dim!r}.")
    diff = data.diff(dim)
    previous = data.isel({dim: slice(0, -1)})
    increment = diff.where(diff >= -reset_tolerance, other=diff + previous)
    return xr.concat([data.isel({dim: 0}), increment], dim=dim).assign_coords({dim: data[dim]})
