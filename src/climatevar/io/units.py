"""Explicit unit conversions for common climate variables."""
from __future__ import annotations
import numpy as np
import xarray as xr

def precipitation_to_mm(data: xr.DataArray, *, from_units: str | None = None) -> xr.DataArray:
    """Convert precipitation depth/rate to mm while preserving provenance."""
    unit = (from_units or data.attrs.get("units") or "").lower().replace(" ", "")
    out = data.copy()
    if unit == "mm": return out.assign_attrs({**out.attrs, "units": "mm"})
    if unit in {"mm/day", "mm/d"}: return out.assign_attrs({**out.attrs, "units": "mm/day"})
    if unit == "m": return (out * 1000.0).assign_attrs({**out.attrs, "units": "mm"})
    if unit in {"m/s", "msec-1", "ms-1", "kgm-2s-1", "kgm^-2s^-1", "kgm-2sec-1"}:
        if "time" not in out.coords: raise ValueError("A time coordinate is required to convert a rate to depth.")
        seconds = _time_step_seconds(out["time"])
        factor = 1000.0 if unit in {"m/s", "msec-1", "ms-1"} else 1.0
        return (out * seconds * factor).assign_attrs({**out.attrs, "units": "mm"})
    raise ValueError(f"Unsupported precipitation units: {from_units or data.attrs.get('units')!r}")

def _time_step_seconds(time: xr.DataArray) -> xr.DataArray:
    if time.size < 2: raise ValueError("At least two time points are required for rate-to-depth conversion.")
    delta = np.diff(time.values).astype("timedelta64[s]").astype(float)
    if np.any(delta <= 0): raise ValueError("Time coordinate must be strictly increasing.")
    return xr.DataArray(np.r_[delta, delta[-1]], dims=time.dims, coords=time.coords)

def temperature_to_celsius(data: xr.DataArray) -> xr.DataArray:
    """Convert Kelvin temperatures to Celsius when units explicitly identify it."""
    unit = str(data.attrs.get("units", "")).lower().replace(" ", "")
    if unit in {"k", "kelvin"}: return (data - 273.15).assign_attrs({**data.attrs, "units": "degC"})
    if unit in {"degc", "c", "celsius", "degrees_celsius"}: return data.assign_attrs({**data.attrs, "units": "degC"})
    raise ValueError("Temperature units must explicitly identify Kelvin or Celsius.")
