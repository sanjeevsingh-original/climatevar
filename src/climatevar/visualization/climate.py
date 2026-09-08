"""Simple plotting helpers with explicit, dependency-light behavior."""

from __future__ import annotations

import xarray as xr


def plot_timeseries(data: xr.DataArray, **kwargs):
    """Plot an xarray time series and return the matplotlib Axes."""
    if "time" not in data.dims:
        raise ValueError("plot_timeseries expects a 'time' dimension.")
    ax = data.plot(**kwargs)
    return ax


def plot_map(data: xr.DataArray, x: str = "lon", y: str = "lat", **kwargs):
    """Plot a 2-D geographic field and return the matplotlib artist."""
    if x not in data.dims and x not in data.coords:
        raise ValueError(f"Longitude coordinate {x!r} was not found.")
    if y not in data.dims and y not in data.coords:
        raise ValueError(f"Latitude coordinate {y!r} was not found.")
    return data.plot(x=x, y=y, **kwargs)
