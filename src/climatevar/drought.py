"""Drought-index foundations for climate time series."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _standardize(precip: xr.DataArray, dim: str) -> xr.DataArray:
    mean = precip.mean(dim=dim, skipna=True)
    std = precip.std(dim=dim, skipna=True, ddof=1)
    return (precip - mean) / std


def spi_like(precip: xr.DataArray, dim: str = "time", scale: int = 1) -> xr.DataArray:
    """Return a standardized precipitation anomaly as a transparent SPI-like index.

    This is intentionally *not* a fitted-distribution SPI implementation. It
    standardizes rolling precipitation totals and is useful as a diagnostic
    baseline. A formal SPI requires a probability distribution and zero-rain
    treatment appropriate to the application.
    """
    if scale < 1 or int(scale) != scale:
        raise ValueError("scale must be a positive integer.")
    if dim not in precip.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the data.")
    accumulated = precip.rolling({dim: int(scale)}, min_periods=int(scale)).sum()
    return _standardize(accumulated, dim=dim)
