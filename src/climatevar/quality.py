"""Data-quality and completeness utilities for climate time series."""

from __future__ import annotations

import xarray as xr


def valid_fraction(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Return the fraction of non-missing observations along ``dim``."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the data.")
    return data.notnull().mean(dim=dim)


def require_completeness(
    data: xr.DataArray, min_fraction: float = 0.9, dim: str = "time"
) -> xr.DataArray:
    """Mask locations whose valid-observation fraction is below a threshold."""
    if not 0 <= min_fraction <= 1:
        raise ValueError("min_fraction must be between 0 and 1.")
    fraction = valid_fraction(data, dim=dim)
    return data.where(fraction >= min_fraction)


def annual_valid_fraction(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Calculate valid-observation fraction separately for each year."""
    if dim not in data.dims or dim not in data.coords:
        raise ValueError(f"{dim!r} must be a dimension with a coordinate.")
    return data.notnull().groupby(f"{dim}.year").mean(dim=dim)
