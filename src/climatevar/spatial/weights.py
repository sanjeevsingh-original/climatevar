"""Latitude-based area weighting for regular lat/lon grids."""

from __future__ import annotations

import numpy as np
import xarray as xr


def cosine_latitude_weights(latitude: xr.DataArray) -> xr.DataArray:
    """Return normalized cosine(latitude) weights."""
    if latitude.ndim != 1:
        raise ValueError("latitude must be a one-dimensional coordinate.")
    weights = np.cos(np.deg2rad(latitude))
    if np.any(weights < 0):
        raise ValueError("latitude values must lie between -90 and 90 degrees.")
    return weights / weights.mean()


def area_weighted_mean(
    data: xr.DataArray,
    lat_dim: str = "lat",
    spatial_dims: tuple[str, ...] | None = None,
) -> xr.DataArray:
    """Calculate an area-weighted mean using cosine latitude weights.

    This method is appropriate for regular geographic latitude/longitude
    grids. For curvilinear grids or exact cell areas, provide true cell-area
    weights instead of this approximation.
    """
    if lat_dim not in data.dims or lat_dim not in data.coords:
        raise ValueError(f"{lat_dim!r} must be a dimension with a coordinate.")
    if spatial_dims is None:
        spatial_dims = tuple(d for d in data.dims if d in {lat_dim, "lon", "longitude"})
    if lat_dim not in spatial_dims:
        raise ValueError("lat_dim must be included in spatial_dims.")
    weights = cosine_latitude_weights(data[lat_dim])
    return data.weighted(weights).mean(dim=spatial_dims, skipna=True)
