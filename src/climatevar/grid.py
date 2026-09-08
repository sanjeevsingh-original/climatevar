"""Grid normalization and conservative regridding entry points."""
from __future__ import annotations

import numpy as np
import xarray as xr


def normalize_longitude(ds: xr.Dataset, *, center: float = 0.0) -> xr.Dataset:
    """Normalize 1-D longitude to either [-180, 180) or [0, 360)."""
    if "longitude" not in ds.coords:
        raise KeyError("Dataset must contain a canonical 'longitude' coordinate.")
    lon = ds["longitude"]
    if lon.ndim != 1:
        raise ValueError("normalize_longitude currently requires a 1-D longitude coordinate.")
    if center == 0.0:
        new_lon = ((lon + 180.0) % 360.0) - 180.0
    elif center == 180.0:
        new_lon = lon % 360.0
    else:
        raise ValueError("center must be 0.0 or 180.0.")
    order = np.argsort(new_lon.values)
    return ds.assign_coords(longitude=new_lon).isel(longitude=order)


def regrid(
    data: xr.DataArray | xr.Dataset,
    target: xr.Dataset | xr.DataArray,
    *,
    method: str = "linear",
) -> xr.DataArray | xr.Dataset:
    """Regrid data to a target grid.

    Rectilinear 1-D latitude/longitude grids use xarray interpolation. Curvilinear
    grids such as native WRF require the optional ``xesmf`` dependency and use
    xESMF. No regridding is silently performed by normalization.
    """
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'.")
    src = data
    target_lat = target["latitude"]
    target_lon = target["longitude"]
    src_lat = src["latitude"]
    src_lon = src["longitude"]

    if src_lat.ndim == 1 and src_lon.ndim == 1 and target_lat.ndim == 1 and target_lon.ndim == 1:
        return src.interp(latitude=target_lat, longitude=target_lon, method=method)

    try:
        import xesmf as xe
    except ImportError as exc:
        raise ImportError(
            "Curvilinear regridding requires the optional 'xesmf' dependency. "
            "Install climatevar[regrid]."
        ) from exc

    src_grid = xr.Dataset({"lat": src_lat, "lon": src_lon})
    target_grid = xr.Dataset({"lat": target_lat, "lon": target_lon})
    engine = xe.Regridder(src_grid, target_grid, method=method, periodic=False, reuse_weights=False)
    return engine(src)
