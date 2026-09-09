"""Grid conventions and safe spatial normalization."""
from __future__ import annotations
import xarray as xr

def normalize_longitude(data: xr.Dataset | xr.DataArray, *, target: str = "-180_180"):
    """Normalize longitude convention to -180..180 or 0..360 and sort it.

    For the signed convention, +180 is retained as +180 rather than being
    wrapped to -180. This avoids surprising endpoint changes for grids that
    explicitly use 180E.
    """
    if "longitude" not in data.coords: raise ValueError("Canonical 'longitude' coordinate is required.")
    lon = data["longitude"]
    if target == "-180_180":
        new = ((lon + 180) % 360) - 180
        new = xr.where((new == -180) & (lon > 0), 180, new)
    elif target == "0_360": new = lon % 360
    else: raise ValueError("target must be '-180_180' or '0_360'.")
    return data.assign_coords(longitude=new).sortby("longitude")

def normalize_latitude(data: xr.Dataset | xr.DataArray, *, ascending: bool = True):
    """Sort latitude ascending or descending without changing values."""
    if "latitude" not in data.coords: raise ValueError("Canonical 'latitude' coordinate is required.")
    return data.sortby("latitude", ascending=ascending)
