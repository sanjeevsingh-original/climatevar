"""Canonical data schema and metadata validation for climatevar."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import xarray as xr

@dataclass(frozen=True)
class ClimateSchema:
    """Names used by the normalized climatevar data model."""
    time: str = "time"
    latitude: str = "latitude"
    longitude: str = "longitude"
    precipitation: str = "precipitation"
    temperature: str = "temperature"
    surface_pressure: str = "surface_pressure"
    relative_humidity: str = "relative_humidity"
    specific_humidity: str = "specific_humidity"
    u_wind: str = "u_wind"
    v_wind: str = "v_wind"
    geopotential: str = "geopotential"

SCHEMA = ClimateSchema()

def validate_dataset(ds: xr.Dataset, *, required: tuple[str, ...] = (), require_cf_metadata: bool = False) -> xr.Dataset:
    """Validate normalized coordinates and optionally CF-style metadata."""
    missing = [name for name in required if name not in ds and name not in ds.coords and name not in ds.dims]
    if missing:
        raise ValueError(f"Missing required climatevar field(s): {', '.join(missing)}")
    for name in (SCHEMA.latitude, SCHEMA.longitude):
        if name in ds:
            attrs = ds[name].attrs
            if require_cf_metadata and not attrs.get("units"):
                raise ValueError(f"Coordinate {name!r} must have units for CF validation.")
    return ds

def describe_dataset(ds: xr.Dataset) -> dict[str, Any]:
    """Return a compact, serializable description useful for QA reports."""
    return {
        "dimensions": dict(ds.sizes),
        "data_variables": list(ds.data_vars),
        "coordinates": list(ds.coords),
        "calendar": ds.time.encoding.get("calendar") if "time" in ds.coords else None,
        "feature_type": ds.attrs.get("featureType"),
        "normalization": ds.attrs.get("climatevar:normalization", []),
    }
