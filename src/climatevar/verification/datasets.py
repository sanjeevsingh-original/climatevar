"""Dataset-family adapters for comparative precipitation verification.

Adapters intentionally separate *dataset identification* from scientific
normalization. Variable names may be overridden because provider files vary by
version and distribution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import xarray as xr

from climatevar.precipitation.normalize import normalize_precipitation
from climatevar.precipitation.wrf import wrf_total_precipitation


PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "era5": {"precipitation": ("tp", "total_precipitation", "precip")},
    "era5-land": {"precipitation": ("tp", "total_precipitation", "precip")},
    "imerg": {"precipitation": ("precipitationCal", "precipitation", "precip")},
    "gpm": {"precipitation": ("precipitationCal", "precipitation", "precip")},
    "imdaa": {"precipitation": ("pr", "precipitation", "rainfall", "precip")},
    "imd": {"precipitation": ("rainfall", "precipitation", "pr", "precip")},
    "cmip6": {"precipitation": ("pr", "precipitation", "prec")},
    "wrf": {"precipitation": ("RAINNC", "RAINC", "RAINSH")},
}


def _family(name: str) -> str:
    key = name.lower().strip().replace("_", "-")
    aliases = {"era5land": "era5-land", "gpm-imerg": "imerg"}
    return aliases.get(key, key)


def _open(source, **kwargs):
    if isinstance(source, xr.Dataset):
        return source
    path = Path(source)
    if path.suffix.lower() in {".zarr"} or path.is_dir():
        return xr.open_zarr(source, **kwargs)
    return xr.open_dataset(source, **kwargs)


def _pick_variable(ds: xr.Dataset, family: str, variable: str | None) -> str:
    if variable:
        if variable not in ds:
            raise KeyError(f"Variable {variable!r} is not present in the dataset.")
        return variable
    candidates = PRESETS.get(family, {}).get("precipitation", ())
    for candidate in candidates:
        if candidate in ds:
            return candidate
    raise KeyError(
        f"Could not identify precipitation variable for {family!r}. "
        f"Pass variable=... explicitly. Available variables: {list(ds.data_vars)}"
    )


def _rename_spatial(data: xr.DataArray) -> xr.DataArray:
    renames = {}
    if "latitude" in data.dims and "lat" not in data.dims:
        renames["latitude"] = "lat"
    if "longitude" in data.dims and "lon" not in data.dims:
        renames["longitude"] = "lon"
    if "XLAT" in data.dims and "lat" not in data.dims:
        renames["XLAT"] = "lat"
    if "XLONG" in data.dims and "lon" not in data.dims:
        renames["XLONG"] = "lon"
    if renames:
        data = data.rename(renames)
    return data


def load_precipitation(
    source,
    *,
    family: str,
    variable: str | None = None,
    dim: str = "time",
    normalize: bool = True,
    chunks=None,
    open_kwargs: Mapping | None = None,
) -> xr.DataArray:
    """Load one precipitation product using a dataset-family preset.

    The returned field is standardized to ``time, lat, lon`` naming where
    possible. ``normalize=True`` converts native precipitation rates/amounts
    to climatevar's analysis-ready interval representation.
    """
    family = _family(family)
    ds = _open(source, chunks=chunks, **(dict(open_kwargs or {})))
    if family == "wrf":
        if not {"RAINC", "RAINNC"}.issubset(ds.data_vars):
            raise KeyError("WRF input must contain RAINC and RAINNC for total precipitation.")
        data = wrf_total_precipitation(ds)
    else:
        name = _pick_variable(ds, family, variable)
        data = ds[name]
        if dim != "time":
            if dim not in data.dims:
                raise ValueError(f"Time dimension {dim!r} is not present in {name!r}.")
            data = data.rename({dim: "time"})
        if normalize:
            data = normalize_precipitation(data, dim="time")
    data = _rename_spatial(data)
    if "time" not in data.dims:
        raise ValueError("Precipitation data must have a time dimension.")
    if not {"lat", "lon"}.issubset(data.dims):
        raise ValueError("Precipitation data must expose 1-D lat/lon dimensions after loading.")
    data.name = variable or f"{family}_precipitation"
    data.attrs["climatevar:dataset_family"] = family
    return data


@dataclass(frozen=True)
class ProductSpec:
    """Specification for one precipitation product."""

    name: str
    family: str
    source: object
    variable: str | None = None
    chunks: object = None
    open_kwargs: Mapping | None = None

    def load(self, *, normalize: bool = True) -> xr.DataArray:
        return load_precipitation(
            self.source,
            family=self.family,
            variable=self.variable,
            chunks=self.chunks,
            normalize=normalize,
            open_kwargs=self.open_kwargs,
        )
