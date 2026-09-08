"""Normalize heterogeneous climate datasets to a small canonical schema.

The same physical quantity is often stored under different names across ERA5,
IMD, IMDAA, IMERG and WRF. This module makes those differences explicit while
keeping an audit trail in dataset attributes. It does not silently convert units.
"""

from __future__ import annotations

from collections.abc import Mapping

import xarray as xr


CANONICAL_COORDS = {"latitude", "longitude", "time"}

COORD_ALIASES: dict[str, tuple[str, ...]] = {
    "latitude": (
        "latitude", "lat", "Latitude", "LATITUDE", "nav_lat", "y_lat",
        "XLAT", "XLAT_M", "XLAT_U", "XLAT_V",
    ),
    "longitude": (
        "longitude", "lon", "Longitude", "LONGITUDE", "nav_lon", "x_lon",
        "XLONG", "XLONG_M", "XLONG_U", "XLONG_V",
    ),
    "time": (
        "time", "Time", "TIME", "valid_time", "datetime", "date", "Date",
        "Times",
    ),
}

VARIABLE_ALIASES: dict[str, tuple[str, ...]] = {
    "precipitation": (
        "precipitation", "precip", "precipitationCal", "precipitationCalES",
        "pr", "tp", "rain", "rainfall", "rf", "APCP_sfc", "RAINC", "RAINNC", "RAINSH",
    ),
    "temperature": (
        "temperature", "temp", "t2m", "TMP_2m", "TMP_sfc", "T2", "tas", "air_temperature",
    ),
    "surface_pressure": (
        "surface_pressure", "sp", "ps", "PRES_sfc", "PSFC", "pres", "pressure",
    ),
    "relative_humidity": (
        "relative_humidity", "rh", "RH2", "RH_2m", "RH_prl", "r", "hur",
    ),
    "specific_humidity": (
        "specific_humidity", "q", "q2", "Q2", "hus",
    ),
    "u_wind": (
        "u_wind", "u10", "U10", "UGRD_10m", "UGRD_prl", "ua", "uwnd", "u",
    ),
    "v_wind": (
        "v_wind", "v10", "V10", "VGRD_10m", "VGRD_prl", "va", "vwnd", "v",
    ),
    "geopotential": (
        "geopotential", "z", "gh", "HGT_prl", "HGT", "GHT",
    ),
}

DATASET_PRESETS: dict[str, dict[str, str]] = {
    "era5": {
        "latitude": "latitude", "longitude": "longitude", "time": "time",
        "precipitation": "tp", "temperature": "t2m", "surface_pressure": "sp",
    },
    "imerg": {
        "latitude": "lat", "longitude": "lon", "time": "time",
        "precipitation": "precipitation",
    },
    "wrf": {
        "latitude": "XLAT", "longitude": "XLONG", "time": "Time",
        "precipitation": "RAINNC",
    },
    "imd": {
        "latitude": "lat", "longitude": "lon", "time": "time",
    },
    "imdaa": {
        "latitude": "latitude", "longitude": "longitude", "time": "time",
        "precipitation": "APCP_sfc", "temperature": "TMP_2m",
        "surface_pressure": "PRES_sfc", "relative_humidity": "RH_2m",
        "u_wind": "UGRD_10m", "v_wind": "VGRD_10m",
    },
}


def _find_name(names: list[str], aliases: tuple[str, ...]) -> str | None:
    """Find an alias using exact, then case-insensitive matching."""
    for alias in aliases:
        if alias in names:
            return alias
    folded = {name.casefold(): name for name in names}
    for alias in aliases:
        if alias.casefold() in folded:
            return folded[alias.casefold()]
    return None


def _wrf_times(ds: xr.Dataset) -> xr.Dataset:
    """Decode WRF ``Times`` character arrays when they are not already decoded."""
    if "Times" not in ds.variables or "time" in ds.coords:
        return ds
    values = ds["Times"].values
    if getattr(values, "ndim", 0) != 2:
        return ds
    try:
        strings = [b"".join(row).decode("utf-8") for row in values]
    except (AttributeError, UnicodeDecodeError, TypeError):
        try:
            strings = ["".join(str(x) for x in row) for row in values]
        except Exception:
            return ds
    parsed = [s.replace("_", " ") for s in strings]
    try:
        import pandas as pd
        time = pd.to_datetime(parsed).to_numpy(dtype="datetime64[ns]")
    except Exception:
        return ds
    if "Time" in ds.dims:
        ds = ds.assign_coords(time=("Time", time))
    return ds


def normalize_coords(
    ds: xr.Dataset,
    *,
    dataset: str | None = None,
    strict: bool = False,
) -> xr.Dataset:
    """Rename common latitude/longitude/time coordinates to CF-like names."""
    ds = _wrf_times(ds)
    names = list(ds.coords) + [name for name in ds.variables if name not in ds.coords]
    preset = DATASET_PRESETS.get((dataset or "").lower(), {})
    renames: dict[str, str] = {}

    for canonical, aliases in COORD_ALIASES.items():
        preferred = preset.get(canonical)
        source = preferred if preferred in names else _find_name(names, aliases)
        if source and source != canonical and canonical not in ds.variables:
            renames[source] = canonical

    out = ds.rename(renames) if renames else ds
    if strict:
        missing = [name for name in ("latitude", "longitude") if name not in out]
        if missing:
            raise ValueError(f"Could not identify required coordinate(s): {', '.join(missing)}")

    history = list(out.attrs.get("climatevar:normalization", []))
    if isinstance(history, str):
        history = [history]
    history.append(f"normalize_coords(dataset={dataset!r})")
    out.attrs["climatevar:normalization"] = history
    return out


def standardize_variables(
    ds: xr.Dataset,
    *,
    variables: Mapping[str, str] | None = None,
    dataset: str | None = None,
    strict: bool = False,
) -> xr.Dataset:
    """Rename common physical variables to climatevar canonical names.

    ``variables`` is the safest mechanism for ambiguous products, e.g.
    ``{"precipitation": "RAINNC"}``. If omitted, aliases are searched.
    Existing canonical names are never overwritten.
    """
    mapping = dict(variables or {})
    preset = DATASET_PRESETS.get((dataset or "").lower(), {})
    names = list(ds.data_vars)
    renames: dict[str, str] = {}

    for canonical, aliases in VARIABLE_ALIASES.items():
        if canonical in ds.data_vars:
            continue
        source = mapping.get(canonical) or preset.get(canonical)
        if source not in names:
            source = _find_name(names, aliases)
        if source and source != canonical:
            renames[source] = canonical

    if strict and variables:
        missing = [target for target, source in mapping.items() if source not in ds.data_vars]
        if missing:
            raise KeyError(f"Requested source variable(s) not found: {', '.join(missing)}")

    out = ds.rename(renames) if renames else ds
    history = list(out.attrs.get("climatevar:normalization", []))
    if isinstance(history, str):
        history = [history]
    history.append(f"standardize_variables(dataset={dataset!r}, variables={mapping!r})")
    out.attrs["climatevar:normalization"] = history
    return out


def normalize_dataset(
    ds: xr.Dataset,
    *,
    dataset: str | None = None,
    variables: Mapping[str, str] | None = None,
    strict: bool = True,
) -> xr.Dataset:
    """Normalize coordinates and common variables in one auditable operation."""
    out = normalize_coords(ds, dataset=dataset, strict=strict)
    return standardize_variables(out, variables=variables, dataset=dataset, strict=False)


def find_variable(ds: xr.Dataset, standard_name: str) -> str | None:
    """Return the matching source variable for a canonical physical quantity."""
    if standard_name in ds.data_vars:
        return standard_name
    aliases = VARIABLE_ALIASES.get(standard_name, ())
    return _find_name(list(ds.data_vars), aliases)
