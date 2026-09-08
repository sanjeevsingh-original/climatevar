"""Explicit conversion of common climate variables to SI units.

The library treats physical meaning separately from variable names. Conversions
are applied only when the source unit is explicit or supplied by the caller.
Precipitation amount and precipitation flux are intentionally different
quantities and are never silently converted between one another.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import xarray as xr


SI_UNITS = {
    "temperature": "K",
    "surface_pressure": "Pa",
    "pressure": "Pa",
    "precipitation": "m",
    "precipitation_amount": "m",
    "precipitation_flux": "kg m-2 s-1",
    "relative_humidity": "1",
    "specific_humidity": "kg kg-1",
    "u_wind": "m s-1",
    "v_wind": "m s-1",
    "wind_speed": "m s-1",
    "geopotential": "m2 s-2",
    "geopotential_height": "m",
}


def _unit_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("°", "deg").replace("*", "")


def convert_units(data: xr.DataArray, target: str, *, source_units: str | None = None) -> xr.DataArray:
    """Convert a DataArray to an explicit SI-compatible target unit.

    Supported conversions cover the common ERA5, IMD/IMDAA, IMERG, WRF and
    CMIP climate variables. Unknown or dimensionally ambiguous units raise a
    ValueError instead of guessing.
    """
    src = _unit_text(source_units if source_units is not None else data.attrs.get("units"))
    if not src:
        raise ValueError(f"No source units supplied for {data.name!r}.")
    target = target.strip()

    factors = {
        ("mm", "m"): 1e-3,
        ("millimeter", "m"): 1e-3,
        ("millimetre", "m"): 1e-3,
        ("cm", "m"): 1e-2,
        ("meter", "m"): 1.0,
        ("metre", "m"): 1.0,
        ("m", "m"): 1.0,
        ("hpa", "Pa"): 100.0,
        ("mb", "Pa"): 100.0,
        ("mbar", "Pa"): 100.0,
        ("kpa", "Pa"): 1000.0,
        ("pa", "Pa"): 1.0,
        ("m s-1", "m s-1"): 1.0,
        ("m/s", "m s-1"): 1.0,
        ("ms-1", "m s-1"): 1.0,
        ("km h-1", "m s-1"): 1000.0 / 3600.0,
        ("km/h", "m s-1"): 1000.0 / 3600.0,
        ("m2 s-2", "m2 s-2"): 1.0,
        ("m", "m"): 1.0,
    }

    out = data.copy()
    if target == "K":
        if src in {"k", "kelvin"}:
            values = out
        elif src in {"degc", "c", "celsius", "degree_celsius", "degrees_celsius"}:
            values = out + 273.15
        else:
            raise ValueError(f"Cannot safely convert temperature from {source_units!r} to K.")
    elif target == "1":
        if src in {"1", "1.0", "fraction", "unitless", "dimensionless"}:
            values = out
        elif src in {"%", "percent", "percentage"}:
            values = out / 100.0
        else:
            raise ValueError(f"Cannot safely convert relative humidity from {source_units!r} to 1.")
    elif target == "kg kg-1":
        if src in {"kg kg-1", "kg/kg", "1", "g kg-1", "g/kg"}:
            values = out if src in {"kg kg-1", "kg/kg", "1"} else out * 1e-3
        else:
            raise ValueError(f"Cannot safely convert specific humidity from {source_units!r} to kg kg-1.")
    elif target == "kg m-2 s-1":
        if src in {"kg m-2 s-1", "kg/m2/s", "kg m-2 s^-1", "mm s-1", "m s-1"}:
            values = out
        elif src in {"mm day-1", "mm/day", "mm d-1", "mm day^-1"}:
            values = out * 1e-3 / 86400.0
        elif src in {"mm h-1", "mm/hour", "mm hr-1"}:
            values = out * 1e-3 / 3600.0
        elif src in {"mm min-1", "mm/min"}:
            values = out * 1e-3 / 60.0
        else:
            raise ValueError("Precipitation amount and precipitation flux are different quantities; provide an explicit rate unit.")
    elif (src, target) in factors:
        values = out * factors[(src, target)]
    else:
        raise ValueError(f"Unsupported conversion from {source_units!r} to {target!r}.")

    values = xr.DataArray(values, coords=out.coords, dims=out.dims, name=out.name, attrs=dict(out.attrs))
    values.attrs["units"] = target
    values.attrs["climatevar:source_units"] = source_units if source_units is not None else data.attrs.get("units")
    values.attrs["climatevar:unit_system"] = "SI"
    return values


def to_si(data: xr.DataArray, quantity: str, *, source_units: str | None = None) -> xr.DataArray:
    """Convert a canonical climate variable to its SI unit."""
    if quantity not in SI_UNITS:
        raise KeyError(f"Unknown canonical quantity: {quantity!r}")
    return convert_units(data, SI_UNITS[quantity], source_units=source_units)


def normalize_units(ds: xr.Dataset, *, variables: list[str] | None = None) -> xr.Dataset:
    """Convert recognized canonical variables in a Dataset to SI units.

    Variables without an explicit ``units`` attribute are left untouched. This
    is intentional: silently assuming units would be scientifically unsafe.
    """
    out = ds.copy()
    selected = variables or list(SI_UNITS)
    for name in selected:
        if name not in out.data_vars or name not in SI_UNITS:
            continue
        if "units" not in out[name].attrs:
            continue
        out[name] = to_si(out[name], name)
    history = list(out.attrs.get("climatevar:normalization", []))
    if isinstance(history, str):
        history = [history]
    history.append("normalize_units(target=SI)")
    out.attrs["climatevar:normalization"] = history
    return out
