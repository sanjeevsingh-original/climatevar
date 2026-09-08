"""Explicit conversion of common climate variables to analysis units."""
from __future__ import annotations
from typing import Any
import xarray as xr

# Precipitation amount is intentionally kept in millimetres because mm is the
# standard depth unit used throughout rainfall/climate-extremes analysis.
# Other physical quantities use SI units. Precipitation flux remains an SI
# mass flux and is never silently converted to an amount without a time step.
SI_UNITS = {
    "temperature": "K", "surface_pressure": "Pa", "pressure": "Pa",
    "precipitation": "mm", "precipitation_amount": "mm", "precipitation_flux": "kg m-2 s-1",
    "relative_humidity": "1", "specific_humidity": "kg kg-1", "u_wind": "m s-1",
    "v_wind": "m s-1", "wind_speed": "m s-1", "geopotential": "m2 s-2",
    "geopotential_height": "m",
}


def _unit_text(value: Any) -> str:
    return str(value or "").strip().lower().replace("°", "deg").replace("*", "")


def _is_precip_flux(unit: str) -> bool:
    return unit in {"kg m-2 s-1", "kg/m2/s", "kg m-2 s^-1", "mm s-1", "mm/s", "mm day-1", "mm/day", "mm d-1", "mm h-1", "mm/hour", "mm hr-1", "mm min-1", "mm/min"}


def convert_units(data: xr.DataArray, target: str, *, source_units: str | None = None) -> xr.DataArray:
    """Convert a DataArray to an explicit target unit without guessing dimensions."""
    src = _unit_text(source_units if source_units is not None else data.attrs.get("units"))
    if not src:
        raise ValueError(f"No source units supplied for {data.name!r}.")
    target = target.strip()
    out = data.copy()

    if target == "K":
        if src in {"k", "kelvin"}: values = out
        elif src in {"degc", "c", "celsius", "degree_celsius", "degrees_celsius"}: values = out + 273.15
        else: raise ValueError(f"Cannot safely convert temperature from {source_units!r} to K.")
    elif target == "Pa":
        factors = {"pa": 1.0, "hpa": 100.0, "mb": 100.0, "mbar": 100.0, "kpa": 1000.0}
        if src not in factors: raise ValueError(f"Cannot safely convert pressure from {source_units!r} to Pa.")
        values = out * factors[src]
    elif target == "1":
        if src in {"1", "1.0", "fraction", "unitless", "dimensionless"}: values = out
        elif src in {"%", "percent", "percentage"}: values = out / 100.0
        else: raise ValueError(f"Cannot safely convert relative humidity from {source_units!r} to 1.")
    elif target == "kg kg-1":
        if src in {"kg kg-1", "kg/kg", "1"}: values = out
        elif src in {"g kg-1", "g/kg"}: values = out * 1e-3
        else: raise ValueError(f"Cannot safely convert specific humidity from {source_units!r} to kg kg-1.")
    elif target == "mm":
        factors = {"mm": 1.0, "meter": 1000.0, "metre": 1000.0, "m": 1000.0, "cm": 10.0}
        if src not in factors:
            raise ValueError("A precipitation rate cannot be converted to an amount without a time interval. Use precipitation_flux for rate data.")
        values = out * factors[src]
    elif target == "kg m-2 s-1":
        if src in {"kg m-2 s-1", "kg/m2/s", "kg m-2 s^-1"}: values = out
        elif src in {"mm s-1", "mm/s"}: values = out * 1e-3
        elif src in {"mm day-1", "mm/day", "mm d-1"}: values = out * 1e-3 / 86400.0
        elif src in {"mm h-1", "mm/hour", "mm hr-1"}: values = out * 1e-3 / 3600.0
        elif src in {"mm min-1", "mm/min"}: values = out * 1e-3 / 60.0
        else: raise ValueError(f"Cannot safely convert precipitation flux from {source_units!r} to kg m-2 s-1.")
    elif target == "m s-1":
        factors = {"m s-1": 1.0, "m/s": 1.0, "ms-1": 1.0, "km h-1": 1000.0 / 3600.0, "km/h": 1000.0 / 3600.0}
        if src not in factors: raise ValueError(f"Cannot safely convert wind speed from {source_units!r} to m s-1.")
        values = out * factors[src]
    elif target == "m2 s-2":
        if src != "m2 s-2": raise ValueError(f"Cannot safely convert geopotential from {source_units!r} to m2 s-2.")
        values = out
    else:
        raise ValueError(f"Unsupported conversion from {source_units!r} to {target!r}.")

    result = xr.DataArray(values, coords=out.coords, dims=out.dims, name=out.name, attrs=dict(out.attrs))
    result.attrs["units"] = target
    result.attrs["climatevar:source_units"] = source_units if source_units is not None else data.attrs.get("units")
    result.attrs["climatevar:unit_system"] = "SI_except_precipitation_amount_mm"
    return result


def to_si(data: xr.DataArray, quantity: str, *, source_units: str | None = None) -> xr.DataArray:
    """Convert a canonical climate variable to its project-standard unit."""
    if quantity not in SI_UNITS: raise KeyError(f"Unknown canonical quantity: {quantity!r}")
    return convert_units(data, SI_UNITS[quantity], source_units=source_units)


def normalize_units(ds: xr.Dataset, *, variables: list[str] | None = None) -> xr.Dataset:
    """Normalize recognized canonical variables.

    Precipitation amounts are always represented explicitly in ``mm``.
    Precipitation rates/fluxes are represented as ``kg m-2 s-1`` and tagged
    as fluxes; no rate-to-amount conversion is attempted without a temporal
    integration step.
    """
    out = ds.copy()
    selected = variables or list(SI_UNITS)
    for name in selected:
        if name not in out.data_vars or "units" not in out[name].attrs:
            continue
        if name == "precipitation" and _is_precip_flux(_unit_text(out[name].attrs["units"])):
            out[name] = convert_units(out[name], "kg m-2 s-1")
            out[name].attrs["climatevar:quantity"] = "precipitation_flux"
        else:
            out[name] = to_si(out[name], name)
            if name == "precipitation":
                out[name].attrs["climatevar:quantity"] = "precipitation_amount"
    history = list(out.attrs.get("climatevar:normalization", []))
    if isinstance(history, str): history = [history]
    history.append("normalize_units(target=SI_except_precipitation_amount_mm)")
    out.attrs["climatevar:normalization"] = history
    return out
