"""Potential-temperature diagnostics."""

from __future__ import annotations

import xarray as xr

RD = 287.05
CP = 1004.0
EPSILON = 0.622
LV = 2.5e6


def potential_temperature(temperature: xr.DataArray | float, pressure: xr.DataArray | float, p0: float = 100000.0):
    """Calculate dry potential temperature in K.

    Parameters use SI units: temperature in K and pressure in Pa.
    """
    return temperature * (p0 / pressure) ** (RD / CP)


def equivalent_potential_temperature(
    temperature: xr.DataArray | float,
    pressure: xr.DataArray | float,
    specific_humidity: xr.DataArray | float,
    p0: float = 100000.0,
):
    """Approximate equivalent potential temperature (Bolton-style form).

    Inputs are temperature (K), pressure (Pa), and specific humidity (kg/kg).
    The calculation estimates vapor pressure from specific humidity and uses
    Bolton's commonly applied lifting-condensation-level approximation.
    """
    theta = potential_temperature(temperature, pressure, p0=p0)
    q = xr.DataArray(specific_humidity) if not isinstance(specific_humidity, xr.DataArray) else specific_humidity
    t = xr.DataArray(temperature) if not isinstance(temperature, xr.DataArray) else temperature
    p = xr.DataArray(pressure) if not isinstance(pressure, xr.DataArray) else pressure
    e = q * p / (EPSILON + (1.0 - EPSILON) * q)
    e_hpa = e / 100.0
    tl = 1.0 / (1.0 / (t - 55.0) - xr.apply_ufunc(xr.ufuncs.log, e_hpa / 6.112) / 2840.0) + 55.0
    return theta * xr.apply_ufunc(xr.ufuncs.exp, (LV * q) / (CP * tl))
