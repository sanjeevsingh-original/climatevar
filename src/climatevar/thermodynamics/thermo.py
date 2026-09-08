"""Potential-temperature diagnostics."""

from __future__ import annotations

import numpy as np
import xarray as xr

RD_CP = 0.2854


def potential_temperature(
    temperature: xr.DataArray | float,
    pressure: xr.DataArray | float,
    p0: float = 100000.0,
):
    """Calculate dry potential temperature in K.

    Parameters use SI units: temperature in K and pressure in Pa.
    """
    return temperature * (p0 / pressure) ** RD_CP


def equivalent_potential_temperature(
    temperature: xr.DataArray | float,
    pressure: xr.DataArray | float,
    specific_humidity: xr.DataArray | float,
    p0: float = 100000.0,
):
    """Calculate approximate equivalent potential temperature in K.

    This follows the widely used Bolton formulation. Inputs are temperature
    (K), pressure (Pa), and specific humidity (kg/kg).
    """
    t = xr.DataArray(temperature) if not isinstance(temperature, xr.DataArray) else temperature
    p = xr.DataArray(pressure) if not isinstance(pressure, xr.DataArray) else pressure
    q = xr.DataArray(specific_humidity) if not isinstance(specific_humidity, xr.DataArray) else specific_humidity
    if np.any(t <= 0) or np.any(p <= 0):
        raise ValueError("temperature and pressure must be positive.")
    if np.any((q < 0) | (q >= 1)):
        raise ValueError("specific_humidity must satisfy 0 <= q < 1.")

    r = q / (1.0 - q)
    e_pa = r * p / (0.622 + r)
    e_hpa = e_pa / 100.0
    tl = 1.0 / (1.0 / (t - 55.0) - np.log(e_hpa / 6.112) / 2840.0) + 55.0
    theta_l = t * (p0 / p) ** (RD_CP * (1.0 - 0.28 * r))
    return theta_l * np.exp((3036.0 / tl - 1.78) * r * (1.0 + 0.448 * r))
