"""Moisture diagnostics."""

from __future__ import annotations

import numpy as np


def saturation_vapor_pressure(temperature):
    """Saturation vapor pressure in Pa using the Bolton formulation.

    Temperature must be in K.
    """
    tc = temperature - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def specific_humidity_from_rh(temperature, pressure, relative_humidity):
    """Estimate specific humidity from temperature, pressure and RH.

    Temperature is K, pressure is Pa, and relative humidity is fractional
    (0--1). The formulation assumes water vapor over liquid water.
    """
    if np.any(relative_humidity < 0) or np.any(relative_humidity > 1):
        raise ValueError("relative_humidity must be between 0 and 1.")
    e_s = saturation_vapor_pressure(temperature)
    e = relative_humidity * e_s
    epsilon = 0.622
    return epsilon * e / (pressure - (1.0 - epsilon) * e)
