"""Precipitation normalization, indices, and hydroclimate diagnostics."""

from .indices import (
    cdd, cwd, prcptot, r10mm, r20mm, r95p, r99p, r95p_fraction, r99p_fraction,
    rx1day, rx5day, sdii, wet_day_count,
)
from .normalize import daily_amount, interval_amount_mm, normalize_precipitation
from .wrf import wrf_precipitation_amount, wrf_total_precipitation

__all__ = [
    "cdd", "cwd", "daily_amount", "interval_amount_mm", "normalize_precipitation",
    "prcptot", "r10mm", "r20mm", "r95p", "r99p", "r95p_fraction", "r99p_fraction",
    "rx1day", "rx5day", "sdii", "wet_day_count", "wrf_precipitation_amount", "wrf_total_precipitation",
]
