"""Precipitation normalization, indices, and hydroclimate diagnostics."""

from .indices import (
    cdd,
    cwd,
    prcptot,
    r10mm,
    r20mm,
    r95p,
    r99p,
    rx1day,
    rx5day,
)
from .normalize import daily_amount, interval_amount_mm, normalize_precipitation

__all__ = [
    "cdd",
    "cwd",
    "daily_amount",
    "interval_amount_mm",
    "normalize_precipitation",
    "prcptot",
    "r10mm",
    "r20mm",
    "r95p",
    "r99p",
    "rx1day",
    "rx5day",
]
