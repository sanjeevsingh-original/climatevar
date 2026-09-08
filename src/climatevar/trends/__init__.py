"""Trend and monotonic-change diagnostics."""

from .trend import mann_kendall, sens_slope

__all__ = ["mann_kendall", "sens_slope"]
