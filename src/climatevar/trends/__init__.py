"""Trend and monotonic-change diagnostics."""

from .trend import mann_kendall, modified_mann_kendall, sens_slope, trend_free_prewhitening

__all__ = ["mann_kendall", "modified_mann_kendall", "sens_slope", "trend_free_prewhitening"]
