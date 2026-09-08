"""Trend and monotonic-change diagnostics."""

from .seasonal import fdr_mask, seasonal_mann_kendall, seasonal_sen_slope, seasonal_series, spatial_trend
from .trend import mann_kendall, modified_mann_kendall, sens_slope, trend_free_prewhitening

__all__ = [
    "mann_kendall", "modified_mann_kendall", "sens_slope", "trend_free_prewhitening",
    "seasonal_series", "seasonal_mann_kendall", "seasonal_sen_slope", "fdr_mask", "spatial_trend",
]
