"""Public non-parametric trend API.

Implementation is kept in :mod:`trend_impl` so reference-validated numerical
routines remain isolated from the public compatibility layer.
"""
from .trend_impl import (
    _clean, _lag_autocorrelation, _mk_stats, _mk_variance, _sen_1d,
    mann_kendall, modified_mann_kendall, sens_slope, sens_slope_ci,
    trend_free_prewhitening,
)

__all__ = ["mann_kendall", "modified_mann_kendall", "sens_slope", "sens_slope_ci", "trend_free_prewhitening"]
