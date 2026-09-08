"""Climate variability and atmospheric science tools."""

from ._version import __version__
from .climatology import anomaly, climatology
from .drought import spi_like
from .quality import annual_valid_fraction, require_completeness, valid_fraction
from .seasonal import monsoon_total, seasonal_mean

__all__ = [
    "__version__",
    "anomaly",
    "annual_valid_fraction",
    "climatology",
    "monsoon_total",
    "require_completeness",
    "seasonal_mean",
    "spi_like",
    "valid_fraction",
]
