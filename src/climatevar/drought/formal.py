"""Formal SPI and SPEI drought indices.

The implementation follows the standard probability-transformation approach:
accumulate the input series, fit a distribution to each calendar month, and
transform its cumulative probability to a standard normal variate.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import gamma, fisk, norm


def _fit_spi_1d(values: np.ndarray, distribution: str) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    out = np.full(y.shape, np.nan, dtype=float)
    finite = np.isfinite(y)
    sample = y[finite]
    if sample.size < 3:
        return out
    positive = sample > 0
    p0 = 1.0 - positive.mean()
    dist = distribution.lower()
    if dist == "gamma":
        if positive.sum() < 3:
            return out
        a, loc, scale = gamma.fit(sample[positive], floc=0)
        cdf = np.full(sample.shape, p0, dtype=float)
        cdf[positive] = p0 + (1.0 - p0) * gamma.cdf(sample[positive], a, loc=loc, scale=scale)
    elif dist in {"pearson3", "pearson_iii", "pearson-iii"}:
        from scipy.stats import pearson3
        if positive.sum() < 3:
            return out
        skew, loc, scale = pearson3.fit(sample[positive])
        cdf = np.full(sample.shape, p0, dtype=float)
        cdf[positive] = p0 + (1.0 - p0) * pearson3.cdf(sample[positive], skew, loc=loc, scale=scale)
    else:
        raise ValueError("distribution must be 'gamma' or 'pearson3'.")
    z = norm.ppf(np.clip(cdf, 1e-8, 1.0 - 1e-8))
    out[finite] = z
    return out


def _monthly_fit(values: np.ndarray, months: np.ndarray, distribution: str) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for month in range(1, 13):
        idx = months == month
        if idx.any():
            out[idx] = _fit_spi_1d(values[idx], distribution)
    return out


def spi(
    precipitation: xr.DataArray,
    scale: int = 3,
    dim: str = "time",
    distribution: str = "gamma",
) -> xr.DataArray:
    """Calculate formal Standardized Precipitation Index (SPI).

    Parameters
    ----------
    precipitation:
        Non-negative precipitation amounts at a regular temporal resolution,
        preferably monthly totals for conventional SPI applications.
    scale:
        Accumulation scale in input time steps (e.g. 3 for SPI-3 on monthly data).
    dim:
        Time dimension.
    distribution:
        ``"gamma"`` (default) or ``"pearson3"``.

    Notes
    -----
    Zero precipitation is handled through the mixed-distribution probability
    ``H(x) = q + (1-q)G(x)`` before transformation with the standard normal CDF.
    Distribution parameters are fitted separately for each calendar month.
    """
    if dim not in precipitation.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the data.")
    if scale < 1 or int(scale) != scale:
        raise ValueError("scale must be a positive integer.")
    if np.any(precipitation < 0):
        raise ValueError("precipitation must be non-negative.")
    accumulated = precipitation.rolling({dim: int(scale)}, min_periods=int(scale)).sum()
    months = accumulated[dim].dt.month
    result = xr.apply_ufunc(
        _monthly_fit,
        accumulated,
        months,
        kwargs={"distribution": distribution},
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    result = result.transpose(*precipitation.dims)
    result.name = "spi"
    result.attrs = dict(precipitation.attrs)
    result.attrs.update({"standard_name": "standardized_precipitation_index", "climatevar:scale": int(scale), "climatevar:distribution": distribution, "climatevar:method": "calendar-month probability fit"})
    return result


def spei(
    precipitation: xr.DataArray,
    potential_evapotranspiration: xr.DataArray,
    scale: int = 3,
    dim: str = "time",
) -> xr.DataArray:
    """Calculate SPEI from the climatic water balance ``P - PET``.

    The three-parameter log-logistic (Fisk) distribution is fitted separately
    for each calendar month to accumulated water-balance values, then mapped to
    a standard normal variate. This is a practical publication-oriented
    implementation; users should document PET method and calibration period.
    """
    if precipitation.dims != potential_evapotranspiration.dims or precipitation.sizes != potential_evapotranspiration.sizes:
        raise ValueError("precipitation and potential_evapotranspiration must have matching dimensions and sizes.")
    if dim not in precipitation.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the data.")
    if scale < 1 or int(scale) != scale:
        raise ValueError("scale must be a positive integer.")
    water_balance = precipitation - potential_evapotranspiration
    accumulated = water_balance.rolling({dim: int(scale)}, min_periods=int(scale)).sum()
    months = accumulated[dim].dt.month

    def fit(values, month_values):
        out = np.full(values.shape, np.nan, dtype=float)
        for month in range(1, 13):
            idx = month_values == month
            sample = values[idx]
            finite = np.isfinite(sample)
            if finite.sum() < 3:
                continue
            y = sample[finite]
            shape, loc, scale_ = fisk.fit(y)
            cdf = fisk.cdf(y, shape, loc=loc, scale=scale_)
            z = norm.ppf(np.clip(cdf, 1e-8, 1 - 1e-8))
            positions = np.where(idx)[0][finite]
            out[positions] = z
        return out

    result = xr.apply_ufunc(fit, accumulated, months, input_core_dims=[[dim], [dim]], output_core_dims=[[dim]], vectorize=True, dask="parallelized", output_dtypes=[float])
    result = result.transpose(*precipitation.dims)
    result.name = "spei"
    result.attrs = {"standard_name": "standardized_precipitation_evapotranspiration_index", "climatevar:scale": int(scale), "climatevar:distribution": "log-logistic (Fisk)", "climatevar:method": "calendar-month water-balance fit"}
    return result
