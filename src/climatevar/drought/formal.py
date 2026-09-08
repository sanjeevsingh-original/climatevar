"""Formal SPI and SPEI drought indices.

The implementation follows the standard probability-transformation approach:
accumulate the input series, fit a distribution separately by calendar month,
and transform cumulative probabilities to standard normal scores.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import gamma, fisk, norm, kstest


def _calibration_mask(time, start, end):
    mask = xr.ones_like(time, dtype=bool)
    if start is not None:
        mask = mask & (time >= np.datetime64(start))
    if end is not None:
        mask = mask & (time <= np.datetime64(end))
    return mask


def _fit_spi_1d(values, months, calibration, distribution, min_nonzero):
    values = np.asarray(values, dtype=float)
    months = np.asarray(months)
    calibration = np.asarray(calibration, dtype=bool)
    out = np.full(values.shape, np.nan, dtype=float)
    for month in range(1, 13):
        target = (months == month) & np.isfinite(values)
        fit_mask = target & calibration
        sample = values[fit_mask]
        if sample.size == 0:
            continue
        positive = sample > 0
        if positive.sum() < min_nonzero:
            continue
        p0 = 1.0 - positive.mean()
        dist = distribution.lower()
        if dist == "gamma":
            a, loc, scale = gamma.fit(sample[positive], floc=0)
            cdf = np.full(values.shape, np.nan)
            positive_target = target & (values > 0)
            cdf[target & (values <= 0)] = p0
            cdf[positive_target] = p0 + (1.0 - p0) * gamma.cdf(values[positive_target], a, loc=loc, scale=scale)
        elif dist in {"pearson3", "pearson_iii", "pearson-iii"}:
            from scipy.stats import pearson3
            skew, loc, scale = pearson3.fit(sample[positive])
            cdf = np.full(values.shape, np.nan)
            positive_target = target & (values > 0)
            cdf[target & (values <= 0)] = p0
            cdf[positive_target] = p0 + (1.0 - p0) * pearson3.cdf(values[positive_target], skew, loc=loc, scale=scale)
        else:
            raise ValueError("distribution must be 'gamma' or 'pearson3'.")
        out[target] = norm.ppf(np.clip(cdf[target], 1e-8, 1.0 - 1e-8))
    return out


def _fit_spei_1d(values, months, calibration, min_samples):
    values = np.asarray(values, dtype=float)
    months = np.asarray(months)
    calibration = np.asarray(calibration, dtype=bool)
    out = np.full(values.shape, np.nan, dtype=float)
    for month in range(1, 13):
        target = (months == month) & np.isfinite(values)
        fit_mask = target & calibration
        sample = values[fit_mask]
        if sample.size < min_samples:
            continue
        shape, loc, scale = fisk.fit(sample)
        out[target] = norm.ppf(np.clip(fisk.cdf(values[target], shape, loc=loc, scale=scale), 1e-8, 1.0 - 1e-8))
    return out


def spi(
    precipitation: xr.DataArray,
    scale: int = 3,
    dim: str = "time",
    distribution: str = "gamma",
    calibration_start=None,
    calibration_end=None,
    min_nonzero: int = 10,
    clip: float | None = None,
) -> xr.DataArray:
    """Calculate formal Standardized Precipitation Index (SPI).

    ``calibration_start`` and ``calibration_end`` define the reference period
    used to estimate parameters; fitted parameters are then applied to the
    complete input record. The default minimum of 10 non-zero calibration
    values follows common operational SPI practice for a 30-year baseline.
    """
    if dim not in precipitation.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the data.")
    if scale < 1 or int(scale) != scale:
        raise ValueError("scale must be a positive integer.")
    if min_nonzero < 3:
        raise ValueError("min_nonzero must be at least 3.")
    if bool((precipitation < 0).any()):
        raise ValueError("precipitation must be non-negative.")
    accumulated = precipitation.rolling({dim: int(scale)}, min_periods=int(scale)).sum()
    months = accumulated[dim].dt.month
    calibration = _calibration_mask(accumulated[dim], calibration_start, calibration_end)
    result = xr.apply_ufunc(
        _fit_spi_1d, accumulated, months, calibration,
        kwargs={"distribution": distribution, "min_nonzero": int(min_nonzero)},
        input_core_dims=[[dim], [dim], [dim]], output_core_dims=[[dim]],
        vectorize=True, dask="parallelized", output_dtypes=[float],
    ).transpose(*precipitation.dims)
    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive.")
        result = result.clip(min=-float(clip), max=float(clip))
    result.name = "spi"
    result.attrs = dict(precipitation.attrs)
    result.attrs.update({
        "standard_name": "standardized_precipitation_index",
        "climatevar:scale": int(scale),
        "climatevar:distribution": distribution,
        "climatevar:min_nonzero": int(min_nonzero),
        "climatevar:calibration_start": str(calibration_start) if calibration_start is not None else "full_record",
        "climatevar:calibration_end": str(calibration_end) if calibration_end is not None else "full_record",
    })
    return result


def spei(
    precipitation: xr.DataArray,
    potential_evapotranspiration: xr.DataArray,
    scale: int = 3,
    dim: str = "time",
    calibration_start=None,
    calibration_end=None,
    min_samples: int = 10,
    clip: float | None = None,
) -> xr.DataArray:
    """Calculate SPEI from climatic water balance ``P - PET``.

    The three-parameter log-logistic distribution is represented by SciPy's
    Fisk distribution. A fixed calibration period can be supplied so that
    historical and future periods are standardized against the same climate.
    """
    if dim not in precipitation.dims or dim not in potential_evapotranspiration.dims:
        raise ValueError(f"Dimension {dim!r} must be present in both inputs.")
    if precipitation.dims != potential_evapotranspiration.dims or precipitation.sizes != potential_evapotranspiration.sizes:
        raise ValueError("precipitation and potential_evapotranspiration must have matching dimensions and sizes.")
    if scale < 1 or int(scale) != scale:
        raise ValueError("scale must be a positive integer.")
    if min_samples < 3:
        raise ValueError("min_samples must be at least 3.")
    water_balance = precipitation - potential_evapotranspiration
    accumulated = water_balance.rolling({dim: int(scale)}, min_periods=int(scale)).sum()
    months = accumulated[dim].dt.month
    calibration = _calibration_mask(accumulated[dim], calibration_start, calibration_end)
    result = xr.apply_ufunc(
        _fit_spei_1d, accumulated, months, calibration,
        kwargs={"min_samples": int(min_samples)},
        input_core_dims=[[dim], [dim], [dim]], output_core_dims=[[dim]],
        vectorize=True, dask="parallelized", output_dtypes=[float],
    ).transpose(*precipitation.dims)
    if clip is not None:
        if clip <= 0:
            raise ValueError("clip must be positive.")
        result = result.clip(min=-float(clip), max=float(clip))
    result.name = "spei"
    result.attrs = {
        "standard_name": "standardized_precipitation_evapotranspiration_index",
        "climatevar:scale": int(scale),
        "climatevar:distribution": "three-parameter log-logistic (Fisk)",
        "climatevar:min_samples": int(min_samples),
        "climatevar:calibration_start": str(calibration_start) if calibration_start is not None else "full_record",
        "climatevar:calibration_end": str(calibration_end) if calibration_end is not None else "full_record",
    }
    return result


def drought_category(index: xr.DataArray) -> xr.DataArray:
    """Classify SPI/SPEI values using standard drought/wetness categories."""
    result = xr.full_like(index, "near normal", dtype=object)
    result = xr.where(index <= -2.0, "extreme drought", result)
    result = xr.where((index > -2.0) & (index <= -1.5), "severe drought", result)
    result = xr.where((index > -1.5) & (index <= -1.0), "moderate drought", result)
    result = xr.where((index >= 1.0) & (index < 1.5), "moderately wet", result)
    result = xr.where((index >= 1.5) & (index < 2.0), "severely wet", result)
    result = xr.where(index >= 2.0, "extremely wet", result)
    result = result.where(np.isfinite(index))
    result.name = f"{index.name or 'index'}_category"
    result.attrs["climatevar:classification"] = "SPI/SPEI standard drought categories"
    return result


def fit_quality(index: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Return basic standardization diagnostics for an SPI/SPEI result.

    The diagnostics are calculated on finite index values and provide sample
    size, mean, standard deviation, and a normality KS-test p-value. This is a
    diagnostic of the standardized output, not a substitute for testing the
    fitted source distribution itself.
    """
    def _quality(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 5:
            return np.nan, np.nan, np.nan, float(x.size)
        _, p = kstest(x, "norm")
        return float(x.mean()), float(x.std(ddof=1)), float(p), float(x.size)

    mean, std, pvalue, n = xr.apply_ufunc(
        _quality, index,
        input_core_dims=[[dim]], output_core_dims=[[], [], [], []],
        vectorize=True, dask="parallelized", output_dtypes=[float, float, float, float],
    )
    return xr.Dataset({"mean": mean, "std": std, "normality_pvalue": pvalue, "n": n})
