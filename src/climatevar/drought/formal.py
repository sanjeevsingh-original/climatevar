"""Formal SPI, SPEI, drought classification, and diagnostics."""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import gamma, fisk, norm, kstest


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
    out[finite] = norm.ppf(np.clip(cdf, 1e-8, 1.0 - 1e-8))
    return out


def _monthly_fit(values: np.ndarray, months: np.ndarray, distribution: str) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    for month in range(1, 13):
        idx = months == month
        if idx.any():
            out[idx] = _fit_spi_1d(values[idx], distribution)
    return out


def _validate_scale(scale: int) -> int:
    if scale < 1 or int(scale) != scale:
        raise ValueError("scale must be a positive integer.")
    return int(scale)


def spi(
    precipitation: xr.DataArray,
    scale: int = 3,
    dim: str = "time",
    distribution: str = "gamma",
    calibration_start=None,
    calibration_end=None,
    min_samples: int = 10,
    clip: float | None = 8.0,
) -> xr.DataArray:
    """Calculate formal Standardized Precipitation Index (SPI).

    Distribution parameters are fitted independently for each calendar month
    using only the requested calibration period, then applied to the full
    accumulated series. Zero precipitation is treated as a mixed probability
    component before transformation to the standard normal distribution.
    """
    scale = _validate_scale(scale)
    if dim not in precipitation.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the data.")
    if bool((precipitation < 0).any()):
        raise ValueError("precipitation must be non-negative.")
    accumulated = precipitation.rolling({dim: scale}, min_periods=scale).sum()
    calibration = accumulated.sel({dim: slice(calibration_start, calibration_end)}) if calibration_start or calibration_end else accumulated
    months = accumulated[dim].dt.month
    cal_months = calibration[dim].dt.month

    def fit_apply(values, month_values, cal_values, cal_month_values):
        out = np.full(values.shape, np.nan, dtype=float)
        for month in range(1, 13):
            cal = np.asarray(cal_values[cal_month_values == month], dtype=float)
            cal = cal[np.isfinite(cal)]
            if cal.size < min_samples:
                continue
            positive = cal > 0
            if positive.sum() < max(3, min_samples // 2):
                continue
            p0 = 1.0 - positive.mean()
            if distribution.lower() == "gamma":
                a, loc, sc = gamma.fit(cal[positive], floc=0)
                x = np.asarray(values[month_values == month], dtype=float)
                finite = np.isfinite(x)
                cdf = np.full(x.shape, np.nan)
                cdf[finite & (x <= 0)] = p0
                pos = finite & (x > 0)
                cdf[pos] = p0 + (1-p0) * gamma.cdf(x[pos], a, loc=loc, scale=sc)
            elif distribution.lower() in {"pearson3", "pearson_iii", "pearson-iii"}:
                from scipy.stats import pearson3
                skew, loc, sc = pearson3.fit(cal[positive])
                x = np.asarray(values[month_values == month], dtype=float)
                finite = np.isfinite(x)
                cdf = np.full(x.shape, np.nan)
                cdf[finite & (x <= 0)] = p0
                pos = finite & (x > 0)
                cdf[pos] = p0 + (1-p0) * pearson3.cdf(x[pos], skew, loc=loc, scale=sc)
            else:
                raise ValueError("distribution must be 'gamma' or 'pearson3'.")
            z = norm.ppf(np.clip(cdf, 1e-8, 1-1e-8))
            if clip is not None:
                z = np.clip(z, -float(clip), float(clip))
            out[month_values == month] = z
        return out

    result = xr.apply_ufunc(
        fit_apply, accumulated, months, calibration, cal_months,
        input_core_dims=[[dim], [dim], [dim], [dim]], output_core_dims=[[dim]],
        vectorize=True, dask="parallelized", output_dtypes=[float],
    ).transpose(*precipitation.dims)
    result.name = "spi"
    result.attrs = dict(precipitation.attrs)
    result.attrs.update({
        "standard_name": "standardized_precipitation_index",
        "climatevar:scale": scale,
        "climatevar:distribution": distribution,
        "climatevar:calibration_start": calibration_start or "full_record",
        "climatevar:calibration_end": calibration_end or "full_record",
        "climatevar:min_calibration_samples": int(min_samples),
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
    clip: float | None = 8.0,
) -> xr.DataArray:
    """Calculate SPEI from accumulated climatic water balance ``P - PET``."""
    scale = _validate_scale(scale)
    if dim not in precipitation.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the data.")
    if precipitation.dims != potential_evapotranspiration.dims or precipitation.sizes != potential_evapotranspiration.sizes:
        raise ValueError("precipitation and potential_evapotranspiration must have matching dimensions and sizes.")
    water_balance = precipitation - potential_evapotranspiration
    accumulated = water_balance.rolling({dim: scale}, min_periods=scale).sum()
    calibration = accumulated.sel({dim: slice(calibration_start, calibration_end)}) if calibration_start or calibration_end else accumulated
    months = accumulated[dim].dt.month
    cal_months = calibration[dim].dt.month

    def fit_apply(values, month_values, cal_values, cal_month_values):
        out = np.full(values.shape, np.nan, dtype=float)
        for month in range(1, 13):
            cal = np.asarray(cal_values[cal_month_values == month], dtype=float)
            cal = cal[np.isfinite(cal)]
            if cal.size < min_samples:
                continue
            shape, loc, sc = fisk.fit(cal)
            x = np.asarray(values[month_values == month], dtype=float)
            finite = np.isfinite(x)
            z = np.full(x.shape, np.nan)
            z[finite] = norm.ppf(np.clip(fisk.cdf(x[finite], shape, loc=loc, scale=sc), 1e-8, 1-1e-8))
            if clip is not None:
                z = np.clip(z, -float(clip), float(clip))
            out[month_values == month] = z
        return out

    result = xr.apply_ufunc(
        fit_apply, accumulated, months, calibration, cal_months,
        input_core_dims=[[dim], [dim], [dim], [dim]], output_core_dims=[[dim]],
        vectorize=True, dask="parallelized", output_dtypes=[float],
    ).transpose(*precipitation.dims)
    result.name = "spei"
    result.attrs = {
        "standard_name": "standardized_precipitation_evapotranspiration_index",
        "climatevar:scale": scale,
        "climatevar:distribution": "log-logistic (Fisk)",
        "climatevar:calibration_start": calibration_start or "full_record",
        "climatevar:calibration_end": calibration_end or "full_record",
        "climatevar:min_calibration_samples": int(min_samples),
    }
    return result


def spi_like(precip: xr.DataArray, dim: str = "time", scale: int = 1) -> xr.DataArray:
    """Return a transparent standardized rolling precipitation diagnostic.

    This is deliberately not formal fitted-distribution SPI.
    """
    scale = _validate_scale(scale)
    if dim not in precip.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the data.")
    accumulated = precip.rolling({dim: scale}, min_periods=scale).sum()
    return (accumulated - accumulated.mean(dim=dim, skipna=True)) / accumulated.std(dim=dim, skipna=True, ddof=1)


def drought_category(index: xr.DataArray) -> xr.DataArray:
    """Classify standardized drought/wetness index values."""
    return xr.apply_ufunc(
        lambda x: np.select(
            [x <= -2, x <= -1.5, x <= -1, x < 1, x < 1.5, x < 2],
            ["extreme_drought", "severe_drought", "moderate_drought", "near_normal", "moderately_wet", "severely_wet"],
            default="extreme_wet",
        ),
        index, vectorize=True, output_dtypes=[str],
    )


def fit_quality(index: xr.DataArray) -> xr.Dataset:
    """Return basic diagnostics for a standardized index."""
    values = index.values[np.isfinite(index.values)]
    if values.size < 3:
        return xr.Dataset({"n": xr.DataArray(int(values.size)), "mean": xr.DataArray(np.nan), "std": xr.DataArray(np.nan), "ks_pvalue": xr.DataArray(np.nan)})
    ks = kstest(values, "norm")
    return xr.Dataset({
        "n": xr.DataArray(int(values.size)),
        "mean": xr.DataArray(float(np.mean(values))),
        "std": xr.DataArray(float(np.std(values, ddof=1))),
        "ks_pvalue": xr.DataArray(float(ks.pvalue)),
    })
