"""Non-parametric trend diagnostics for climate time series."""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import kendalltau, norm


def _clean(values):
    y = np.asarray(values, dtype=float)
    return y[np.isfinite(y)]


def _mk_stats(y):
    n = y.size
    if n < 2:
        return np.nan, np.nan, np.nan, n
    tau, p = kendalltau(np.arange(n), y)
    s = np.sign(y[np.newaxis, :] - y[:, np.newaxis])
    s = float(s[np.triu_indices(n, k=1)].sum())
    return s, float(tau), float(p), n


def _sen_1d(values):
    y = _clean(values)
    n = y.size
    if n < 2:
        return np.nan
    slopes = []
    for j in range(1, n):
        slopes.extend((y[j] - y[:j]) / np.arange(1, j + 1))
    return float(np.median(slopes))


def _lag_autocorrelation(y, max_lag):
    n = y.size
    if n < 3:
        return np.array([])
    max_lag = min(int(max_lag), n - 2)
    z = y - np.mean(y)
    denom = np.sum(z * z)
    if denom == 0:
        return np.zeros(max_lag)
    return np.array([np.sum(z[k:] * z[:-k]) / denom for k in range(1, max_lag + 1)])


def _mk_variance(y):
    n = y.size
    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    return (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0


def _modified_mk_1d(values, alpha=0.05, max_lag=None):
    y = _clean(values)
    n = y.size
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan, float(n)

    s, tau, _, _ = _mk_stats(y)
    # Yue & Wang-style effective sample size correction.  Estimate
    # persistence from the detrended residuals so a monotonic trend does not
    # masquerade as serial correlation.
    slope = _sen_1d(y)
    residual = y - slope * np.arange(n)
    max_lag = max_lag or min(n - 1, int(np.sqrt(n) * 3))
    r = _lag_autocorrelation(residual, max_lag)
    neff_factor = 1.0 + (2.0 / n) * np.sum((n - np.arange(1, len(r) + 1)) * r)
    neff_factor = max(neff_factor, 1e-6)
    n_eff = float(np.clip(n / neff_factor, 1.0, n))
    var_s = _mk_variance(y) * n / n_eff
    if var_s <= 0:
        return s, tau, np.nan, n_eff, float(n)
    z = (s - 1.0) / np.sqrt(var_s) if s > 0 else (s + 1.0) / np.sqrt(var_s) if s < 0 else 0.0
    p = float(2.0 * norm.sf(abs(z)))
    return s, tau, p, n_eff, float(n)


def mann_kendall(data: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Compute classical Mann-Kendall statistic, tau, p-value and sample size."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    result = xr.apply_ufunc(_mk_stats, data, input_core_dims=[[dim]], output_core_dims=[[], [], [], []], vectorize=True, dask="parallelized", output_dtypes=[float, float, float, int])
    return xr.Dataset({"s": result[0], "tau": result[1], "pvalue": result[2], "n": result[3]})


def modified_mann_kendall(data: xr.DataArray, dim: str = "time", max_lag: int | None = None) -> xr.Dataset:
    """Modified Mann-Kendall test with serial-correlation correction.

    Uses a Yue-Wang-style effective-sample-size variance inflation after
    detrending with Sen's slope. This addresses the independence assumption
    of classical MK without altering the observed series.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    result = xr.apply_ufunc(
        _modified_mk_1d, data,
        kwargs={"max_lag": max_lag},
        input_core_dims=[[dim]], output_core_dims=[[], [], [], [], []],
        vectorize=True, dask="parallelized", output_dtypes=[float, float, float, float, float],
    )
    return xr.Dataset({"s": result[0], "tau": result[1], "pvalue": result[2], "n_eff": result[3], "n": result[4]})


def trend_free_prewhitening(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Apply trend-free pre-whitening using Sen's slope and lag-1 AR(1)."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")

    def tfpw(values):
        y = _clean(values)
        if y.size < 3:
            return np.full(np.asarray(values).shape, np.nan)
        slope = _sen_1d(y)
        t = np.arange(y.size, dtype=float)
        detrended = y - slope * t
        z = detrended - detrended.mean()
        denom = np.sum(z[:-1] ** 2)
        r1 = np.sum(z[1:] * z[:-1]) / denom if denom > 0 else 0.0
        residual = detrended[1:] - r1 * detrended[:-1]
        return np.r_[residual[0], residual + slope * np.arange(1, y.size)]

    return xr.apply_ufunc(tfpw, data, input_core_dims=[[dim]], output_core_dims=[[dim]], vectorize=True, dask="parallelized", output_dtypes=[float])


def sens_slope(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Estimate Sen's slope per observation step along ``dim``."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    return xr.apply_ufunc(_sen_1d, data, input_core_dims=[[dim]], output_core_dims=[[]], vectorize=True, dask="parallelized", output_dtypes=[float])
