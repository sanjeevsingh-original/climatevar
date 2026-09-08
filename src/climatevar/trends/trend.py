"""Non-parametric trend diagnostics for climate time series."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _mk_1d(values: np.ndarray) -> tuple[float, float, float, int]:
    """Return Kendall S, tau, two-sided p-value and valid sample size."""
    from scipy.stats import kendalltau

    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 2:
        return np.nan, np.nan, np.nan, n
    diffs = y[np.newaxis, :] - y[:, np.newaxis]
    s = float(np.sign(diffs[np.triu_indices(n, k=1)]).sum())
    tau, p = kendalltau(np.arange(n), y, nan_policy="omit")
    return s, float(tau), float(p), n


def mann_kendall(data: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Compute the Mann-Kendall trend statistic along ``dim``.

    Returns a dataset containing ``s``, Kendall ``tau``, two-sided ``pvalue``
    and the number of valid observations ``n``. The implementation uses the
    supplied observation order; for irregular time axes, callers should first
    decide whether the series is suitable for this formulation.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    result = xr.apply_ufunc(
        _mk_1d,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, int],
    )
    return xr.Dataset({"s": result[0], "tau": result[1], "pvalue": result[2], "n": result[3]})


def _sen_1d(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 2:
        return np.nan
    slopes = (y[np.newaxis, :] - y[:, np.newaxis]) / (
        np.arange(n)[np.newaxis, :] - np.arange(n)[:, np.newaxis]
    )
    return float(np.nanmedian(slopes[np.triu_indices(n, k=1)]))


def sens_slope(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Estimate Sen's slope per observation step along ``dim``.

    The slope is the median of all pairwise slopes. If observations represent
    annual values, the result is therefore in input-units per year.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    return xr.apply_ufunc(
        _sen_1d,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
