"""Non-parametric trend diagnostics for climate time series."""
from __future__ import annotations
import numpy as np
import xarray as xr


def _mk_1d(values: np.ndarray) -> tuple[float, float, float, int]:
    from scipy.stats import kendalltau
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 2:
        return np.nan, np.nan, np.nan, n
    diffs = y[np.newaxis, :] - y[:, np.newaxis]
    s = float(np.sign(diffs[np.triu_indices(n, k=1)]).sum())
    tau, p = kendalltau(np.arange(n), y, nan_policy="omit")
    if np.isfinite(tau):
        tau = float(np.clip(tau, -1.0, 1.0))
    return s, tau, float(p), n


def mann_kendall(data: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Compute the Mann-Kendall statistic, tau, p-value and valid sample size."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    result = xr.apply_ufunc(_mk_1d, data, input_core_dims=[[dim]], output_core_dims=[[], [], [], []], vectorize=True, dask="parallelized", output_dtypes=[float, float, float, int])
    return xr.Dataset({"s": result[0], "tau": result[1], "pvalue": result[2], "n": result[3]})


def _sen_1d(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 2:
        return np.nan
    slopes = []
    for j in range(1, n):
        slopes.extend((y[j] - y[:j]) / np.arange(1, j + 1))
    return float(np.median(slopes))


def sens_slope(data: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Estimate Sen's slope per observation step along ``dim``."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    return xr.apply_ufunc(_sen_1d, data, input_core_dims=[[dim]], output_core_dims=[[]], vectorize=True, dask="parallelized", output_dtypes=[float])
