"""Generalized Extreme Value (GEV) distribution tools."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _fit_1d(values: np.ndarray) -> tuple[float, float, float, int]:
    from scipy.stats import genextreme

    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return np.nan, np.nan, np.nan, y.size
    shape, loc, scale = genextreme.fit(y)
    return float(shape), float(loc), float(scale), int(y.size)


def gev_fit(data: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Fit a GEV distribution along ``dim`` using SciPy maximum likelihood."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    result = xr.apply_ufunc(
        _fit_1d,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, int],
    )
    return xr.Dataset({"shape": result[0], "loc": result[1], "scale": result[2], "n": result[3]})


def gev_return_level(
    shape: xr.DataArray | float,
    loc: xr.DataArray | float,
    scale: xr.DataArray | float,
    return_period: float,
) -> xr.DataArray:
    """Calculate a GEV return level for a specified return period."""
    if return_period <= 1:
        raise ValueError("return_period must be greater than 1.")
    shape = xr.DataArray(shape) if not isinstance(shape, xr.DataArray) else shape
    loc = xr.DataArray(loc) if not isinstance(loc, xr.DataArray) else loc
    scale = xr.DataArray(scale) if not isinstance(scale, xr.DataArray) else scale
    if bool(np.any(scale <= 0)):
        raise ValueError("scale must be positive.")
    p = 1.0 - 1.0 / return_period
    log_term = -np.log(p)
    return xr.where(
        np.abs(shape) < 1e-8,
        loc - scale * np.log(log_term),
        loc + scale / shape * (log_term ** (-shape) - 1.0),
    )


def gev_return_level_ci(
    data: xr.DataArray,
    return_period: float,
    dim: str = "time",
    alpha: float = 0.05,
    n_resamples: int = 1000,
    random_state: int | None = 0,
) -> xr.Dataset:
    """Estimate a GEV return-level confidence interval by parametric bootstrap.

    The fitted GEV is used as the parametric data-generating model. Each
    bootstrap sample is refit, and the requested return level is recalculated.
    The interval is the empirical percentile interval. ``random_state`` makes
    the stochastic result reproducible.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100 for a stable bootstrap interval.")

    def _ci(values):
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if y.size < 3:
            return np.nan, np.nan, np.nan, y.size
        shape, loc, scale = _fit_1d(y)[:3]
        if not np.isfinite(scale) or scale <= 0:
            return np.nan, np.nan, np.nan, y.size
        rng = np.random.default_rng(random_state)
        from scipy.stats import genextreme

        samples = genextreme.rvs(shape, loc=loc, scale=scale, size=(n_resamples, y.size), random_state=rng)
        levels = np.empty(n_resamples)
        for i, sample in enumerate(samples):
            bs, bl, bc, _ = _fit_1d(sample)
            levels[i] = float(gev_return_level(bs, bl, bc, return_period).item()) if np.isfinite(bc) and bc > 0 else np.nan
        levels = levels[np.isfinite(levels)]
        if levels.size < max(50, n_resamples // 2):
            return np.nan, np.nan, np.nan, y.size
        observed = float(gev_return_level(shape, loc, scale, return_period).item())
        return observed, float(np.quantile(levels, alpha / 2)), float(np.quantile(levels, 1 - alpha / 2)), y.size

    result = xr.apply_ufunc(
        _ci,
        data,
        input_core_dims=[[dim]],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float, int],
    )
    return xr.Dataset({"return_level": result[0], "ci_lower": result[1], "ci_upper": result[2], "n": result[3]})
