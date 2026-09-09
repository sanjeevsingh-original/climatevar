"""Peaks-over-threshold (POT) analysis with a generalized Pareto model.

The implementation follows the standard POT formulation: exceedances above a
high threshold are modeled with a GPD, while dependent exceedances can be
reduced to cluster peaks using a runs-based declustering rule.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _as_1d(values):
    y = np.asarray(values, dtype=float).ravel()
    return y[np.isfinite(y)]


def _fit_gpd_1d(excesses):
    from scipy.stats import genpareto

    y = _as_1d(excesses)
    if y.size < 5:
        return np.nan, np.nan, y.size
    if np.any(y < 0):
        raise ValueError("GPD excesses must be non-negative.")
    shape, _, scale = genpareto.fit(y, floc=0.0)
    return float(shape), float(scale), int(y.size)


def gpd_fit(excesses: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Fit a GPD to non-negative threshold excesses using maximum likelihood."""
    if dim not in excesses.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    result = xr.apply_ufunc(
        _fit_gpd_1d,
        excesses,
        input_core_dims=[[dim]], output_core_dims=[[], [], []],
        vectorize=True, dask="parallelized", output_dtypes=[float, float, int],
    )
    return xr.Dataset({"shape": result[0], "scale": result[1], "n_exceedances": result[2]})


def pot_exceedances(data: xr.DataArray, threshold: float, dim: str = "time") -> xr.DataArray:
    """Return positive excesses above ``threshold``; values at/below threshold are NaN."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return (data - threshold).where(data > threshold)


def decluster_exceedances(
    data: xr.DataArray,
    threshold: float,
    run_length: int = 3,
    dim: str = "time",
) -> xr.DataArray:
    """Extract cluster peaks from threshold exceedances using a runs rule.

    Consecutive exceedances separated by fewer than ``run_length`` non-exceeding
    observations belong to the same cluster. The maximum value from each
    cluster is retained, with all other observations set to NaN.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if run_length < 1:
        raise ValueError("run_length must be at least 1.")
    y = np.asarray(data.values, dtype=float)
    if y.ndim != 1:
        raise ValueError("decluster_exceedances currently requires a one-dimensional series.")
    out = np.full(y.shape, np.nan, dtype=float)
    exceed = np.isfinite(y) & (y > threshold)
    i = 0
    n = y.size
    while i < n:
        if not exceed[i]:
            i += 1
            continue
        end = i + 1
        gap = 0
        while end < n:
            if exceed[end]:
                gap = 0
            else:
                gap += 1
                if gap >= run_length:
                    break
            end += 1
        stop = end - gap if end < n else n
        segment = y[i:stop]
        valid = np.isfinite(segment) & (segment > threshold)
        if np.any(valid):
            local = np.where(valid)[0][np.argmax(segment[valid])]
            out[i + local] = segment[local]
        i = max(end, i + 1)
    return xr.DataArray(out, coords=data.coords, dims=data.dims, attrs=data.attrs, name=data.name)


def threshold_diagnostics(
    data: xr.DataArray,
    thresholds,
    dim: str = "time",
) -> xr.Dataset:
    """Compute threshold diagnostics: exceedance count/rate and GPD parameter stability."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0 or np.any(~np.isfinite(thresholds)):
        raise ValueError("thresholds must be a non-empty one-dimensional finite sequence.")
    y = _as_1d(data.values)
    n = y.size
    counts = np.array([np.sum(y > u) for u in thresholds], dtype=int)
    rates = counts / n if n else np.full(thresholds.size, np.nan)
    shapes = np.full(thresholds.size, np.nan)
    scales = np.full(thresholds.size, np.nan)
    for j, u in enumerate(thresholds):
        if counts[j] >= 5:
            shapes[j], scales[j], _ = _fit_gpd_1d(y[y > u] - u)
    return xr.Dataset({"threshold": ("threshold", thresholds), "n_exceedances": ("threshold", counts),
                       "exceedance_rate": ("threshold", rates), "shape": ("threshold", shapes),
                       "scale": ("threshold", scales)})


def _return_level(shape, scale, threshold, rate, return_period):
    if return_period <= 0 or rate <= 0 or scale <= 0:
        return np.nan
    target = rate * return_period
    if shape < 0 and target <= 0:
        return np.nan
    if abs(shape) < 1e-8:
        return threshold + scale * np.log(target)
    return threshold + scale / shape * (target ** shape - 1.0)


def pot_return_level(
    threshold: float,
    shape: float,
    scale: float,
    exceedance_rate: float,
    return_period: float,
) -> float:
    """Return level for a stationary POT model.

    ``exceedance_rate`` is the expected number of independent threshold
    exceedances per observation unit, and ``return_period`` uses the same unit.
    For daily data, for example, a rate per day requires a return period in
    days; for annualized results use an annualized rate and years.
    """
    return float(_return_level(shape, scale, threshold, exceedance_rate, return_period))


def pot_return_level_ci(
    data: xr.DataArray,
    threshold: float,
    return_period: float,
    dim: str = "time",
    decluster_run_length: int | None = None,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = 0,
) -> xr.Dataset:
    """Fit a POT-GPD model and estimate return-level uncertainty by bootstrap.

    The bootstrap resamples independent excesses with replacement and refits
    the GPD. If ``decluster_run_length`` is supplied, cluster peaks are used
    and the exceedance rate is based on the retained independent peaks.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100.")
    if return_period <= 0:
        raise ValueError("return_period must be positive.")

    y = np.asarray(data.values, dtype=float)
    if y.ndim != 1:
        raise ValueError("pot_return_level_ci currently requires a one-dimensional series.")
    if decluster_run_length is not None:
        peaks = decluster_exceedances(data, threshold, decluster_run_length, dim=dim).values
        series = peaks[np.isfinite(peaks)]
    else:
        series = y[np.isfinite(y) & (y > threshold)]
    excess = series - threshold
    n_obs = np.sum(np.isfinite(y))
    if excess.size < 5 or n_obs == 0:
        return xr.Dataset({"return_level": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                           "shape": np.nan, "scale": np.nan, "n_exceedances": excess.size,
                           "exceedance_rate": np.nan})
    shape, scale, n_exc = _fit_gpd_1d(excess)
    rate = n_exc / n_obs
    observed = _return_level(shape, scale, threshold, rate, return_period)
    rng = np.random.default_rng(random_state)
    from scipy.stats import genpareto
    levels = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        sample = rng.choice(excess, size=n_exc, replace=True)
        bs, bc, _ = _fit_gpd_1d(sample)
        levels[i] = _return_level(bs, bc, threshold, rate, return_period)
    levels = levels[np.isfinite(levels)]
    if levels.size < max(50, n_resamples // 2):
        lo = hi = np.nan
    else:
        lo, hi = np.quantile(levels, [alpha / 2, 1 - alpha / 2])
    return xr.Dataset({"return_level": observed, "ci_lower": lo, "ci_upper": hi,
                       "shape": shape, "scale": scale, "n_exceedances": n_exc,
                       "exceedance_rate": rate})


__all__ = ["gpd_fit", "pot_exceedances", "decluster_exceedances", "threshold_diagnostics", "pot_return_level", "pot_return_level_ci"]
