"""Peaks-over-threshold (POT) analysis with a generalized Pareto model."""

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
    result = xr.apply_ufunc(_fit_gpd_1d, excesses, input_core_dims=[[dim]],
                            output_core_dims=[[], [], []], vectorize=True,
                            dask="parallelized", output_dtypes=[float, float, int])
    return xr.Dataset({"shape": result[0], "scale": result[1], "n_exceedances": result[2]})


def pot_exceedances(data: xr.DataArray, threshold: float, dim: str = "time") -> xr.DataArray:
    """Return positive excesses above ``threshold``; other values are NaN."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    return (data - threshold).where(data > threshold)


def decluster_exceedances(data: xr.DataArray, threshold: float, run_length: int = 3,
                          dim: str = "time") -> xr.DataArray:
    """Extract cluster peaks using a runs rule."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if run_length < 1:
        raise ValueError("run_length must be at least 1.")
    y = np.asarray(data.values, dtype=float)
    if y.ndim != 1:
        raise ValueError("decluster_exceedances currently requires a one-dimensional series.")
    out = np.full(y.shape, np.nan, dtype=float)
    exceed = np.isfinite(y) & (y > threshold)
    i, n = 0, y.size
    while i < n:
        if not exceed[i]:
            i += 1
            continue
        end, gap = i + 1, 0
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


def _threshold_table(data, thresholds):
    y = _as_1d(data)
    n = y.size
    counts = np.array([np.sum(y > u) for u in thresholds], dtype=int)
    rates = counts / n if n else np.full(len(thresholds), np.nan)
    shapes = np.full(len(thresholds), np.nan)
    scales = np.full(len(thresholds), np.nan)
    mean_excess = np.full(len(thresholds), np.nan)
    for j, u in enumerate(thresholds):
        excess = y[y > u] - u
        if excess.size >= 5:
            shapes[j], scales[j], _ = _fit_gpd_1d(excess)
            mean_excess[j] = np.mean(excess)
    return counts, rates, shapes, scales, mean_excess


def threshold_diagnostics(data: xr.DataArray, thresholds, dim: str = "time") -> xr.Dataset:
    """Diagnose threshold choice using exceedance rate, mean excess and GPD stability."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0 or np.any(~np.isfinite(thresholds)):
        raise ValueError("thresholds must be a non-empty one-dimensional finite sequence.")
    counts, rates, shapes, scales, mean_excess = _threshold_table(data.values, thresholds)
    return xr.Dataset({"threshold": ("threshold", thresholds), "n_exceedances": ("threshold", counts),
                       "exceedance_rate": ("threshold", rates), "shape": ("threshold", shapes),
                       "scale": ("threshold", scales), "mean_excess": ("threshold", mean_excess)})


def _return_level(shape, scale, threshold, rate, return_period):
    if return_period <= 0 or rate <= 0 or scale <= 0:
        return np.nan
    target = rate * return_period
    # Standard POT return levels above the threshold require a return period
    # longer than the mean recurrence interval of threshold exceedances.
    if target <= 1:
        return np.nan
    if shape < 0 and target ** shape >= 1:
        return np.nan
    if abs(shape) < 1e-8:
        return threshold + scale * np.log(target)
    return threshold + scale / shape * (target ** shape - 1.0)


def pot_return_level(threshold: float, shape: float, scale: float,
                     exceedance_rate: float, return_period: float) -> float:
    """Return level for a stationary POT model.

    ``exceedance_rate`` and ``return_period`` must use the same observation
    unit. The return period must exceed the threshold exceedance recurrence
    interval so the reported level is above the threshold.
    """
    return float(_return_level(shape, scale, threshold, exceedance_rate, return_period))


def pot_return_level_ci(data: xr.DataArray, threshold: float, return_period: float,
                        dim: str = "time", decluster_run_length: int | None = None,
                        n_resamples: int = 1000, alpha: float = 0.05,
                        random_state: int | None = 0) -> xr.Dataset:
    """Fit POT-GPD and estimate return-level uncertainty by bootstrap."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100.")
    if return_period <= 0:
        raise ValueError("return_period must be positive.")
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
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


def pot_threshold_sensitivity(data: xr.DataArray, thresholds, return_period: float,
                              dim: str = "time") -> xr.Dataset:
    """Evaluate POT parameters and return level across candidate thresholds.

    This is a sensitivity analysis, not an automatic threshold selector. Stable
    parameter estimates and a defensible mean-residual-life region should be
    assessed before choosing a threshold.
    """
    if return_period <= 0:
        raise ValueError("return_period must be positive.")
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or thresholds.size == 0 or np.any(~np.isfinite(thresholds)):
        raise ValueError("thresholds must be a non-empty one-dimensional finite sequence.")
    counts, rates, shapes, scales, mean_excess = _threshold_table(data.values, thresholds)
    levels = np.array([_return_level(xi, sig, u, rate, return_period)
                       for u, xi, sig, rate in zip(thresholds, shapes, scales, rates)])
    return xr.Dataset({"threshold": ("threshold", thresholds), "n_exceedances": ("threshold", counts),
                       "exceedance_rate": ("threshold", rates), "shape": ("threshold", shapes),
                       "scale": ("threshold", scales), "mean_excess": ("threshold", mean_excess),
                       "return_level": ("threshold", levels)})


def gpd_goodness_of_fit(excesses: xr.DataArray, dim: str = "time") -> xr.Dataset:
    """Return PIT/QQ/PP diagnostics and KS/AD statistics for a fitted GPD.

    The tests are descriptive diagnostics: because GPD parameters are estimated
    from the same sample, their p-values are not treated as exact null p-values.
    Use bootstrap or simulation for formal calibrated inference.
    """
    from scipy.stats import anderson, genpareto, kstest
    if dim not in excesses.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    y = _as_1d(excesses.values)
    if y.size < 5:
        return xr.Dataset({"ks_statistic": np.nan, "ks_pvalue": np.nan,
                           "ad_statistic": np.nan, "n_exceedances": y.size})
    shape, scale, _ = _fit_gpd_1d(y)
    pit = genpareto.cdf(y, shape, loc=0, scale=scale)
    ks = kstest(pit, "uniform")
    # Anderson-Darling is computed on the PIT sample; scipy's uniform option
    # is not available on all supported SciPy versions, so use the equivalent
    # exponential transform for the AD diagnostic.
    z = -np.log(np.clip(1.0 - pit, np.finfo(float).eps, 1.0))
    ad = anderson(z, dist="expon")
    order = np.sort(y)
    probs = (np.arange(1, y.size + 1) - 0.5) / y.size
    theoretical = genpareto.ppf(probs, shape, loc=0, scale=scale)
    return xr.Dataset({"ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                       "ad_statistic": float(ad.statistic), "n_exceedances": y.size,
                       "pit": ("exceedance", np.sort(pit)),
                       "qq_observed": ("exceedance", order),
                       "qq_theoretical": ("exceedance", theoretical)})


__all__ = ["gpd_fit", "pot_exceedances", "decluster_exceedances", "threshold_diagnostics",
           "pot_threshold_sensitivity", "gpd_goodness_of_fit", "pot_return_level", "pot_return_level_ci"]
