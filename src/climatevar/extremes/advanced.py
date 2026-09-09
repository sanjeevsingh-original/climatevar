"""Advanced POT inference: threshold uncertainty and non-stationary model comparison."""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import genpareto, chi2


def _clean_pair(excesses, x):
    y = np.asarray(excesses.values if hasattr(excesses, "values") else excesses, float).ravel()
    z = np.asarray(x.values if hasattr(x, "values") else x, float).ravel()
    if y.size != z.size:
        raise ValueError("excesses and covariate must have the same size.")
    ok = np.isfinite(y) & np.isfinite(z)
    y, z = y[ok], z[ok]
    if y.size < 10 or np.any(y < 0):
        raise ValueError("at least 10 non-negative finite excesses are required.")
    zs = (z - z.mean()) / (z.std() or 1.0)
    return y, z, zs


def _stationary_nll(y):
    shape, _, scale = genpareto.fit(y, floc=0.0)
    return float(-np.sum(genpareto.logpdf(y, shape, loc=0.0, scale=scale))), float(shape), float(scale)


def _nonstationary_fit(y, zs):
    def nll(p):
        xi, b0, b1 = p
        sigma = np.exp(b0 + b1 * zs)
        support = 1.0 + xi * y / sigma
        if np.any(support <= 0):
            return 1e100
        return float(-np.sum(genpareto.logpdf(y, xi, loc=0.0, scale=sigma)))
    _, shape0, scale0 = _stationary_nll(y)
    fit = minimize(nll, [shape0, np.log(scale0), 0.0], method="Nelder-Mead", options={"maxiter": 10000})
    if not fit.success:
        raise RuntimeError(f"Non-stationary GPD optimization failed: {fit.message}")
    return fit


def gpd_model_comparison(excesses, x, dim="time"):
    """Compare stationary and log-linear-scale non-stationary GPD models."""
    if hasattr(excesses, "dims") and dim not in excesses.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    y, _, zs = _clean_pair(excesses, x)
    n = y.size
    nll_s, xi, scale = _stationary_nll(y)
    fit = _nonstationary_fit(y, zs)
    nll_ns = float(fit.fun)
    ll_s, ll_ns = -nll_s, -nll_ns
    aic_s, aic_ns = 4 - 2 * ll_s, 6 - 2 * ll_ns
    bic_s, bic_ns = 2 * np.log(n) - 2 * ll_s, 3 * np.log(n) - 2 * ll_ns
    lr = max(0.0, 2 * (ll_ns - ll_s))
    return xr.Dataset({"stationary_shape": xi, "stationary_scale": scale, "nonstationary_shape": float(fit.x[0]), "log_scale_intercept": float(fit.x[1]), "log_scale_covariate": float(fit.x[2]), "stationary_loglik": ll_s, "nonstationary_loglik": ll_ns, "stationary_aic": aic_s, "nonstationary_aic": aic_ns, "stationary_bic": bic_s, "nonstationary_bic": bic_ns, "lr_statistic": lr, "lr_pvalue_asymptotic": float(chi2.sf(lr, 1)), "n_exceedances": n})


def gpd_model_comparison_bootstrap(excesses, x, n_resamples=500, alpha=0.05, random_state=0):
    """Calibrate the stationary-vs-nonstationary likelihood-ratio test by bootstrap."""
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    y, _, zs = _clean_pair(excesses, x)
    base = gpd_model_comparison(xr.DataArray(y, dims="time"), xr.DataArray(zs, dims="time"))
    observed = float(base.lr_statistic)
    _, xi, scale = _stationary_nll(y)
    rng = np.random.default_rng(random_state)
    null = np.full(n_resamples, np.nan)
    for i in range(n_resamples):
        sample = genpareto.rvs(xi, loc=0.0, scale=scale, size=y.size, random_state=rng)
        s_nll, _, _ = _stationary_nll(sample)
        fit = _nonstationary_fit(sample, zs)
        null[i] = max(0.0, 2 * (s_nll - float(fit.fun)))
    p = (1 + np.sum(null >= observed)) / (n_resamples + 1)
    return xr.merge([base, xr.Dataset({"lr_pvalue_bootstrap": float(p), "lr_critical": float(np.quantile(null, 1 - alpha)), "n_resamples": n_resamples, "alpha": alpha})])


def gpd_nonstationary_ci(excesses, x, alpha=0.05, n_resamples=500, random_state=0):
    """Bootstrap confidence intervals for non-stationary GPD parameters."""
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    y, _, zs = _clean_pair(excesses, x)
    fit = _nonstationary_fit(y, zs)
    rng = np.random.default_rng(random_state)
    pars = np.full((n_resamples, 3), np.nan)
    sigma = np.exp(fit.x[1] + fit.x[2] * zs)
    for i in range(n_resamples):
        sample = genpareto.rvs(fit.x[0], loc=0.0, scale=sigma, size=y.size, random_state=rng)
        pars[i] = _nonstationary_fit(sample, zs).x
    good = np.all(np.isfinite(pars), axis=1)
    pars = pars[good]
    lo, hi = np.quantile(pars, [alpha / 2, 1 - alpha / 2], axis=0) if pars.shape[0] >= max(50, n_resamples // 2) else (np.full(3, np.nan), np.full(3, np.nan))
    return xr.Dataset({"shape": float(fit.x[0]), "log_scale_intercept": float(fit.x[1]), "log_scale_covariate": float(fit.x[2]), "shape_ci_lower": float(lo[0]), "shape_ci_upper": float(hi[0]), "log_scale_intercept_ci_lower": float(lo[1]), "log_scale_intercept_ci_upper": float(hi[1]), "log_scale_covariate_ci_lower": float(lo[2]), "log_scale_covariate_ci_upper": float(hi[2]), "n_exceedances": int(y.size), "n_resamples": n_resamples, "n_valid_bootstrap": int(pars.shape[0]), "alpha": alpha})


def pot_threshold_bootstrap(data, thresholds, return_period, n_resamples=500, alpha=0.05, random_state=0):
    """Propagate sampling and threshold-choice uncertainty into return levels."""
    from .pot import _threshold_table, _return_level
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if return_period <= 0:
        raise ValueError("return_period must be positive.")
    t = np.asarray(thresholds, float)
    if t.ndim != 1 or t.size == 0 or np.any(~np.isfinite(t)):
        raise ValueError("thresholds must be a non-empty finite one-dimensional sequence.")
    y = np.asarray(data.values, float).ravel(); y = y[np.isfinite(y)]
    if y.size < 20:
        raise ValueError("at least 20 finite observations are required.")
    rng = np.random.default_rng(random_state); selected = np.full(n_resamples, np.nan); selected_threshold = np.full(n_resamples, np.nan)
    for b in range(n_resamples):
        sample = rng.choice(y, size=y.size, replace=True); counts, rates, shapes, scales, _ = _threshold_table(sample, t); aics = np.full(t.size, np.inf)
        for j, u in enumerate(t):
            e = sample[sample > u] - u
            if e.size >= 5 and np.isfinite(shapes[j]) and np.isfinite(scales[j]):
                ll = np.sum(genpareto.logpdf(e, shapes[j], loc=0.0, scale=scales[j])); aics[j] = 4 - 2 * ll
        j = int(np.argmin(aics))
        if np.isfinite(aics[j]):
            selected_threshold[b] = t[j]; selected[b] = _return_level(shapes[j], scales[j], t[j], rates[j], return_period)
    ok = np.isfinite(selected)
    if ok.sum() < max(50, n_resamples // 2):
        raise RuntimeError("Too few valid bootstrap replicates for threshold uncertainty inference.")
    q = np.quantile(selected[ok], [alpha / 2, 0.5, 1 - alpha / 2]); valid_t = selected_threshold[np.isfinite(selected_threshold)]
    return xr.Dataset({"return_level_median": float(q[1]), "ci_lower": float(q[0]), "ci_upper": float(q[2]), "selected_threshold_median": float(np.median(valid_t)), "n_valid_bootstrap": int(ok.sum()), "n_resamples": n_resamples, "alpha": alpha})


__all__ = ["pot_threshold_bootstrap", "gpd_model_comparison", "gpd_model_comparison_bootstrap", "gpd_nonstationary_ci"]
