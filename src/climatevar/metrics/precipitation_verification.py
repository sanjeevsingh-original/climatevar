"""Advanced precipitation verification metrics."""
from __future__ import annotations
import numpy as np
import xarray as xr

def _safe_divide(a, b):
    return xr.where(b != 0, a / b, np.nan)

def threat_score(observation, prediction, threshold=1.0, dim=None):
    """Threat/critical-success score: H/(H+M+F)."""
    o = observation >= threshold; p = prediction >= threshold
    h = (o & p).sum(dim=dim); m = (o & ~p).sum(dim=dim); f = (~o & p).sum(dim=dim)
    return _safe_divide(h, h + m + f)

def equitable_threat_score(observation, prediction, threshold=1.0, dim=None):
    """Equitable Threat Score with random-hit correction."""
    o = observation >= threshold; p = prediction >= threshold
    h = (o & p).sum(dim=dim); m = (o & ~p).sum(dim=dim); f = (~o & p).sum(dim=dim)
    n = h + m + f + (~o & ~p).sum(dim=dim)
    hr = _safe_divide((h + m) * (h + f), n)
    return _safe_divide(h - hr, h + m + f - hr)

def frequency_bias(observation, prediction, threshold=1.0, dim=None):
    """Forecast event frequency divided by observed event frequency."""
    o = observation >= threshold; p = prediction >= threshold
    return _safe_divide(p.sum(dim=dim), o.sum(dim=dim))

def bias_ratio(observation, prediction, dim=None):
    """Multiplicative precipitation bias ratio, prediction/observation."""
    return _safe_divide(prediction.sum(dim=dim, skipna=True), observation.sum(dim=dim, skipna=True))

def brier_score(observation, probability, dim=None):
    """Brier score for probabilistic precipitation-event forecasts."""
    if bool(((probability < 0) | (probability > 1)).any()):
        raise ValueError("probability must be in [0, 1].")
    return ((probability - observation) ** 2).mean(dim=dim, skipna=True)

def reliability_components(observation, probability, bins=10):
    """Return reliability, resolution and uncertainty for fixed probability bins."""
    if bins < 2: raise ValueError("bins must be at least 2.")
    p = np.asarray(probability.values, float).ravel(); y = np.asarray(observation.values, float).ravel()
    ok = np.isfinite(p) & np.isfinite(y); p, y = p[ok], y[ok]
    if np.any((p < 0) | (p > 1)): raise ValueError("probability must be in [0, 1].")
    edges = np.linspace(0, 1, bins + 1); clim = y.mean() if y.size else np.nan
    rel = res = 0.0; rows = []
    for i in range(bins):
        idx = (p >= edges[i]) & ((p <= edges[i+1]) if i == bins-1 else (p < edges[i+1])); n = int(idx.sum())
        if n == 0: rows.append(((edges[i]+edges[i+1])/2, np.nan, np.nan, 0)); continue
        pf, of = p[idx].mean(), y[idx].mean(); rel += n*(pf-of)**2; res += n*(of-clim)**2; rows.append(((edges[i]+edges[i+1])/2,pf,of,n))
    n = max(y.size, 1)
    return xr.Dataset({"reliability": rel/n, "resolution": res/n, "uncertainty": clim*(1-clim), "bin_probability": ("bin", [r[0] for r in rows]), "forecast_probability": ("bin", [r[1] for r in rows]), "observed_frequency": ("bin", [r[2] for r in rows]), "n": ("bin", [r[3] for r in rows])})

def fractions_skill_score(observation, prediction, threshold=1.0, window=3, dim=None):
    """Fractions Skill Score on regular spatial grids using square neighborhoods."""
    if window < 1 or window % 2 == 0: raise ValueError("window must be a positive odd integer.")
    obs = (observation >= threshold).astype(float); pred = (prediction >= threshold).astype(float)
    spatial = [d for d in obs.dims if d != dim]
    if len(spatial) < 2: raise ValueError("FSS requires at least two spatial dimensions.")
    ydim, xdim = spatial[-2:]
    fo = obs.rolling({ydim: window, xdim: window}, center=True, min_periods=1).mean()
    fp = pred.rolling({ydim: window, xdim: window}, center=True, min_periods=1).mean()
    mse = ((fp-fo)**2).mean(dim=[ydim, xdim]); ref = (fp**2+fo**2).mean(dim=[ydim, xdim])
    return 1 - _safe_divide(mse, ref)

__all__ = ["threat_score", "equitable_threat_score", "frequency_bias", "bias_ratio", "brier_score", "reliability_components", "fractions_skill_score"]
