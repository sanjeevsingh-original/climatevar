"""Standardized multi-product precipitation verification workflows."""
from __future__ import annotations

import numpy as np
import xarray as xr

from climatevar.metrics import bias, correlation, mae, rmse
from climatevar.metrics.precipitation import contingency_table, f1_score, heidke_skill_score, pod
from climatevar.metrics.precipitation_verification import (
    bias_ratio, equitable_threat_score, frequency_bias, threat_score,
)


def common_grid(reference, candidate, *, method="linear"):
    """Interpolate a regular lat/lon candidate onto the reference grid."""
    rlat = "lat" if "lat" in reference.coords else "latitude"
    rlon = "lon" if "lon" in reference.coords else "longitude"
    clat = "lat" if "lat" in candidate.coords else "latitude"
    clon = "lon" if "lon" in candidate.coords else "longitude"
    if clat not in candidate.coords or clon not in candidate.coords:
        raise ValueError("candidate must contain lat/lon or latitude/longitude")
    return candidate.interp({clat: reference[rlat], clon: reference[rlon]}, method=method)


def align_period(reference, candidate):
    """Restrict products to their exact common coordinates and timestamps."""
    ref, cand = xr.align(reference, candidate, join="inner")
    if "time" not in ref.dims:
        raise ValueError("a time dimension is required")
    return ref, cand


def evaluate_product(reference, candidate, *, name="candidate", threshold=1.0):
    """Return one deterministic and categorical precipitation scorecard."""
    ref, pred = align_period(reference, candidate)
    spatial = tuple(d for d in ref.dims if d != "time")
    reduce_dims = ("time",) + spatial
    out = {
        "dataset": name,
        "n_valid": int(ref.count().values),
        "bias": float(bias(pred, ref).mean()),
        "mae": float(mae(pred, ref).mean()),
        "rmse": float(rmse(pred, ref).mean()),
        "correlation": float(correlation(pred, ref, dim="time").mean()),
        "bias_ratio": float(bias_ratio(pred, ref, dim=reduce_dims)),
    }
    obs_event, pred_event = ref >= threshold, pred >= threshold
    table = contingency_table(obs_event, pred_event, dim=reduce_dims)
    hits = float(table["hits"])
    false_alarms = float(table["false_alarms"])
    out.update({
        "pod": float(pod(table)),
        "far": false_alarms / max(hits + false_alarms, 1e-15),
        "f1": float(f1_score(table)),
        "heidke_skill_score": float(heidke_skill_score(table)),
        "threat_score": float(threat_score(obs_event, pred_event, dim=reduce_dims)),
        "equitable_threat_score": float(equitable_threat_score(obs_event, pred_event, dim=reduce_dims)),
        "frequency_bias": float(frequency_bias(obs_event, pred_event, dim=reduce_dims)),
    })
    return out


def compare_products(reference, products, *, threshold=1.0, common_grid_method=None):
    """Evaluate ERA5/IMERG/IMDAA/WRF/CMIP6-like products with one protocol."""
    rows = []
    for name, product in products.items():
        if common_grid_method is not None:
            product = common_grid(reference, product, method=common_grid_method)
        rows.append(evaluate_product(reference, product, name=name, threshold=threshold))
    return rows


def rank_products(scorecard, *, metrics=None, weights=None):
    """Create a transparent weighted rank; component scores remain available."""
    import pandas as pd
    df = pd.DataFrame(scorecard).copy()
    metrics = metrics or ["bias", "mae", "rmse", "correlation", "pod", "far", "f1", "threat_score"]
    weights = weights or {m: 1.0 for m in metrics}
    higher = {"bias": False, "mae": False, "rmse": False, "correlation": True,
              "pod": True, "far": False, "f1": True, "threat_score": True}
    total = np.zeros(len(df), float)
    wsum = 0.0
    for metric in metrics:
        x = df[metric].to_numpy(float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        score = np.ones_like(x) if hi == lo else (x - lo) / (hi - lo)
        if not higher.get(metric, True):
            score = 1.0 - score
        w = float(weights.get(metric, 1.0))
        total += w * np.nan_to_num(score)
        wsum += w
    df["composite_score"] = total / wsum
    df["rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["rank", "dataset"]).reset_index(drop=True)
