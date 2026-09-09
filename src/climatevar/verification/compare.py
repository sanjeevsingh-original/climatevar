"""Standardized multi-product precipitation verification workflows."""
from __future__ import annotations

import numpy as np
import xarray as xr

from climatevar.metrics import bias, correlation, mae, rmse
from climatevar.metrics.precipitation import f1_score, heidke_skill_score, pod, far
from climatevar.metrics.precipitation_verification import (
    bias_ratio,
    equitable_threat_score,
    frequency_bias,
    threat_score,
)


def _coord_name(data, candidates):
    for name in candidates:
        if name in data.coords:
            return name
    raise ValueError(f"Expected one of {candidates}, found coordinates: {list(data.coords)}")


def common_grid(reference, candidate, *, method="linear"):
    """Interpolate a regular lat/lon candidate onto the reference grid.

    This is a transparent baseline interpolator, not area-conserving
    precipitation remapping. Use the experiment runner's conservative xESMF
    option when precipitation totals must be conserved.
    """
    rlat = _coord_name(reference, ("lat", "latitude"))
    rlon = _coord_name(reference, ("lon", "longitude"))
    clat = _coord_name(candidate, ("lat", "latitude"))
    clon = _coord_name(candidate, ("lon", "longitude"))
    return candidate.interp({clat: reference[rlat], clon: reference[rlon]}, method=method)


def align_period(reference, candidate):
    """Restrict products to their exact common coordinates and timestamps."""
    ref, cand = xr.align(reference, candidate, join="inner")
    if "time" not in ref.dims:
        raise ValueError("a time dimension is required")
    if ref.sizes.get("time", 0) == 0:
        raise ValueError("reference and candidate have no overlapping timestamps")
    return ref, cand


def evaluate_product(reference, candidate, *, name="candidate", threshold=1.0):
    """Return one deterministic and categorical precipitation scorecard."""
    ref, pred = align_period(reference, candidate)
    spatial = tuple(d for d in ref.dims if d != "time")
    reduce_dims = ("time",) + spatial
    out = {
        "dataset": name,
        "n_valid": int(ref.count().values),
        "bias": float(bias(ref, pred).mean()),
        "absolute_bias": float(abs(bias(ref, pred)).mean()),
        "mae": float(mae(ref, pred).mean()),
        "rmse": float(rmse(ref, pred).mean()),
        "correlation": float(correlation(ref, pred, dim="time").mean()),
        "bias_ratio": float(bias_ratio(ref, pred, dim=reduce_dims)),
    }
    out.update({
        "pod": float(pod(ref, pred, threshold=threshold, dim=reduce_dims)),
        "far": float(far(ref, pred, threshold=threshold, dim=reduce_dims)),
        "f1": float(f1_score(ref, pred, threshold=threshold, dim=reduce_dims)),
        "heidke_skill_score": float(heidke_skill_score(ref, pred, threshold=threshold, dim=reduce_dims)),
        "threat_score": float(threat_score(ref, pred, threshold=threshold, dim=reduce_dims)),
        "equitable_threat_score": float(equitable_threat_score(ref, pred, threshold=threshold, dim=reduce_dims)),
        "frequency_bias": float(frequency_bias(ref, pred, threshold=threshold, dim=reduce_dims)),
    })
    return out


def compare_products(reference, products, *, threshold=1.0, common_grid_method=None):
    """Evaluate multiple products with one standardized protocol."""
    rows = []
    for name, product in products.items():
        if common_grid_method is not None:
            product = common_grid(reference, product, method=common_grid_method)
        rows.append(evaluate_product(reference, product, name=name, threshold=threshold))
    return rows


def rank_products(scorecard, *, metrics=None, weights=None):
    """Create a transparent weighted rank; component scores remain available.

    ``absolute_bias`` is used instead of signed bias because both wet and dry
    bias are errors. Min-max normalization is intentionally sample-dependent
    and should be treated as a summary score, not a universal truth metric.
    """
    import pandas as pd

    df = pd.DataFrame(scorecard).copy()
    metrics = metrics or [
        "absolute_bias", "mae", "rmse", "correlation", "pod", "far", "f1", "threat_score"
    ]
    weights = weights or {m: 1.0 for m in metrics}
    higher = {
        "absolute_bias": False,
        "mae": False,
        "rmse": False,
        "correlation": True,
        "pod": True,
        "far": False,
        "f1": True,
        "threat_score": True,
    }
    total = np.zeros(len(df), float)
    wsum = 0.0
    for metric in metrics:
        if metric not in df:
            raise KeyError(f"Metric {metric!r} is not present in the scorecard.")
        x = df[metric].to_numpy(float)
        lo, hi = np.nanmin(x), np.nanmax(x)
        score = np.ones_like(x) if hi == lo else (x - lo) / (hi - lo)
        if not higher.get(metric, True):
            score = 1.0 - score
        w = float(weights.get(metric, 1.0))
        total += w * np.nan_to_num(score)
        wsum += w
    if wsum <= 0:
        raise ValueError("At least one ranking metric must have a positive weight.")
    df["composite_score"] = total / wsum
    df["rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values(["rank", "dataset"]).reset_index(drop=True)
