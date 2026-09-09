"""Configurable scientific labels for precipitation-model errors."""
from __future__ import annotations

import numpy as np
import xarray as xr


def intensity_class(
    observation: xr.DataArray,
    thresholds=(1.0, 10.0, 20.0, 50.0, 100.0),
    labels=("dry", "light", "moderate", "heavy", "very_heavy", "extreme"),
) -> xr.DataArray:
    """Classify observed daily precipitation intensity.

    Defaults are common operational bins and are deliberately configurable;
    users should select thresholds appropriate to their study region and aim.
    """
    if len(thresholds) != len(labels) - 1 or any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be strictly increasing and one shorter than labels")
    return xr.apply_ufunc(
        lambda x: np.asarray(labels, dtype=object)[np.digitize(x, thresholds)],
        observation,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[object],
    ).rename("intensity_class")


def error_class(
    observation: xr.DataArray,
    prediction: xr.DataArray,
    *,
    relative_thresholds=(0.25, 0.50),
    absolute_tolerance=1.0,
    labels=("underestimate_severe", "underestimate_moderate", "near_zero", "overestimate_moderate", "overestimate_severe"),
) -> xr.DataArray:
    """Assign deterministic post-hoc rainfall error classes.

    ``prediction - observation`` defines the signed error. Relative error is
    used when observed rainfall exceeds ``absolute_tolerance``; otherwise an
    absolute tolerance prevents unstable ratios for dry/near-dry cases.
    This is a *diagnostic* target and therefore may use the observation.
    """
    if len(relative_thresholds) != 2 or not (0 < relative_thresholds[0] < relative_thresholds[1]):
        raise ValueError("relative_thresholds must be (moderate, severe) with 0 < moderate < severe")
    if absolute_tolerance < 0:
        raise ValueError("absolute_tolerance must be non-negative")
    if len(labels) != 5:
        raise ValueError("labels must contain five classes")
    obs, pred = xr.align(observation, prediction, join="exact")
    err = pred - obs
    rel = xr.where(np.abs(obs) > absolute_tolerance, err / np.abs(obs), err)
    moderate, severe = relative_thresholds
    out = xr.full_like(err, labels[2], dtype=object)
    out = xr.where(rel <= -severe, labels[0], out)
    out = xr.where((rel < -moderate) & (rel > -severe), labels[1], out)
    out = xr.where((rel > moderate) & (rel < severe), labels[3], out)
    out = xr.where(rel >= severe, labels[4], out)
    return out.rename("error_class")
