"""Feature engineering for rainfall-model error diagnostics."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .labels import error_class, intensity_class


def build_error_features(
    observation: xr.DataArray,
    prediction: xr.DataArray,
    *,
    elevation: xr.DataArray | None = None,
    include_target: bool = True,
    intensity_thresholds=(1.0, 10.0, 20.0, 50.0, 100.0),
    relative_thresholds=(0.25, 0.50),
    absolute_tolerance=1.0,
) -> xr.Dataset:
    """Build a traceable ML-ready error-feature dataset.

    The dataset keeps observation, prediction, signed/absolute/relative error,
    log-ratio, intensity, wet/dry status and calendar features. When
    ``include_target=True`` it also contains ``error_class``. For prospective
    error prediction, set it to False and do not use observation-derived
    variables as model predictors.
    """
    obs, pred = xr.align(observation, prediction, join="exact")
    if obs.dims != pred.dims:
        raise ValueError("observation and prediction must have identical dimensions")
    err = pred - obs
    abs_err = np.abs(err)
    relative_err = xr.where(obs != 0, err / np.abs(obs), np.nan)
    log_ratio = xr.where((obs > 0) & (pred > 0), np.log(pred / obs), np.nan)

    ds = xr.Dataset({
        "observation": obs,
        "prediction": pred,
        "error": err,
        "absolute_error": abs_err,
        "relative_error": relative_err,
        "log_ratio": log_ratio,
        "observed_intensity_class": intensity_class(obs, thresholds=intensity_thresholds),
        "observed_wet": (obs >= 1.0).astype("int8"),
    })
    if "time" in obs.coords:
        ds["month"] = obs["time"].dt.month
        ds["year"] = obs["time"].dt.year
        ds["dayofyear"] = obs["time"].dt.dayofyear
        ds["season"] = xr.DataArray(
            np.select(
                [obs["time"].dt.month.isin([12, 1, 2]), obs["time"].dt.month.isin([3, 4, 5]),
                 obs["time"].dt.month.isin([6, 7, 8]), obs["time"].dt.month.isin([9, 10, 11])],
                ["DJF", "MAM", "JJA", "SON"], default="unknown"
            ),
            coords={"time": obs["time"]}, dims=("time",), name="season",
        )
    if "lat" in obs.coords:
        ds["latitude"] = obs["lat"]
    elif "latitude" in obs.coords:
        ds["latitude"] = obs["latitude"]
    if "lon" in obs.coords:
        ds["longitude"] = obs["lon"]
    elif "longitude" in obs.coords:
        ds["longitude"] = obs["longitude"]
    if elevation is not None:
        elevation, _ = xr.align(elevation, obs, join="exact")
        ds["elevation"] = elevation
    if include_target:
        ds["error_class"] = error_class(
            obs, pred, relative_thresholds=relative_thresholds,
            absolute_tolerance=absolute_tolerance,
        )
    ds.attrs["purpose"] = "diagnostic rainfall model-error tagging"
    ds.attrs["prospective_warning"] = "Do not use observation-derived variables as predictors when forecasting error before observations are available."
    return ds


def to_ml_table(ds: xr.Dataset, *, dropna: bool = True):
    """Convert an error-feature Dataset to a pandas DataFrame.

    Pandas is intentionally optional in climatevar; it is imported only when
    this conversion is requested.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise ImportError("to_ml_table requires pandas; install climatevar[test] or pandas") from exc
    table = ds.to_dataframe().reset_index()
    if dropna:
        table = table.dropna()
    return table
