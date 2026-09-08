"""Deterministic continuous verification metrics."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _validate_pair(observation: xr.DataArray, prediction: xr.DataArray) -> None:
    if observation.dims != prediction.dims or observation.sizes != prediction.sizes:
        raise ValueError("observation and prediction must have matching dimensions and sizes.")


def bias(observation: xr.DataArray, prediction: xr.DataArray, dim: str | None = None):
    """Mean error: prediction minus observation."""
    _validate_pair(observation, prediction)
    return (prediction - observation).mean(dim=dim, skipna=True)


def mae(observation: xr.DataArray, prediction: xr.DataArray, dim: str | None = None):
    """Mean absolute error."""
    _validate_pair(observation, prediction)
    return np.abs(prediction - observation).mean(dim=dim, skipna=True)


def rmse(observation: xr.DataArray, prediction: xr.DataArray, dim: str | None = None):
    """Root mean square error."""
    _validate_pair(observation, prediction)
    return np.sqrt(((prediction - observation) ** 2).mean(dim=dim, skipna=True))


def correlation(observation: xr.DataArray, prediction: xr.DataArray, dim: str = "time"):
    """Pearson correlation coefficient along ``dim``."""
    _validate_pair(observation, prediction)
    return xr.corr(observation, prediction, dim=dim)


def nse(observation: xr.DataArray, prediction: xr.DataArray, dim: str = "time"):
    """Nash-Sutcliffe efficiency along ``dim``."""
    _validate_pair(observation, prediction)
    numerator = ((prediction - observation) ** 2).sum(dim=dim, skipna=True)
    denominator = ((observation - observation.mean(dim=dim, skipna=True)) ** 2).sum(dim=dim, skipna=True)
    return 1.0 - numerator / denominator
