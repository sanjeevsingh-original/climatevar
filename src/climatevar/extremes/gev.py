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
    """Fit a GEV distribution along ``dim`` using SciPy maximum likelihood.

    SciPy uses the opposite sign convention to some extreme-value literature;
    returned ``shape`` follows SciPy's ``genextreme`` convention. Always report
    the convention when publishing fitted parameters.
    """
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
    """Calculate a GEV return level for a specified return period.

    ``return_period`` is expressed in the same block units as the fitted
    sample (for example, years for annual maxima). For shape near zero the
    Gumbel limit is used to avoid numerical instability.
    """
    if return_period <= 1:
        raise ValueError("return_period must be greater than 1.")
    shape = xr.DataArray(shape) if not isinstance(shape, xr.DataArray) else shape
    loc = xr.DataArray(loc) if not isinstance(loc, xr.DataArray) else loc
    scale = xr.DataArray(scale) if not isinstance(scale, xr.DataArray) else scale
    if np.any(scale <= 0):
        raise ValueError("scale must be positive.")
    p = 1.0 - 1.0 / return_period
    log_term = -np.log(p)
    return xr.where(
        np.abs(shape) < 1e-8,
        loc - scale * np.log(log_term),
        loc + scale / shape * (log_term ** (-shape) - 1.0),
    )
