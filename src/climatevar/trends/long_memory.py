"""Diagnostics for persistence and possible long-memory scaling."""
from __future__ import annotations

import numpy as np
import xarray as xr


def _dfa_1d(values, order=1, min_scale=8, max_scale=None, n_scales=20):
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 20:
        return np.nan, np.nan, float(n)
    if order < 0 or order > 3:
        raise ValueError("order must be between 0 and 3")
    max_scale = max_scale or n // 4
    scales = np.unique(np.floor(np.geomspace(min_scale, max_scale, n_scales)).astype(int))
    profile = np.cumsum(y - y.mean())
    fluct = []
    used = []
    for s in scales:
        if s <= order + 2 or n // s < 2:
            continue
        m = n // s
        rms = []
        for k in range(m):
            seg = profile[k * s:(k + 1) * s]
            x = np.arange(s, dtype=float)
            coeff = np.polyfit(x, seg, order)
            detrended = seg - np.polyval(coeff, x)
            rms.append(np.mean(detrended ** 2))
        f = np.sqrt(np.mean(rms))
        if np.isfinite(f) and f > 0:
            used.append(s)
            fluct.append(f)
    if len(used) < 4:
        return np.nan, np.nan, float(n)
    lx, ly = np.log10(used), np.log10(fluct)
    slope, intercept = np.polyfit(lx, ly, 1)
    fitted = slope * lx + intercept
    ss_res = np.sum((ly - fitted) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(r2), float(n)


def dfa_hurst(
    data: xr.DataArray,
    dim: str = "time",
    *,
    order: int = 1,
    min_scale: int = 8,
    max_scale: int | None = None,
    n_scales: int = 20,
) -> xr.Dataset:
    """Estimate a DFA scaling exponent and fit quality.

    The exponent is a persistence diagnostic, not a hypothesis test for long
    memory. Values near 0.5 are broadly consistent with short-memory scaling;
    persistent scaling above 0.5 should be investigated with additional
    diagnostics and sensitivity tests before being labelled long-range
    dependence.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if min_scale < 4:
        raise ValueError("min_scale must be at least 4")
    result = xr.apply_ufunc(
        _dfa_1d,
        data,
        kwargs={"order": order, "min_scale": min_scale, "max_scale": max_scale, "n_scales": n_scales},
        input_core_dims=[[dim]],
        output_core_dims=[[], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float, float, float],
    )
    return xr.Dataset({"dfa_exponent": result[0], "r2": result[1], "n": result[2]}).assign_attrs(
        method="detrended fluctuation analysis",
        polynomial_order=order,
        min_scale=min_scale,
        max_scale=max_scale if max_scale is not None else "automatic",
        n_scales=n_scales,
        interpretation="DFA exponent is a persistence diagnostic; >0.5 is not by itself proof of long-range dependence.",
    )


__all__ = ["dfa_hurst"]
