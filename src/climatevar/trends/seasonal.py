"""Seasonal and spatial trend diagnostics for climate data."""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import norm

from .trend import modified_mann_kendall, sens_slope


def _season_months(season):
    names = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    key = str(season).upper()
    if key not in names:
        raise ValueError("season must be one of DJF, MAM, JJA, SON")
    return names[key]


def _season_year_index(time, season):
    months = _season_months(season)
    month = time.dt.month
    year = time.dt.year
    return xr.where(month == months[0], year + 1, year)


def seasonal_series(data: xr.DataArray, season: str, dim: str = "time") -> xr.DataArray:
    """Extract complete seasonal means/totals as a one-value-per-season series."""
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    months = _season_months(season)
    selected = data.where(data[dim].dt.month.isin(months), drop=True)
    season_year = _season_year_index(selected[dim], season).rename("season_year")
    selected = selected.assign_coords(season_year=(dim, season_year.data))
    grouped = selected.groupby("season_year")
    # Sum is appropriate for precipitation totals; users can pass a precomputed
    # seasonal mean for other variables.
    result = grouped.sum(dim=dim, skipna=False)
    counts = grouped.count(dim=dim)
    result = result.where(counts == len(months))
    result = result.rename({"season_year": "year"})
    result.attrs.update(data.attrs)
    result.attrs["season"] = str(season).upper()
    return result


def seasonal_mann_kendall(data: xr.DataArray, season: str, dim: str = "time", max_lag: int | None = None) -> xr.Dataset:
    """Apply autocorrelation-aware MK to a specified climatological season."""
    series = seasonal_series(data, season, dim=dim)
    result = modified_mann_kendall(series, dim="year", max_lag=max_lag)
    result.attrs["season"] = str(season).upper()
    return result


def seasonal_sen_slope(data: xr.DataArray, season: str, dim: str = "time") -> xr.DataArray:
    """Estimate Sen's slope for a specified climatological season."""
    series = seasonal_series(data, season, dim=dim)
    result = sens_slope(series, dim="year")
    result.attrs.update({"season": str(season).upper(), "slope_units": "input units per season-year"})
    return result


def _bh_fdr(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(p)
    out = np.zeros(p.shape, dtype=bool)
    if not valid.any():
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    threshold = alpha * np.arange(1, ranked.size + 1) / ranked.size
    passed = ranked <= threshold
    if passed.any():
        cutoff = ranked[np.where(passed)[0].max()]
        out[valid] = pv <= cutoff
    return out


def fdr_mask(pvalue: xr.DataArray, alpha: float = 0.05, dim: str | tuple[str, ...] | None = None) -> xr.DataArray:
    """Benjamini-Hochberg false-discovery-rate significance mask.

    By default all finite p-values in the field are treated as one family of
    tests. Pass ``dim`` to apply FDR independently along selected dimensions.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if dim is None:
        return xr.DataArray(_bh_fdr(pvalue.values, alpha), coords=pvalue.coords, dims=pvalue.dims, name="significant_fdr")
    dims = (dim,) if isinstance(dim, str) else tuple(dim)
    return xr.apply_ufunc(
        _bh_fdr, pvalue, kwargs={"alpha": alpha}, input_core_dims=[list(dims)],
        output_core_dims=[list(dims)], vectorize=True, dask="parallelized", output_dtypes=[bool],
    ).rename("significant_fdr")


def spatial_trend(data: xr.DataArray, dim: str = "time", alpha: float = 0.05, max_lag: int | None = None) -> xr.Dataset:
    """Compute grid-cell trend statistics and FDR-corrected significance."""
    result = modified_mann_kendall(data, dim=dim, max_lag=max_lag)
    result["sen_slope"] = sens_slope(data, dim=dim)
    result["significant"] = result["pvalue"] < alpha
    result["significant_fdr"] = fdr_mask(result["pvalue"], alpha=alpha)
    result.attrs["fdr_method"] = "Benjamini-Hochberg"
    result.attrs["alpha"] = alpha
    return result


__all__ = ["seasonal_series", "seasonal_mann_kendall", "seasonal_sen_slope", "fdr_mask", "spatial_trend"]
