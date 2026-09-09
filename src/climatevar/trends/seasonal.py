"""Seasonal and spatial trend diagnostics for climate data."""
from __future__ import annotations

import calendar
import numpy as np
import xarray as xr

from ..time import calendar_name, infer_frequency
from .trend import modified_mann_kendall, sens_slope


def _season_months(season):
    names = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
    key = str(season).upper()
    if key not in names:
        raise ValueError("season must be one of DJF, MAM, JJA, SON")
    return names[key]


def _season_year_index(time, season):
    months = _season_months(season)
    year = time.dt.year
    # Only DJF crosses a calendar year; December belongs to the following
    # season-year. JJA/MAM/SON retain their ordinary calendar year.
    if str(season).upper() == "DJF":
        return xr.where(time.dt.month == 12, year + 1, year)
    return year


def _days_in_month(year: int, month: int, cal: str) -> int:
    if cal == "360_day": return 30
    if cal == "noleap": return (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]
    if cal == "all_leap": return (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)[month - 1]
    return calendar.monthrange(year, month)[1]


def _is_precipitation(data: xr.DataArray) -> bool:
    text = " ".join(str(data.attrs.get(key, "")) for key in ("standard_name", "long_name", "name", "units")).lower()
    return any(token in text for token in ("precip", "rain", "prcp"))


def _expected_samples(year: int, months: tuple[int, ...], frequency: str, cal: str) -> int:
    days = sum(_days_in_month(year, month, cal) for month in months)
    if frequency == "monthly": return len(months)
    samples_per_day = {"daily": 1, "6-hourly": 4, "3-hourly": 8, "hourly": 24}.get(frequency)
    if samples_per_day is None:
        raise ValueError("seasonal_series requires a regular hourly, 3-hourly, 6-hourly, daily, or monthly time axis. Pass a pre-aggregated seasonal series for irregular sampling.")
    return days * samples_per_day


def seasonal_series(data, season, dim="time", *, aggregation="auto", min_valid_fraction=1.0):
    """Build one value per climatological season with calendar-aware completeness."""
    if dim not in data.dims: raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if not 0 <= min_valid_fraction <= 1: raise ValueError("min_valid_fraction must be between 0 and 1.")
    aggregation = str(aggregation).lower()
    if aggregation not in {"auto", "sum", "mean"}: raise ValueError("aggregation must be 'auto', 'sum', or 'mean'")
    months = _season_months(season)
    frequency = infer_frequency(data)
    cal = calendar_name(data)
    selected = data.where(data[dim].dt.month.isin(months), drop=True)
    season_year = _season_year_index(selected[dim], season).rename("season_year")
    selected = selected.assign_coords(season_year=(dim, season_year.data))
    grouped = selected.groupby("season_year")
    if aggregation == "auto": aggregation = "sum" if _is_precipitation(data) else "mean"
    reducer = grouped.sum if aggregation == "sum" else grouped.mean
    result = reducer(dim=dim, skipna=True)
    valid_count = grouped.count(dim=dim)
    years = np.asarray(result["season_year"].values, dtype=int)
    expected = xr.DataArray([_expected_samples(int(y), months, frequency, cal) for y in years], coords={"season_year": result["season_year"]}, dims="season_year")
    complete = xr.ones_like(valid_count, dtype=bool) if min_valid_fraction == 0 else valid_count >= expected * min_valid_fraction
    result = result.where(complete).rename({"season_year": "year"})
    result.attrs.update(data.attrs)
    result.attrs.update({"season": str(season).upper(), "season_aggregation": aggregation, "season_frequency": frequency, "season_min_valid_fraction": min_valid_fraction, "season_calendar": cal})
    return result


def seasonal_mann_kendall(data, season, dim="time", max_lag=None, **kwargs):
    series = seasonal_series(data, season, dim=dim, **kwargs)
    result = modified_mann_kendall(series, dim="year", max_lag=max_lag)
    result.attrs["season"] = str(season).upper()
    return result


def seasonal_sen_slope(data, season, dim="time", **kwargs):
    series = seasonal_series(data, season, dim=dim, **kwargs)
    result = sens_slope(series, dim="year")
    result.attrs.update({"season": str(season).upper(), "slope_units": "input units per season-year"})
    return result


def _bh_fdr(pvalues, alpha):
    p = np.asarray(pvalues, dtype=float); valid = np.isfinite(p); out = np.zeros(p.shape, dtype=bool)
    if not valid.any(): return out
    pv = p[valid]; order = np.argsort(pv); ranked = pv[order]; threshold = alpha * np.arange(1, ranked.size + 1) / ranked.size; passed = ranked <= threshold
    if passed.any(): out[valid] = pv <= ranked[np.where(passed)[0].max()]
    return out


def fdr_mask(pvalue, alpha=0.05, dim=None):
    if not 0 < alpha < 1: raise ValueError("alpha must be between 0 and 1")
    if dim is None: return xr.DataArray(_bh_fdr(pvalue.values, alpha), coords=pvalue.coords, dims=pvalue.dims, name="significant_fdr")
    dims = (dim,) if isinstance(dim, str) else tuple(dim)
    return xr.apply_ufunc(_bh_fdr, pvalue, kwargs={"alpha": alpha}, input_core_dims=[list(dims)], output_core_dims=[list(dims)], vectorize=True, dask="parallelized", output_dtypes=[bool]).rename("significant_fdr")


def spatial_trend(data, dim="time", alpha=0.05, max_lag=None):
    result = modified_mann_kendall(data, dim=dim, max_lag=max_lag)
    result["sen_slope"] = sens_slope(data, dim=dim)
    result["significant"] = result["pvalue"] < alpha
    result["significant_fdr"] = fdr_mask(result["pvalue"], alpha=alpha)
    result.attrs["fdr_method"] = "Benjamini-Hochberg"; result.attrs["alpha"] = alpha
    return result


__all__ = ["seasonal_series", "seasonal_mann_kendall", "seasonal_sen_slope", "fdr_mask", "spatial_trend"]
