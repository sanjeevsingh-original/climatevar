"""ETCCDI-inspired precipitation indices with time-aware safeguards.

The core precipitation definitions follow the ETCCDI/RClimDex convention:
wet days are RR >= 1 mm, dry days are RR < 1 mm, and percentile thresholds
for R95p/R99p are derived from wet-day precipitation in the 1961-1990
baseline unless an explicit reference threshold is supplied.
"""
from __future__ import annotations
import numpy as np
import xarray as xr
from ..time import infer_frequency, year_complete_mask
from .normalize import daily_amount

def _validate_daily(data, dim, *, min_daily_valid_fraction=0.9):
    if not 0 <= min_daily_valid_fraction <= 1: raise ValueError("min_daily_valid_fraction must be between 0 and 1.")
    try: frequency = infer_frequency(data.rename({dim: "time"}) if dim != "time" else data)
    except ValueError: frequency = "irregular"
    daily = daily_amount(data, dim)
    if min_daily_valid_fraction == 0 or frequency == "daily": return daily
    expected = {"hourly": 24, "3-hourly": 8, "6-hourly": 4}.get(frequency)
    if expected is None:
        if frequency == "irregular": raise ValueError("ETCCDI indices require a regular daily/sub-daily time axis; aggregate irregular data to daily totals first.")
        return daily
    counts = data.resample({dim: "1D"}).count(dim=dim)
    return daily.where((counts / expected) >= min_daily_valid_fraction)

def _validate_threshold(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0: raise ValueError(f"{name} must be a finite non-negative value in mm.")
    return value

def _complete_year_mask(data, dim, min_valid_fraction): return year_complete_mask(data, dim, min_valid_fraction=min_valid_fraction)

def _with_mm_units(result):
    result.attrs = dict(result.attrs); result.attrs["units"] = "mm"; result.attrs["climatevar:precipitation_unit"] = "mm"; return result

def rx1day(data, dim="time", min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction); complete = _complete_year_mask(data, dim, min_valid_fraction)
    return _with_mm_units(data.groupby(f"{dim}.year").max(dim=dim, skipna=True).where(complete))

def rx5day(data, dim="time", min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction); complete = _complete_year_mask(data, dim, min_valid_fraction); rolling = data.rolling({dim: 5}, min_periods=5).sum()
    return _with_mm_units(rolling.groupby(f"{dim}.year").max(dim=dim, skipna=True).where(complete))

def prcptot(data, dim="time", wet_day_threshold=1.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction); threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold"); complete = _complete_year_mask(data, dim, min_valid_fraction)
    return _with_mm_units(data.where(data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True).where(complete))

def r10mm(data, dim="time", threshold=10.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction); threshold = _validate_threshold(threshold, "threshold"); complete = _complete_year_mask(data, dim, min_valid_fraction)
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True).where(complete)

def r20mm(data, dim="time", threshold=20.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction); threshold = _validate_threshold(threshold, "threshold"); complete = _complete_year_mask(data, dim, min_valid_fraction)
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True).where(complete)

def sdii(data, dim="time", wet_day_threshold=1.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction); threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold"); complete = _complete_year_mask(data, dim, min_valid_fraction); wet = data.where(data >= threshold)
    result = wet.groupby(f"{dim}.year").sum(dim=dim, skipna=True) / wet.groupby(f"{dim}.year").count(dim=dim); result.attrs.update({"units": "mm/day", "climatevar:wet_day_threshold_mm": threshold}); return result.where(complete)

def wet_day_count(data, dim="time", wet_day_threshold=1.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction); threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold"); complete = _complete_year_mask(data, dim, min_valid_fraction)
    return (data >= threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True).where(complete)

def _reference_mm(reference):
    if not isinstance(reference, xr.DataArray): return xr.DataArray(float(reference), attrs={"units": "mm"})
    units = str(reference.attrs.get("units", "mm")).strip().lower(); factors = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
    if units not in factors: raise ValueError("Percentile reference must use mm, cm, or m.")
    out = reference * factors[units]; out.attrs = dict(reference.attrs); out.attrs["units"] = "mm"; return out

def _percentile_total(data, reference, percentile, dim, baseline_start, baseline_end, wet_day_threshold, min_valid_fraction, min_daily_valid_fraction):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction)
    if not 0 <= percentile <= 1: raise ValueError("percentile must be between 0 and 1.")
    wet_day_threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold"); complete = _complete_year_mask(data, dim, min_valid_fraction)
    if reference is not None: threshold = _reference_mm(reference)
    else:
        baseline = data.sel({dim: slice(baseline_start, baseline_end)}); threshold = baseline.where(baseline >= wet_day_threshold).quantile(percentile, dim=dim, skipna=True)
    result = data.where(data > threshold).groupby(f"{dim}.year").sum(dim=dim, skipna=True); result.attrs["climatevar:percentile_baseline"] = "explicit_reference" if reference is not None else f"{baseline_start}:{baseline_end}"; result.attrs["climatevar:wet_day_threshold_mm"] = wet_day_threshold
    return _with_mm_units(result.where(complete))

def r95p(data, reference=None, dim="time", baseline_start="1961-01-01", baseline_end="1990-12-31", wet_day_threshold=1.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    return _percentile_total(data, reference, 0.95, dim, baseline_start, baseline_end, wet_day_threshold, min_valid_fraction, min_daily_valid_fraction)

def r99p(data, reference=None, dim="time", baseline_start="1961-01-01", baseline_end="1990-12-31", wet_day_threshold=1.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    return _percentile_total(data, reference, 0.99, dim, baseline_start, baseline_end, wet_day_threshold, min_valid_fraction, min_daily_valid_fraction)

def r95p_fraction(data, reference=None, dim="time", baseline_start="1961-01-01", baseline_end="1990-12-31", wet_day_threshold=1.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    numerator = r95p(data, reference, dim, baseline_start, baseline_end, wet_day_threshold, min_valid_fraction, min_daily_valid_fraction=min_daily_valid_fraction); denominator = prcptot(data, dim, wet_day_threshold, min_valid_fraction, min_daily_valid_fraction=min_daily_valid_fraction)
    result = 100 * numerator / denominator; result.attrs.update({"units": "%", "climatevar:definition": "100 * R95p / PRCPTOT"}); return result

def r99p_fraction(data, reference=None, dim="time", baseline_start="1961-01-01", baseline_end="1990-12-31", wet_day_threshold=1.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    numerator = r99p(data, reference, dim, baseline_start, baseline_end, wet_day_threshold, min_valid_fraction, min_daily_valid_fraction=min_daily_valid_fraction); denominator = prcptot(data, dim, wet_day_threshold, min_valid_fraction, min_daily_valid_fraction=min_daily_valid_fraction)
    result = 100 * numerator / denominator; result.attrs.update({"units": "%", "climatevar:definition": "100 * R99p / PRCPTOT"}); return result

def _max_consecutive_1d(values):
    best = current = 0
    for value in values:
        if np.isfinite(value) and bool(value): current += 1; best = max(best, current)
        else: current = 0
    return best

def _max_run(condition, dim, complete):
    result = condition.groupby(f"{dim}.year").map(lambda block: xr.apply_ufunc(_max_consecutive_1d, block, input_core_dims=[[dim]], output_core_dims=[[]], vectorize=True, dask="parallelized", output_dtypes=[int])); return result.where(complete)

def cwd(data, dim="time", wet_day_threshold=1.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction); threshold = _validate_threshold(wet_day_threshold, "wet_day_threshold"); return _max_run(data >= threshold, dim, _complete_year_mask(data, dim, min_valid_fraction))

def cdd(data, dim="time", dry_day_threshold=1.0, min_valid_fraction=0.9, *, min_daily_valid_fraction=0.9):
    data = _validate_daily(data, dim, min_daily_valid_fraction=min_daily_valid_fraction); threshold = _validate_threshold(dry_day_threshold, "dry_day_threshold"); return _max_run(data < threshold, dim, _complete_year_mask(data, dim, min_valid_fraction))
