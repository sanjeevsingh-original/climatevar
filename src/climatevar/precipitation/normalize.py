"""Implicit precipitation normalization for common climate datasets."""
from __future__ import annotations
import numpy as np
import xarray as xr
from ..time import infer_frequency, time_step_seconds
_AMOUNT_FACTORS = {"mm":1.0,"millimeter":1.0,"millimetre":1.0,"cm":10.0,"m":1000.0,"meter":1000.0,"metre":1000.0}
_RATE_TO_MM_PER_SECOND = {"mm/s":1.0,"mm s-1":1.0,"mm s^-1":1.0,"mm/min":1/60,"mm min-1":1/60,"mm/hour":1/3600,"mm/hr":1/3600,"mm h-1":1/3600,"mm hr-1":1/3600,"mm/day":1/86400,"mm day-1":1/86400,"mm d-1":1/86400,"kg m-2 s-1":1.0,"kg/m2/s":1.0,"kg m-2 s^-1":1.0,"kg m-2 min-1":1/60,"kg/m2/min":1/60,"kg m-2 h-1":1/3600,"kg/m2/hour":1/3600,"kg m-2 day-1":1/86400,"kg m-2 d-1":1/86400,"kg/m2/day":1/86400}

def _units(data): return str(data.attrs.get("units","mm")).strip().lower().replace("°","deg").replace("*","")

def _step_seconds(data, dim):
    values = time_step_seconds(data.rename({dim:"time"}) if dim != "time" else data).values
    finite = values[np.isfinite(values) & (values > 0)]
    if not finite.size: raise ValueError("At least two valid time coordinates are required for rate precipitation.")
    fallback = float(np.median(finite))
    values = np.where(np.isfinite(values) & (values > 0), values, fallback)
    return xr.DataArray(values, coords={dim:data[dim]}, dims=dim)

def interval_amount_mm(data, dim="time"):
    if dim not in data.dims or dim not in data.coords: raise ValueError(f"{dim!r} must be a dimension with a coordinate.")
    unit = _units(data)
    if unit in _AMOUNT_FACTORS: out = data * _AMOUNT_FACTORS[unit]
    elif unit in _RATE_TO_MM_PER_SECOND: out = data * _RATE_TO_MM_PER_SECOND[unit] * _step_seconds(data,dim)
    else: raise ValueError(f"Unsupported precipitation units {data.attrs.get('units')!r}.")
    out.attrs=dict(data.attrs); out.attrs.update({"units":"mm","climatevar:quantity":"precipitation_amount","climatevar:source_units":data.attrs.get("units","mm")})
    return out

def normalize_precipitation(data, dim="time"):
    if dim not in data.dims or dim not in data.coords: raise ValueError(f"{dim!r} must be a dimension with a coordinate.")
    amount=interval_amount_mm(data,dim)
    try: frequency=infer_frequency(data.rename({dim:"time"}) if dim!="time" else data)
    except ValueError: frequency="irregular"
    denominators={"hourly":("hr",3600.0),"3-hourly":("3hr",10800.0),"6-hourly":("6hr",21600.0),"daily":("day",86400.0)}
    if frequency not in denominators:
        amount.attrs["climatevar:frequency"]=frequency; amount.attrs["climatevar:precipitation_representation"]="period_amount"; return amount
    denominator,seconds=denominators[frequency]
    out=amount.copy(); out.attrs=dict(data.attrs); out.attrs.update({"units":f"mm/{denominator}","climatevar:quantity":"precipitation_rate","climatevar:frequency":frequency,"climatevar:source_units":data.attrs.get("units","mm"),"climatevar:precipitation_representation":"interval_rate"}); return out

def daily_amount(data, dim="time"):
    amount=interval_amount_mm(data,dim)
    try: frequency=infer_frequency(data.rename({dim:"time"}) if dim!="time" else data)
    except ValueError: frequency="irregular"
    if frequency=="daily": amount.attrs.update({"units":"mm","climatevar:frequency":"daily"}); return amount
    if frequency in {"hourly","3-hourly","6-hourly","irregular"}:
        result=amount.resample({dim:"1D"}).sum(skipna=True); result.attrs=dict(data.attrs); result.attrs.update({"units":"mm","climatevar:quantity":"precipitation_amount","climatevar:frequency":"daily","climatevar:source_units":data.attrs.get("units","mm")}); return result
    return amount
