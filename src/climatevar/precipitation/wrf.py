"""WRF precipitation utilities.

WRF commonly stores RAINC and RAINNC as cumulative precipitation fields.
These helpers combine the configured precipitation components and convert
cumulative fields to interval amounts without requiring users to perform the
bookkeeping themselves.
"""
from __future__ import annotations
import xarray as xr

def wrf_total_precipitation(ds,*,include_shallow=False,convective="RAINC",nonconvective="RAINNC",shallow="RAINSH"):
    missing=[name for name in (convective,nonconvective) if name not in ds]
    if missing:raise KeyError(f"Required WRF precipitation variable(s) missing: {', '.join(missing)}")
    total=ds[convective]+ds[nonconvective]
    if include_shallow:
        if shallow not in ds:raise KeyError(f"Requested shallow-convection variable {shallow!r} is missing.")
        total=total+ds[shallow]
    total=total.copy(); total.name="precipitation_accumulated"; total.attrs=dict(total.attrs); total.attrs["climatevar:precipitation_kind"]="accumulated"; total.attrs["climatevar:source_components"]=",".join([convective,nonconvective]+([shallow] if include_shallow else [])); return total

def wrf_precipitation_amount(ds,*,include_shallow=False,convective="RAINC",nonconvective="RAINNC",shallow="RAINSH",dim="Time"):
    accumulated=wrf_total_precipitation(ds,include_shallow=include_shallow,convective=convective,nonconvective=nonconvective,shallow=shallow)
    if dim not in accumulated.dims:
        if "time" in accumulated.dims:dim="time"
        else:raise ValueError(f"Time dimension {dim!r} was not found in WRF precipitation data.")
    diff=accumulated.diff(dim,label="upper")
    reset=diff<0
    # At a restart, the post-reset cumulative value represents accumulation
    # since restart, so use it. For a normal interval use the difference.
    current=accumulated.isel({dim:slice(1,None)})
    increments=diff.where(~reset,current)
    first=accumulated.isel({dim:0}).expand_dims({dim:[accumulated[dim].values[0]]})
    increments=xr.concat([first,increments],dim=dim)
    increments.name="precipitation"; increments.attrs=dict(accumulated.attrs); increments.attrs.update({"units":"mm","climatevar:precipitation_kind":"interval_amount","climatevar:source_accumulation":"WRF cumulative precipitation"}); return increments
