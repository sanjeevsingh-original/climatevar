"""Normalize heterogeneous climate datasets to a small canonical schema."""
from __future__ import annotations
from collections.abc import Mapping
import warnings
import xarray as xr
from ..units import normalize_units

CANONICAL_COORDS={"latitude","longitude","time"}
COORD_ALIASES={"latitude":("latitude","lat","Latitude","LATITUDE","nav_lat","y_lat","XLAT","XLAT_M","XLAT_U","XLAT_V"),"longitude":("longitude","lon","Longitude","LONGITUDE","nav_lon","x_lon","XLONG","XLONG_M","XLONG_U","XLONG_V"),"time":("time","Time","TIME","valid_time","datetime","date","Date","Times")}
VARIABLE_ALIASES={"precipitation":("precipitation","precip","precipitationCal","precipitationCalES","pr","tp","rain","rainfall","rf","APCP_sfc","RAINC","RAINNC","RAINSH"),"temperature":("temperature","temp","t2m","TMP_2m","TMP_sfc","T2","tas","air_temperature"),"surface_pressure":("surface_pressure","sp","ps","psl","PRES_sfc","PSFC","pres","pressure"),"relative_humidity":("relative_humidity","rh","RH2","RH_2m","RH_prl","hur","r"),"specific_humidity":("specific_humidity","q","q2","Q2","hus"),"u_wind":("u_wind","u10","U10","UGRD_10m","UGRD_prl","ua","uwnd","u"),"v_wind":("v_wind","v10","V10","VGRD_10m","VGRD_prl","va","vwnd","v"),"geopotential":("geopotential","z","gh","zg","HGT_prl","HGT","GHT")}
DATASET_PRESETS={"era5":{"latitude":"latitude","longitude":"longitude","time":"time","precipitation":"tp","temperature":"t2m","surface_pressure":"sp"},"era5-land":{"latitude":"latitude","longitude":"longitude","time":"time","precipitation":"tp"},"imerg":{"latitude":"lat","longitude":"lon","time":"time","precipitation":"precipitation"},"gpm":{"latitude":"lat","longitude":"lon","time":"time","precipitation":"precipitation"},"wrf":{"latitude":"XLAT","longitude":"XLONG","time":"Time"},"imd":{"latitude":"lat","longitude":"lon","time":"time"},"imdaa":{"latitude":"latitude","longitude":"longitude","time":"time","precipitation":"APCP_sfc","temperature":"TMP_2m","surface_pressure":"PRES_sfc","relative_humidity":"RH_2m","u_wind":"UGRD_10m","v_wind":"VGRD_10m"},"cmip6":{"latitude":"lat","longitude":"lon","time":"time","precipitation":"pr","temperature":"tas"}}

def _find_name(names,aliases):
    for alias in aliases:
        if alias in names:return alias
    folded={name.casefold():name for name in names}
    for alias in aliases:
        if alias.casefold() in folded:return folded[alias.casefold()]
    return None

def _wrf_times(ds):
    if "Times" not in ds.variables or "time" in ds.coords:return ds
    values=ds["Times"].values
    if getattr(values,"ndim",0)!=2:return ds
    try: strings=[b"".join(row).decode("utf-8") for row in values]
    except Exception:
        try: strings=["".join(str(x) for x in row) for row in values]
        except Exception:return ds
    try:
        import pandas as pd
        time=pd.to_datetime([s.replace("_"," ") for s in strings]).to_numpy(dtype="datetime64[ns]")
    except Exception:return ds
    if "Time" in ds.dims:ds=ds.assign_coords(time=("Time",time))
    return ds

def normalize_coords(ds,*,dataset=None,strict=False):
    ds=_wrf_times(ds); names=list(ds.coords)+[n for n in ds.variables if n not in ds.coords]; preset=DATASET_PRESETS.get((dataset or "").lower(),{}); renames={}
    for canonical,aliases in COORD_ALIASES.items():
        preferred=preset.get(canonical); source=preferred if preferred in names else _find_name(names,aliases)
        if source and source!=canonical and canonical not in ds.variables:renames[source]=canonical
    out=ds.rename(renames) if renames else ds
    if strict:
        missing=[n for n in ("latitude","longitude") if n not in out]
        if missing:raise ValueError(f"Could not identify required coordinate(s): {', '.join(missing)}")
    history=list(out.attrs.get("climatevar:normalization",[])); history=[history] if isinstance(history,str) else history; history.append(f"normalize_coords(dataset={dataset!r})"); out.attrs["climatevar:normalization"]=history; return out

def standardize_variables(ds,*,variables=None,dataset=None,strict=False):
    mapping=dict(variables or {}); preset=DATASET_PRESETS.get((dataset or "").lower(),{}); names=list(ds.data_vars); renames={}
    if (dataset or "").lower()=="wrf" and variables is None and "RAINC" in names and "RAINNC" in names:
        warnings.warn("WRF contains both RAINC and RAINNC; use wrf_precipitation_amount() to form total precipitation. No ambiguous canonical precipitation variable is created.",UserWarning,stacklevel=2)
    for canonical,aliases in VARIABLE_ALIASES.items():
        if canonical in ds.data_vars:continue
        source=mapping.get(canonical) or preset.get(canonical)
        if source not in names:source=_find_name(names,aliases)
        if source and source!=canonical:
            if canonical=="precipitation" and (dataset or "").lower()=="wrf" and {"RAINC","RAINNC"}.issubset(set(names)):continue
            renames[source]=canonical
    if strict and variables:
        missing=[target for target,source in mapping.items() if source not in ds.data_vars]
        if missing:raise KeyError(f"Requested source variable(s) not found: {', '.join(missing)}")
    out=ds.rename(renames) if renames else ds; history=list(out.attrs.get("climatevar:normalization",[])); history=[history] if isinstance(history,str) else history; history.append(f"standardize_variables(dataset={dataset!r}, variables={mapping!r})"); out.attrs["climatevar:normalization"]=history; return out

def normalize_dataset(ds,*,dataset=None,variables=None,strict=True,si=True):
    out=standardize_variables(normalize_coords(ds,dataset=dataset,strict=strict),variables=variables,dataset=dataset); return normalize_units(out) if si else out

def find_variable(ds,standard_name):
    if standard_name in ds.data_vars:return standard_name
    return _find_name(list(ds.data_vars),VARIABLE_ALIASES.get(standard_name,()))
