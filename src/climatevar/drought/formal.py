"""Formal SPI, SPEI, drought classification, and diagnostics."""
from __future__ import annotations
import numpy as np
import xarray as xr
from scipy.stats import gamma, fisk, norm, kstest

def _fit_spi_1d(values, distribution):
    y=np.asarray(values,dtype=float); out=np.full(y.shape,np.nan); finite=np.isfinite(y); sample=y[finite]
    if sample.size<3:return out
    positive=sample>0; p0=1-positive.mean()
    if positive.sum()<3:return out
    dist=distribution.lower()
    if dist=="gamma": a,loc,scale=gamma.fit(sample[positive],floc=0); cdf=np.full(sample.shape,p0); cdf[positive]=p0+(1-p0)*gamma.cdf(sample[positive],a,loc=loc,scale=scale)
    elif dist in {"pearson3","pearson_iii","pearson-iii"}:
        from scipy.stats import pearson3
        skew,loc,scale=pearson3.fit(sample[positive]); cdf=np.full(sample.shape,p0); cdf[positive]=p0+(1-p0)*pearson3.cdf(sample[positive],skew,loc=loc,scale=scale)
    else: raise ValueError("distribution must be 'gamma' or 'pearson3'.")
    out[finite]=norm.ppf(np.clip(cdf,1e-8,1-1e-8)); return out

def _validate_scale(scale):
    if scale<1 or int(scale)!=scale: raise ValueError("scale must be a positive integer.")
    return int(scale)

def spi(precipitation,scale=3,dim="time",distribution="gamma",calibration_start=None,calibration_end=None,min_samples=10,min_nonzero=None,clip=8.0):
    scale=_validate_scale(scale)
    if dim not in precipitation.dims: raise ValueError(f"Dimension {dim!r} is not present in the data.")
    if bool((precipitation<0).any()): raise ValueError("precipitation must be non-negative.")
    accumulated=precipitation.rolling({dim:scale},min_periods=scale).sum()
    calibration=accumulated.sel({dim:slice(calibration_start,calibration_end)}) if calibration_start or calibration_end else accumulated
    months=accumulated[dim].dt.month; cal_months=calibration[dim].dt.month
    def fit_apply(values,month_values,cal_values,cal_month_values):
        out=np.full(values.shape,np.nan)
        for month in range(1,13):
            cal=np.asarray(cal_values[cal_month_values==month],dtype=float); cal=cal[np.isfinite(cal)]
            if cal.size<min_samples: continue
            positive=cal>0
            required=max(3,min_samples//2) if min_nonzero is None else int(min_nonzero)
            if positive.sum()<required: continue
            p0=1-positive.mean(); x=np.asarray(values[month_values==month],dtype=float); finite=np.isfinite(x); cdf=np.full(x.shape,np.nan)
            dist=distribution.lower()
            if dist=="gamma": a,loc,sc=gamma.fit(cal[positive],floc=0); cdf[finite&(x<=0)]=p0; pos=finite&(x>0); cdf[pos]=p0+(1-p0)*gamma.cdf(x[pos],a,loc=loc,scale=sc)
            elif dist in {"pearson3","pearson_iii","pearson-iii"}:
                from scipy.stats import pearson3
                skew,loc,sc=pearson3.fit(cal[positive]); cdf[finite&(x<=0)]=p0; pos=finite&(x>0); cdf[pos]=p0+(1-p0)*pearson3.cdf(x[pos],skew,loc=loc,scale=sc)
            else: raise ValueError("distribution must be 'gamma' or 'pearson3'.")
            z=norm.ppf(np.clip(cdf,1e-8,1-1e-8)); z=np.clip(z,-float(clip),float(clip)) if clip is not None else z; out[month_values==month]=z
        return out
    try:
        result=xr.apply_ufunc(fit_apply,accumulated,months,calibration,cal_months,input_core_dims=[[dim],[dim],[dim],[dim]],output_core_dims=[[dim]],vectorize=True,dask="parallelized",output_dtypes=[float]).transpose(*precipitation.dims)
    except Exception:
        result=precipitation.copy(data=fit_apply(accumulated.values,months.values,calibration.values,cal_months.values))
    result.name="spi"; result.attrs=dict(precipitation.attrs); result.attrs.update({"standard_name":"standardized_precipitation_index","climatevar:scale":scale,"climatevar:distribution":distribution,"climatevar:calibration_start":calibration_start or "full_record","climatevar:calibration_end":calibration_end or "full_record","climatevar:min_calibration_samples":int(min_samples),"climatevar:min_nonzero":min_nonzero}); return result

def spei(precipitation,potential_evapotranspiration,scale=3,dim="time",calibration_start=None,calibration_end=None,min_samples=10,clip=8.0):
    scale=_validate_scale(scale)
    if dim not in precipitation.dims: raise ValueError(f"Dimension {dim!r} is not present in the data.")
    water_balance=precipitation-potential_evapotranspiration; accumulated=water_balance.rolling({dim:scale},min_periods=scale).sum(); calibration=accumulated.sel({dim:slice(calibration_start,calibration_end)}) if calibration_start or calibration_end else accumulated; months=accumulated[dim].dt.month; cal_months=calibration[dim].dt.month
    if precipitation.dims!=potential_evapotranspiration.dims or precipitation.sizes!=potential_evapotranspiration.sizes: raise ValueError("precipitation and potential_evapotranspiration must have matching dimensions and sizes.")
    def fit_apply(values,month_values,cal_values,cal_month_values):
        out=np.full(values.shape,np.nan)
        for month in range(1,13):
            cal=np.asarray(cal_values[cal_month_values==month],dtype=float); cal=cal[np.isfinite(cal)]
            if cal.size<min_samples: continue
            shape,loc,sc=fisk.fit(cal); x=np.asarray(values[month_values==month],dtype=float); finite=np.isfinite(x); z=np.full(x.shape,np.nan); z[finite]=norm.ppf(np.clip(fisk.cdf(x[finite],shape,loc=loc,scale=sc),1e-8,1-1e-8)); z=np.clip(z,-float(clip),float(clip)) if clip is not None else z; out[month_values==month]=z
        return out
    result=xr.apply_ufunc(fit_apply,accumulated,months,calibration,cal_months,input_core_dims=[[dim],[dim],[dim],[dim]],output_core_dims=[[dim]],vectorize=True,dask="parallelized",output_dtypes=[float]).transpose(*precipitation.dims); result.name="spei"; result.attrs={"standard_name":"standardized_precipitation_evapotranspiration_index","climatevar:scale":scale,"climatevar:distribution":"three-parameter log-logistic (Fisk)","climatevar:calibration_start":calibration_start or "full_record","climatevar:calibration_end":calibration_end or "full_record","climatevar:min_calibration_samples":int(min_samples)}; return result

def spi_like(precip,dim="time",scale=1):
    scale=_validate_scale(scale); accumulated=precip.rolling({dim:scale},min_periods=scale).sum(); return (accumulated-accumulated.mean(dim=dim,skipna=True))/accumulated.std(dim=dim,skipna=True,ddof=1)

def drought_category(index):
    return xr.apply_ufunc(lambda x:np.select([x<=-2,x<=-1.5,x<=-1,x<1,x<1.5,x<2],["extreme drought","severe drought","moderate drought","near normal","moderately wet","severely wet"],default="extremely wet"),index,vectorize=True,output_dtypes=[str])

def fit_quality(index):
    values=index.values[np.isfinite(index.values)]
    if values.size<3:return xr.Dataset({"n":xr.DataArray(int(values.size)),"mean":xr.DataArray(np.nan),"std":xr.DataArray(np.nan),"ks_pvalue":xr.DataArray(np.nan)})
    ks=kstest(values,"norm"); return xr.Dataset({"n":xr.DataArray(int(values.size)),"mean":xr.DataArray(float(np.mean(values))),"std":xr.DataArray(float(np.std(values,ddof=1))),"ks_pvalue":xr.DataArray(float(ks.pvalue))})
