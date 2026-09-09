"""Validated implementations for non-parametric trend diagnostics."""
from __future__ import annotations
import numpy as np
import xarray as xr
from scipy.stats import kendalltau, norm, theilslopes


def _clean(v):
    y=np.asarray(v,dtype=float); return y[np.isfinite(y)]

def _mk_stats(y):
    n=y.size
    if n<2: return np.nan,np.nan,np.nan,n
    tau,p=kendalltau(np.arange(n),y)
    s=float(sum(np.sign(y[j]-y[i]) for i in range(n) for j in range(i+1,n)))
    return s,float(tau),float(p),n

def _sen_1d(v):
    y=_clean(v)
    return float(theilslopes(y,np.arange(y.size)).slope) if y.size>=2 else np.nan

def _lag_autocorrelation(y,max_lag):
    n=y.size
    if n<3:return np.array([])
    max_lag=min(int(max_lag),n-2); z=y-y.mean(); denom=np.sum(z*z)
    if denom==0:return np.zeros(max_lag)
    return np.array([np.sum(z[k:]*z[:-k])/denom for k in range(1,max_lag+1)])

def _mk_variance(y):
    n=y.size; _,counts=np.unique(y,return_counts=True)
    tie=np.sum(counts*(counts-1)*(2*counts+5))
    return (n*(n-1)*(2*n+5)-tie)/18.0

def _modified_mk_1d(v,max_lag=None):
    y=_clean(v); n=y.size
    if n<3:return np.nan,np.nan,np.nan,np.nan,float(n)
    s,tau,_,_= _mk_stats(y); slope=_sen_1d(y); residual=y-slope*np.arange(n)
    max_lag=max_lag or min(n-1,int(np.sqrt(n)*3)); r=_lag_autocorrelation(residual,max_lag)
    factor=max(1.0+(2.0/n)*np.sum((n-np.arange(1,len(r)+1))*r),1e-6)
    n_eff=float(np.clip(n/factor,1.0,n)); var_s=_mk_variance(y)*n/n_eff
    if var_s<=0:return s,tau,np.nan,n_eff,float(n)
    z=(s-1)/np.sqrt(var_s) if s>0 else (s+1)/np.sqrt(var_s) if s<0 else 0.0
    return s,tau,float(2*norm.sf(abs(z))),n_eff,float(n)

def _block_indices(n,block_length,rng):
    if not 1<=block_length<=n:raise ValueError("block_length must be between 1 and n")
    starts=rng.integers(0,n-block_length+1,size=int(np.ceil(n/block_length)))
    return np.concatenate([np.arange(s,s+block_length) for s in starts])[:n]

def _sen_slope_ci_1d(v,alpha=0.05,n_resamples=1000,block_length=None,random_state=0):
    y=_clean(v); n=y.size
    if n<3:return np.nan,np.nan,np.nan,float(n)
    if block_length is None:block_length=max(1,int(round(n**(1/3))))
    est=_sen_1d(y); rng=np.random.default_rng(random_state); boot=np.empty(n_resamples)
    for i in range(n_resamples):boot[i]=_sen_1d(y[_block_indices(n,block_length,rng)])
    q=np.quantile(boot,[alpha/2,1-alpha/2]); return est,float(q[0]),float(q[1]),float(n)

def mann_kendall(data,dim="time"):
    if dim not in data.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    r=xr.apply_ufunc(_mk_stats,data,input_core_dims=[[dim]],output_core_dims=[[],[],[],[]],vectorize=True,dask="parallelized",output_dtypes=[float,float,float,int])
    return xr.Dataset({"s":r[0],"tau":r[1],"pvalue":r[2],"n":r[3]})

def modified_mann_kendall(data,dim="time",max_lag=None):
    if dim not in data.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    r=xr.apply_ufunc(_modified_mk_1d,data,kwargs={"max_lag":max_lag},input_core_dims=[[dim]],output_core_dims=[[],[],[],[],[]],vectorize=True,dask="parallelized",output_dtypes=[float,float,float,float,float])
    return xr.Dataset({"s":r[0],"tau":r[1],"pvalue":r[2],"n_eff":r[3],"n":r[4]})

def trend_free_prewhitening(data,dim="time"):
    if dim not in data.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    def tfpw(v):
        y=_clean(v)
        if y.size<3:return np.full(np.asarray(v).shape,np.nan)
        slope=_sen_1d(y); t=np.arange(y.size,dtype=float); d=y-slope*t; z=d-d.mean(); denom=np.sum(z[:-1]**2)
        r1=np.sum(z[1:]*z[:-1])/denom if denom>0 else 0.0; residual=d[1:]-r1*d[:-1]
        return np.r_[residual[0],residual+slope*np.arange(1,y.size)]
    return xr.apply_ufunc(tfpw,data,input_core_dims=[[dim]],output_core_dims=[[dim]],vectorize=True,dask="parallelized",output_dtypes=[float])

def sens_slope(data,dim="time"):
    if dim not in data.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    return xr.apply_ufunc(_sen_1d,data,input_core_dims=[[dim]],output_core_dims=[[]],vectorize=True,dask="parallelized",output_dtypes=[float])

def sens_slope_ci(data,dim="time",alpha=0.05,n_resamples=1000,block_length=None,random_state=0):
    if not 0<alpha<1:raise ValueError("alpha must be between 0 and 1")
    r=xr.apply_ufunc(_sen_slope_ci_1d,data,kwargs={"alpha":alpha,"n_resamples":n_resamples,"block_length":block_length,"random_state":random_state},input_core_dims=[[dim]],output_core_dims=[[],[],[],[]],vectorize=True,dask="parallelized",output_dtypes=[float,float,float,float])
    return xr.Dataset({"sen_slope":r[0],"ci_lower":r[1],"ci_upper":r[2],"n":r[3]}).assign_attrs(method="moving-block bootstrap percentile interval",alpha=alpha,confidence_level=1-alpha,n_resamples=n_resamples,block_length="automatic" if block_length is None else block_length,random_state=random_state,slope_units="per observation step")

__all__=["_clean","_mk_stats","_sen_1d","_lag_autocorrelation","_mk_variance","mann_kendall","modified_mann_kendall","trend_free_prewhitening","sens_slope","sens_slope_ci"]
