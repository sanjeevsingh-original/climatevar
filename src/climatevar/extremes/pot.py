"""Peaks-over-threshold (POT) analysis with generalized Pareto models."""
from __future__ import annotations
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import genpareto

def _as_1d(values):
    y=np.asarray(values,dtype=float).ravel(); return y[np.isfinite(y)]

def _fit_gpd_1d(excesses):
    y=_as_1d(excesses)
    if y.size<5:return np.nan,np.nan,y.size
    if np.any(y<0):raise ValueError("GPD excesses must be non-negative.")
    shape,_,scale=genpareto.fit(y,floc=0.0)
    return float(shape),float(scale),int(y.size)

def gpd_fit(excesses:xr.DataArray,dim="time"):
    if dim not in excesses.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    r=xr.apply_ufunc(_fit_gpd_1d,excesses,input_core_dims=[[dim]],output_core_dims=[[],[],[]],vectorize=True,dask="parallelized",output_dtypes=[float,float,int])
    return xr.Dataset({"shape":r[0],"scale":r[1],"n_exceedances":r[2]})

def pot_exceedances(data:xr.DataArray,threshold:float,dim="time"):
    if dim not in data.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if not np.isfinite(threshold):raise ValueError("threshold must be finite.")
    return (data-threshold).where(data>threshold)

def decluster_exceedances(data:xr.DataArray,threshold:float,run_length:int=3,dim="time"):
    """Retain the maximum observation from each runs-rule exceedance cluster."""
    if dim not in data.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if run_length<1:raise ValueError("run_length must be at least 1.")
    y=np.asarray(data.values,dtype=float)
    if y.ndim!=1:raise ValueError("decluster_exceedances currently requires a one-dimensional series.")
    out=np.full(y.shape,np.nan); exceed=np.isfinite(y)&(y>threshold); i=0;n=y.size
    while i<n:
        if not exceed[i]:i+=1;continue
        j=i; gap=0
        while j<n:
            if exceed[j]:gap=0
            else:
                gap+=1
                if gap>=run_length:break
            j+=1
        stop=j-gap if j<n else n
        idx=np.where(exceed[i:stop])[0]
        if idx.size: out[i+idx[np.argmax(y[i+idx])]]=y[i+idx[np.argmax(y[i+idx])]]
        i=max(j,i+1)
    return xr.DataArray(out,coords=data.coords,dims=data.dims,attrs=data.attrs,name=data.name)

def _threshold_table(data,thresholds):
    y=_as_1d(data);n=y.size;counts=np.array([np.sum(y>u) for u in thresholds],int);rates=counts/n if n else np.full(len(thresholds),np.nan);shapes=np.full(len(thresholds),np.nan);scales=np.full(len(thresholds),np.nan);means=np.full(len(thresholds),np.nan)
    for j,u in enumerate(thresholds):
        e=y[y>u]-u
        if e.size>=5:shapes[j],scales[j],_= _fit_gpd_1d(e);means[j]=e.mean()
    return counts,rates,shapes,scales,means

def _validate_thresholds(thresholds):
    t=np.asarray(thresholds,dtype=float)
    if t.ndim!=1 or t.size==0 or np.any(~np.isfinite(t)):raise ValueError("thresholds must be a non-empty one-dimensional finite sequence.")
    return t

def threshold_diagnostics(data,thresholds,dim="time"):
    if dim not in data.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    t=_validate_thresholds(thresholds);c,r,s,sc,m=_threshold_table(data.values,t)
    return xr.Dataset({"threshold":("threshold",t),"n_exceedances":("threshold",c),"exceedance_rate":("threshold",r),"shape":("threshold",s),"scale":("threshold",sc),"mean_excess":("threshold",m)})

def _return_level(shape,scale,threshold,rate,return_period):
    if return_period<=0 or rate<=0 or scale<=0:return np.nan
    target=rate*return_period
    if target<=1:return np.nan
    if shape<0 and target**shape>=1:return np.nan
    return threshold+scale*np.log(target) if abs(shape)<1e-8 else threshold+scale/shape*(target**shape-1)

def pot_return_level(threshold,shape,scale,exceedance_rate,return_period):
    if not np.isfinite(exceedance_rate) or exceedance_rate<=0:raise ValueError("exceedance_rate must be positive and finite.")
    if not np.isfinite(return_period) or return_period<=0:raise ValueError("return_period must be positive and finite.")
    if exceedance_rate*return_period<=1:raise ValueError("return_period must exceed the threshold-exceedance recurrence interval.")
    level=_return_level(shape,scale,threshold,exceedance_rate,return_period)
    if not np.isfinite(level):raise ValueError("The requested return level is outside the fitted GPD support.")
    return float(level)

def pot_return_level_ci(data,threshold,return_period,dim="time",decluster_run_length=None,n_resamples=1000,alpha=.05,random_state=0):
    if not 0<alpha<1:raise ValueError("alpha must be between 0 and 1.")
    if n_resamples<100:raise ValueError("n_resamples must be at least 100.")
    if not np.isfinite(return_period) or return_period<=0:raise ValueError("return_period must be positive and finite.")
    if dim not in data.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    y=np.asarray(data.values,float)
    if y.ndim!=1:raise ValueError("pot_return_level_ci currently requires a one-dimensional series.")
    series=decluster_exceedances(data,threshold,decluster_run_length,dim).values if decluster_run_length is not None else y
    series=series[np.isfinite(series)] if decluster_run_length is not None else series[np.isfinite(series)&(series>threshold)]
    excess=series-threshold;n_obs=np.sum(np.isfinite(y))
    if excess.size<5 or n_obs==0:return xr.Dataset({"return_level":np.nan,"ci_lower":np.nan,"ci_upper":np.nan,"shape":np.nan,"scale":np.nan,"n_exceedances":excess.size,"exceedance_rate":np.nan})
    shape,scale,n_exc=_fit_gpd_1d(excess);rate=n_exc/n_obs
    if rate*return_period<=1:raise ValueError("return_period must exceed the threshold-exceedance recurrence interval.")
    observed=_return_level(shape,scale,threshold,rate,return_period);rng=np.random.default_rng(random_state);levels=np.full(n_resamples,np.nan)
    for i in range(n_resamples):
        bs,bc,_=_fit_gpd_1d(rng.choice(excess,size=n_exc,replace=True));levels[i]=_return_level(bs,bc,threshold,rate,return_period)
    levels=levels[np.isfinite(levels)];lo,hi=(np.nan,np.nan) if levels.size<max(50,n_resamples//2) else np.quantile(levels,[alpha/2,1-alpha/2])
    return xr.Dataset({"return_level":observed,"ci_lower":lo,"ci_upper":hi,"shape":shape,"scale":scale,"n_exceedances":n_exc,"exceedance_rate":rate})

def pot_threshold_sensitivity(data,thresholds,return_period,dim="time"):
    if not np.isfinite(return_period) or return_period<=0:raise ValueError("return_period must be positive and finite.")
    t=_validate_thresholds(thresholds);c,r,s,sc,m=_threshold_table(data.values,t);levels=np.array([_return_level(xi,sig,u,rate,return_period) for u,xi,sig,rate in zip(t,s,sc,r)])
    return xr.Dataset({"threshold":("threshold",t),"n_exceedances":("threshold",c),"exceedance_rate":("threshold",r),"shape":("threshold",s),"scale":("threshold",sc),"mean_excess":("threshold",m),"return_level":("threshold",levels)})

def gpd_goodness_of_fit(excesses,dim="time"):
    if dim not in excesses.dims:raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    y=_as_1d(excesses.values)
    if y.size<5:return xr.Dataset({"ks_statistic":np.nan,"ks_pvalue":np.nan,"ad_statistic":np.nan,"n_exceedances":y.size})
    from scipy.stats import anderson,kstest
    shape,scale,_=_fit_gpd_1d(y);pit=genpareto.cdf(y,shape,loc=0,scale=scale);ks=kstest(pit,"uniform");z=-np.log(np.clip(1-pit,np.finfo(float).eps,1));ad=anderson(z,dist="expon");order=np.sort(y);p=(np.arange(1,y.size+1)-.5)/y.size;theory=genpareto.ppf(p,shape,loc=0,scale=scale)
    return xr.Dataset({"ks_statistic":float(ks.statistic),"ks_pvalue":float(ks.pvalue),"ad_statistic":float(ad.statistic),"n_exceedances":y.size,"pit":("exceedance",np.sort(pit)),"qq_observed":("exceedance",order),"qq_theoretical":("exceedance",theory)})

def gpd_goodness_of_fit_bootstrap(excesses,dim="time",n_resamples=1000,alpha=.05,random_state=0):
    """Calibrate KS and AD statistics by parametric bootstrap after refitting."""
    if n_resamples<100:raise ValueError("n_resamples must be at least 100.")
    if not 0<alpha<1:raise ValueError("alpha must be between 0 and 1.")
    y=_as_1d(excesses.values)
    if y.size<5:return xr.Dataset({"ks_statistic":np.nan,"ks_pvalue_bootstrap":np.nan,"ad_statistic":np.nan,"ad_pvalue_bootstrap":np.nan,"n_exceedances":y.size})
    from scipy.stats import anderson,kstest
    shape,scale,_=_fit_gpd_1d(y);pit=genpareto.cdf(y,shape,loc=0,scale=scale);obs_ks=kstest(pit,"uniform").statistic;obs_ad=anderson(-np.log(np.clip(1-pit,np.finfo(float).eps,1)),dist="expon").statistic;rng=np.random.default_rng(random_state);ksb=[];adb=[]
    for _ in range(n_resamples):
        sample=genpareto.rvs(shape,loc=0,scale=scale,size=y.size,random_state=rng);bs,bc,_=_fit_gpd_1d(sample);bp=genpareto.cdf(sample,bs,loc=0,scale=bc);ksb.append(kstest(bp,"uniform").statistic);adb.append(anderson(-np.log(np.clip(1-bp,np.finfo(float).eps,1)),dist="expon").statistic)
    ksb=np.asarray(ksb);adb=np.asarray(adb);pk=(1+np.sum(ksb>=obs_ks))/(n_resamples+1);pa=(1+np.sum(adb>=obs_ad))/(n_resamples+1)
    return xr.Dataset({"ks_statistic":obs_ks,"ks_pvalue_bootstrap":float(pk),"ad_statistic":obs_ad,"ad_pvalue_bootstrap":float(pa),"n_exceedances":y.size,"n_resamples":n_resamples,"alpha":alpha})

def pot_threshold_uncertainty(data,thresholds,return_period,dim="time",alpha=.05,n_resamples=500,random_state=0):
    """Propagate threshold choice as an empirical return-level envelope.

    Each candidate threshold is fitted separately; the returned interval is the
    envelope across thresholds, not a probability-weighted confidence interval.
    """
    out=pot_threshold_sensitivity(data,thresholds,return_period,dim);rl=out.return_level.values[np.isfinite(out.return_level.values)]
    if rl.size==0:return xr.Dataset({"return_level_min":np.nan,"return_level_median":np.nan,"return_level_max":np.nan,"threshold_count":0})
    return xr.Dataset({"return_level_min":float(np.min(rl)),"return_level_median":float(np.median(rl)),"return_level_max":float(np.max(rl)),"threshold_count":int(rl.size),"alpha":alpha,"n_resamples":n_resamples})

def gpd_fit_nonstationary(excesses,x,dim="time"):
    """Fit a non-stationary GPD with log-linear scale and constant shape."""
    y=_as_1d(excesses.values);cov=_as_1d(x.values if hasattr(x,"values") else x)
    if y.size!=cov.size:raise ValueError("excesses and covariate must have the same number of finite observations.")
    finite=np.isfinite(y)&np.isfinite(cov);y=y[finite];cov=cov[finite]
    if y.size<10 or np.any(y<0):raise ValueError("at least 10 non-negative finite excesses are required.")
    xc=(cov-np.mean(cov))/(np.std(cov) or 1.0)
    s0= max(float(np.mean(y)),np.finfo(float).eps)
    init=np.array([0.0,np.log(s0),0.0])
    def nll(par):
        xi,b0,b1=par; sigma=np.exp(b0+b1*xc); z=1+xi*y/sigma
        if np.any(sigma<=0) or np.any(z<=0):return 1e100
        if abs(xi)<1e-6:return np.sum(np.log(sigma)+y/sigma)
        return np.sum(np.log(sigma)+(1/xi+1)*np.log(z))
    fit=minimize(nll,init,method="Nelder-Mead",options={"maxiter":5000});
    if not fit.success:raise RuntimeError(f"Non-stationary GPD optimization failed: {fit.message}")
    xi,b0,b1=fit.x
    return xr.Dataset({"shape":float(xi),"log_scale_intercept":float(b0),"log_scale_covariate":float(b1),"covariate_mean":float(np.mean(cov)),"covariate_std":float(np.std(cov) or 1.0),"n_exceedances":int(y.size),"nll":float(fit.fun)})

__all__=["gpd_fit","pot_exceedances","decluster_exceedances","threshold_diagnostics","pot_threshold_sensitivity","pot_threshold_uncertainty","gpd_goodness_of_fit","gpd_goodness_of_fit_bootstrap","gpd_fit_nonstationary","pot_return_level","pot_return_level_ci"]
