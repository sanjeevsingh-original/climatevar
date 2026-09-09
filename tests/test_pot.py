import numpy as np
import xarray as xr
import pytest
from climatevar.extremes import decluster_exceedances, gpd_fit, gpd_goodness_of_fit, gpd_goodness_of_fit_bootstrap, gpd_fit_nonstationary, pot_exceedances, pot_return_level, pot_return_level_ci, pot_threshold_sensitivity, pot_threshold_uncertainty, threshold_diagnostics

def test_pot_exceedances_and_fit():
    data=xr.DataArray([0,2,3,8,1,12,2,7,15,4],dims="time"); excess=pot_exceedances(data,5); assert np.isnan(excess.sel(time=0)); assert np.allclose(excess.dropna("time"),[3,7,2,10]); assert gpd_fit(excess).n_exceedances.item()==4

def test_runs_declustering_retains_cluster_maxima():
    data=xr.DataArray([0,8,12,2,1,9,7,0,11,0],dims="time"); peaks=decluster_exceedances(data,5,2); assert np.allclose(peaks.dropna("time"),[12,11])

def test_threshold_diagnostics_and_return_level():
    rng=np.random.default_rng(42); data=xr.DataArray(rng.gamma(2.,10.,2000),dims="time"); diag=threshold_diagnostics(data,[15,20,25,30]); assert list(diag.threshold.values)==[15,20,25,30]; assert np.all(diag.n_exceedances.values[:-1]>=diag.n_exceedances.values[1:]); assert np.all(np.isfinite(diag.shape.values)); assert np.all(np.isfinite(diag.mean_excess.values)); assert pot_return_level(20,.1,10,.05,100)>20
    with pytest.raises(ValueError): pot_return_level(20,.1,10,.05,10)

def test_threshold_sensitivity_and_uncertainty():
    rng=np.random.default_rng(42); data=xr.DataArray(rng.gamma(2.,10.,2000),dims="time"); out=pot_threshold_sensitivity(data,[15,20,25,30],100); assert "return_level" in out; assert np.all(np.isfinite(out.mean_excess.values)); u=pot_threshold_uncertainty(data,[15,20,25,30],100); assert u.threshold_count.item()==4; assert u.return_level_max.item()>=u.return_level_min.item()

def test_gpd_goodness_of_fit_outputs_diagnostics():
    rng=np.random.default_rng(3); data=xr.DataArray(rng.exponential(10.,500),dims="time"); out=gpd_goodness_of_fit(data); assert out.n_exceedances.item()==500; assert np.isfinite(out.ks_statistic.item()); assert out.pit.size==500; assert out.qq_observed.size==500

def test_gpd_goodness_of_fit_bootstrap_is_reproducible():
    rng=np.random.default_rng(8); data=xr.DataArray(rng.exponential(10.,200),dims="time"); a=gpd_goodness_of_fit_bootstrap(data,n_resamples=100,random_state=12); b=gpd_goodness_of_fit_bootstrap(data,n_resamples=100,random_state=12); assert np.isclose(a.ks_pvalue_bootstrap,b.ks_pvalue_bootstrap); assert np.isclose(a.ad_pvalue_bootstrap,b.ad_pvalue_bootstrap)

def test_nonstationary_gpd_fit():
    rng=np.random.default_rng(2); x=np.linspace(-1,1,300); scale=np.exp(np.log(10)+.35*x); y=rng.exponential(scale); out=gpd_fit_nonstationary(xr.DataArray(y,dims="time"),xr.DataArray(x,dims="time")); assert np.isfinite(out.shape); assert out.n_exceedances.item()==300

def test_pot_bootstrap_is_reproducible():
    rng=np.random.default_rng(7); data=xr.DataArray(rng.gamma(2.,10.,500),dims="time"); a=pot_return_level_ci(data,20,100,n_resamples=100,random_state=123); b=pot_return_level_ci(data,20,100,n_resamples=100,random_state=123); assert np.isclose(a.return_level.item(),b.return_level.item()); assert np.isclose(a.ci_lower.item(),b.ci_lower.item()); assert np.isclose(a.ci_upper.item(),b.ci_upper.item())
