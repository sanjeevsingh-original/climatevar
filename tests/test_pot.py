import numpy as np
import xarray as xr
import pytest

from climatevar.extremes import (
    decluster_exceedances,
    gpd_fit,
    gpd_goodness_of_fit,
    pot_exceedances,
    pot_return_level,
    pot_return_level_ci,
    pot_threshold_sensitivity,
    threshold_diagnostics,
)


def test_pot_exceedances_and_fit():
    data = xr.DataArray([0, 2, 3, 8, 1, 12, 2, 7, 15, 4], dims="time")
    excess = pot_exceedances(data, 5)
    assert np.isnan(excess.sel(time=0))
    assert np.allclose(excess.dropna("time"), [3, 7, 2, 10])
    fit = gpd_fit(excess)
    assert fit.n_exceedances.item() == 4


def test_runs_declustering_retains_cluster_maxima():
    data = xr.DataArray([0, 8, 12, 2, 1, 9, 7, 0, 11, 0], dims="time")
    peaks = decluster_exceedances(data, threshold=5, run_length=2)
    assert np.allclose(peaks.dropna("time"), [12, 11])


def test_threshold_diagnostics_and_return_level():
    rng = np.random.default_rng(42)
    data = xr.DataArray(rng.gamma(2.0, 10.0, 2000), dims="time")
    diag = threshold_diagnostics(data, [15, 20, 25, 30])
    assert list(diag.threshold.values) == [15, 20, 25, 30]
    assert np.all(diag.n_exceedances.values[:-1] >= diag.n_exceedances.values[1:])
    assert np.all(np.isfinite(diag.shape.values))
    assert np.all(np.isfinite(diag.mean_excess.values))
    assert pot_return_level(20, 0.1, 10, 0.05, 100) > 20
    assert np.isnan(pot_return_level(20, 0.1, 10, 0.05, 10))


def test_threshold_sensitivity():
    rng = np.random.default_rng(42)
    data = xr.DataArray(rng.gamma(2.0, 10.0, 2000), dims="time")
    out = pot_threshold_sensitivity(data, [15, 20, 25, 30], 100)
    assert "return_level" in out
    assert np.all(np.isfinite(out.mean_excess.values))


def test_gpd_goodness_of_fit_outputs_diagnostics():
    rng = np.random.default_rng(3)
    data = xr.DataArray(rng.exponential(10.0, 500), dims="time")
    out = gpd_goodness_of_fit(data)
    assert out.n_exceedances.item() == 500
    assert np.isfinite(out.ks_statistic.item())
    assert out.pit.size == 500
    assert out.qq_observed.size == 500


def test_invalid_short_return_period():
    with pytest.raises(ValueError):
        pot_return_level_ci(xr.DataArray(np.arange(100.0), dims="time"), 50, 1)


def test_pot_bootstrap_is_reproducible():
    rng = np.random.default_rng(7)
    data = xr.DataArray(rng.gamma(2.0, 10.0, 500), dims="time")
    a = pot_return_level_ci(data, 20, 100, n_resamples=100, random_state=123)
    b = pot_return_level_ci(data, 20, 100, n_resamples=100, random_state=123)
    assert np.isclose(a.return_level.item(), b.return_level.item())
    assert np.isclose(a.ci_lower.item(), b.ci_lower.item())
    assert np.isclose(a.ci_upper.item(), b.ci_upper.item())
