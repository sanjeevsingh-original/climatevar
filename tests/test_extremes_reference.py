"""Independent analytical regression checks for EVT formulas."""
import numpy as np
import xarray as xr
from scipy.stats import genpareto
from climatevar.extremes import pot_return_level, gpd_fit, gpd_model_comparison


def test_pot_return_level_matches_gpd_survival_inversion():
    u, xi, sigma, rate, rp = 100.0, 0.2, 30.0, 0.01, 200.0
    expected = u + sigma / xi * ((rate * rp) ** xi - 1.0)
    assert np.isclose(pot_return_level(u, xi, sigma, rate, rp), expected)


def test_gpd_fit_recovers_known_distribution_reasonably():
    rng = np.random.default_rng(1234)
    y = genpareto.rvs(0.15, loc=0, scale=20.0, size=5000, random_state=rng)
    fit = gpd_fit(xr.DataArray(y, dims="time"))
    assert abs(fit.shape.item() - 0.15) < 0.05
    assert abs(fit.scale.item() - 20.0) < 2.0


def test_stationary_null_comparison_is_finite():
    rng = np.random.default_rng(22)
    x = np.linspace(-1, 1, 300)
    y = genpareto.rvs(0.1, loc=0, scale=15, size=x.size, random_state=rng)
    out = gpd_model_comparison(xr.DataArray(y, dims="time"), xr.DataArray(x, dims="time"))
    assert np.isfinite(out.lr_statistic.item())
    assert np.isfinite(out.lr_pvalue_asymptotic.item())
