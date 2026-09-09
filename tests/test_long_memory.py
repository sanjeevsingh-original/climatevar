"""Tests for persistence diagnostics."""
import numpy as np
import xarray as xr
from climatevar.trends import dfa_hurst


def test_dfa_white_noise_is_near_half():
    rng = np.random.default_rng(123)
    data = xr.DataArray(rng.normal(size=2000), dims="time")
    result = dfa_hurst(data, min_scale=10, max_scale=300, n_scales=18)
    assert np.isfinite(result.dfa_exponent.item())
    assert 0.35 < result.dfa_exponent.item() < 0.65
    assert result.r2.item() > 0.85


def test_dfa_reproducible():
    rng = np.random.default_rng(4)
    data = xr.DataArray(rng.normal(size=500), dims="time")
    a = dfa_hurst(data, min_scale=8, max_scale=100, n_scales=12)
    b = dfa_hurst(data, min_scale=8, max_scale=100, n_scales=12)
    np.testing.assert_allclose(a.dfa_exponent, b.dfa_exponent)
    np.testing.assert_allclose(a.r2, b.r2)
