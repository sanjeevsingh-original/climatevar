import numpy as np
import xarray as xr

from climatevar.trends import mann_kendall, modified_mann_kendall, sens_slope, trend_free_prewhitening


def test_increasing_series_has_positive_slope_and_tau():
    data = xr.DataArray(np.arange(10, dtype=float), dims="time")
    result = mann_kendall(data)
    assert result.n.item() == 10
    assert result.tau.item() == 1.0
    assert result.pvalue.item() < 0.01
    assert sens_slope(data).item() == 1.0


def test_constant_series_has_zero_sen_slope():
    data = xr.DataArray(np.ones(8), dims="time")
    assert sens_slope(data).item() == 0.0


def test_modified_mk_returns_effective_sample_size():
    rng = np.random.default_rng(42)
    y = np.cumsum(rng.normal(size=80))
    result = modified_mann_kendall(xr.DataArray(y, dims="time"))
    assert result.n.item() == 80
    assert 1 <= result.n_eff.item() <= 80
    assert np.isfinite(result.pvalue.item())


def test_trend_free_prewhitening_preserves_shape():
    data = xr.DataArray(np.arange(20, dtype=float), dims="time")
    result = trend_free_prewhitening(data)
    assert result.shape == data.shape
    assert np.isfinite(result).all()
