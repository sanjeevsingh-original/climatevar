import numpy as np
import xarray as xr

from climatevar.metrics import bias, correlation, f1_score, mae, nse, rmse


def test_continuous_metrics():
    obs = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="time")
    pred = xr.DataArray([2.0, 2.0, 4.0, 3.0], dims="time")
    assert np.isclose(bias(obs, pred).item(), 0.25)
    assert np.isclose(mae(obs, pred).item(), 0.75)
    assert np.isclose(rmse(obs, pred).item(), np.sqrt(0.75))
    assert np.isclose(correlation(obs, pred).item(), 0.674199862463242)
    assert np.isclose(nse(obs, pred).item(), 0.4)


def test_binary_f1():
    obs = xr.DataArray([0, 1, 1, 0], dims="time")
    pred = xr.DataArray([0, 1, 0, 1], dims="time")
    assert np.isclose(f1_score(obs, pred, threshold=0.5).item(), 0.5)
