import numpy as np
import pandas as pd
import xarray as xr

from climatevar.trends import fdr_mask, seasonal_mann_kendall, seasonal_sen_slope, spatial_trend


def test_seasonal_sen_slope_for_jja():
    time = pd.date_range("2000-01-01", periods=60, freq="MS")
    values = np.arange(60, dtype=float)
    data = xr.DataArray(values, coords={"time": time}, dims="time")
    slope = seasonal_sen_slope(data, "JJA")
    assert slope.item() == 12.0


def test_seasonal_mk_detects_monotonic_jja():
    time = pd.date_range("2000-01-01", periods=60, freq="MS")
    values = np.arange(60, dtype=float)
    data = xr.DataArray(values, coords={"time": time}, dims="time")
    result = seasonal_mann_kendall(data, "JJA")
    assert result.n.item() == 5
    assert result.tau.item() == 1.0


def test_fdr_mask_controls_multiple_tests():
    p = xr.DataArray([0.001, 0.01, 0.20, 0.80], dims="cell")
    mask = fdr_mask(p, alpha=0.05)
    assert mask.values.tolist() == [True, True, False, False]


def test_spatial_trend_returns_slope_and_fdr():
    time = pd.date_range("2000-01-01", periods=10, freq="YS")
    data = xr.DataArray(
        np.stack([np.arange(10), np.arange(10) * -1], axis=1),
        coords={"time": time, "cell": [0, 1]},
        dims=("time", "cell"),
    )
    result = spatial_trend(data)
    assert "sen_slope" in result
    assert "significant_fdr" in result
