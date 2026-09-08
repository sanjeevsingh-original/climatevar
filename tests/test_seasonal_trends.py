import numpy as np
import pandas as pd
import xarray as xr

from climatevar.trends import fdr_mask, seasonal_mann_kendall, seasonal_sen_slope, seasonal_series, spatial_trend


def test_seasonal_sen_slope_for_jja():
    time = pd.date_range("2000-01-01", periods=60, freq="MS")
    values = np.arange(60, dtype=float)
    data = xr.DataArray(values, coords={"time": time}, dims="time")
    slope = seasonal_sen_slope(data, "JJA", aggregation="sum")
    assert slope.item() == 12.0


def test_seasonal_mk_detects_monotonic_jja():
    time = pd.date_range("2000-01-01", periods=60, freq="MS")
    values = np.arange(60, dtype=float)
    data = xr.DataArray(values, coords={"time": time}, dims="time")
    result = seasonal_mann_kendall(data, "JJA", aggregation="sum")
    assert result.n.item() == 5
    assert result.tau.item() == 1.0


def test_daily_precipitation_seasonal_total_and_missing_day():
    time = pd.date_range("2000-01-01", "2002-12-31", freq="D")
    data = xr.DataArray(
        np.ones(time.size), coords={"time": time}, dims="time", name="precipitation",
        attrs={"units": "mm/day", "standard_name": "precipitation_flux"},
    )
    result = seasonal_series(data, "JJA")
    assert result.sizes["year"] == 3
    assert np.allclose(result.values, [92.0, 92.0, 92.0])
    missing = data.drop_sel(time=["2001-07-15"])
    incomplete = seasonal_series(missing, "JJA")
    assert np.isnan(incomplete.sel(year=2001))


def test_daily_temperature_uses_mean_in_auto_mode():
    time = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    values = time.month.astype(float)
    data = xr.DataArray(values, coords={"time": time}, dims="time", name="temperature",
                        attrs={"units": "K", "standard_name": "air_temperature"})
    result = seasonal_series(data, "JJA")
    assert np.allclose(result.values, [7.0, 7.0])


def test_djf_assigns_december_to_following_year():
    time = pd.date_range("1999-12-01", "2002-02-01", freq="MS")
    data = xr.DataArray(np.arange(time.size, dtype=float), coords={"time": time}, dims="time")
    result = seasonal_series(data, "DJF")
    assert result["year"].values.tolist() == [2000, 2001, 2002]


def test_fdr_mask_controls_multiple_tests():
    p = xr.DataArray([0.001, 0.01, 0.20, 0.80], dims="cell")
    mask = fdr_mask(p, alpha=0.05)
    assert mask.values.tolist() == [True, True, False, False]


def test_spatial_trend_returns_slope_and_fdr():
    time = pd.date_range("2000-01-01", periods=10, freq="YS")
    data = xr.DataArray(
        np.stack([np.arange(10), np.arange(10) * -1], axis=1),
        coords={"time": time, "cell": [0, 1]}, dims=("time", "cell"),
    )
    result = spatial_trend(data)
    assert "sen_slope" in result
    assert "significant_fdr" in result
