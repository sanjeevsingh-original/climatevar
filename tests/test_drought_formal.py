import numpy as np
import pandas as pd
import xarray as xr

from climatevar.drought import drought_category, fit_quality, spei, spi


def test_spi_gamma_returns_standardized_values():
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    rain = xr.DataArray(np.tile(np.arange(1.0, 41.0), 12), dims="time", coords={"time": time})
    result = spi(rain, scale=3, calibration_start="1980-01-01", calibration_end="2009-12-31")
    assert result.name == "spi"
    assert result.attrs["climatevar:distribution"] == "gamma"
    assert result.attrs["climatevar:calibration_start"] == "1980-01-01"
    assert np.isfinite(result.isel(time=slice(3, None))).any()


def test_spi_handles_zero_precipitation():
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    values = np.tile([0.0, 2.0, 4.0, 8.0, 12.0], 96)[:480]
    result = spi(xr.DataArray(values, dims="time", coords={"time": time}), scale=1, min_nonzero=10)
    assert np.isfinite(result).sum() > 0


def test_spi_pearson3():
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    rain = xr.DataArray(np.linspace(1, 20, 480), dims="time", coords={"time": time})
    result = spi(rain, scale=3, distribution="pearson3", min_nonzero=10)
    assert result.name == "spi"


def test_spei_returns_index():
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    p = xr.DataArray(np.full(480, 100.0), dims="time", coords={"time": time})
    pet = xr.DataArray(np.full(480, 80.0), dims="time", coords={"time": time})
    result = spei(p, pet, scale=3, min_samples=10)
    assert result.name == "spei"
    assert result.attrs["climatevar:distribution"] == "three-parameter log-logistic (Fisk)"


def test_drought_categories():
    index = xr.DataArray([-2.2, -1.7, -1.2, 0.0, 1.2, 1.7, 2.2], dims="time")
    result = drought_category(index)
    assert result.values.tolist() == [
        "extreme drought", "severe drought", "moderate drought",
        "near normal", "moderately wet", "severely wet", "extremely wet",
    ]


def test_fit_quality():
    time = pd.date_range("1980-01-01", periods=480, freq="MS")
    rain = xr.DataArray(np.tile(np.arange(1.0, 41.0), 12), dims="time", coords={"time": time})
    result = spi(rain, scale=1, min_nonzero=10)
    quality = fit_quality(result)
    assert set(quality.data_vars) == {"mean", "std", "normality_pvalue", "n"}
