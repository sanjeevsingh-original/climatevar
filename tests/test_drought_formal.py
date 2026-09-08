import numpy as np
import pandas as pd
import xarray as xr

from climatevar.drought import spei, spi


def test_spi_gamma_returns_standardized_values():
    time = pd.date_range("1990-01-01", periods=120, freq="MS")
    rain = xr.DataArray(
        np.tile(np.arange(1.0, 11.0), 12),
        dims="time",
        coords={"time": time},
    )
    result = spi(rain, scale=3)
    assert result.name == "spi"
    assert result.attrs["climatevar:distribution"] == "gamma"
    assert np.isfinite(result.isel(time=slice(3, None))).any()


def test_spi_handles_zero_precipitation():
    time = pd.date_range("1990-01-01", periods=120, freq="MS")
    values = np.tile([0.0, 2.0, 4.0, 8.0, 12.0], 24)[:120]
    result = spi(xr.DataArray(values, dims="time", coords={"time": time}), scale=1)
    assert np.isfinite(result).sum() > 0


def test_spi_pearson3():
    time = pd.date_range("1990-01-01", periods=120, freq="MS")
    rain = xr.DataArray(np.linspace(1, 20, 120), dims="time", coords={"time": time})
    result = spi(rain, scale=3, distribution="pearson3")
    assert result.name == "spi"


def test_spei_returns_index():
    time = pd.date_range("1990-01-01", periods=120, freq="MS")
    p = xr.DataArray(np.full(120, 100.0), dims="time", coords={"time": time})
    pet = xr.DataArray(np.full(120, 80.0), dims="time", coords={"time": time})
    result = spei(p, pet, scale=3)
    assert result.name == "spei"
    assert result.attrs["climatevar:distribution"] == "log-logistic (Fisk)"
