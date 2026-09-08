import numpy as np
import pandas as pd
import xarray as xr

from climatevar.grid import regrid
from climatevar.io import normalize_dataset
from climatevar.time import infer_frequency
from climatevar.units import to_si


def test_normalize_to_si_units():
    time = pd.date_range("2000-01-01", periods=3, freq="D")
    ds = xr.Dataset(
        {
            "tp": xr.DataArray([1.0, 2.0, 3.0], dims="time", attrs={"units": "mm"}),
            "t2m": xr.DataArray([0.0, 10.0, 20.0], dims="time", attrs={"units": "degC"}),
            "sp": xr.DataArray([1000.0, 1005.0, 1010.0], dims="time", attrs={"units": "hPa"}),
        },
        coords={"time": time, "latitude": 20.0, "longitude": 85.0},
    )
    out = normalize_dataset(ds, dataset="era5")
    assert out["precipitation"].attrs["units"] == "m"
    assert out["temperature"].attrs["units"] == "K"
    assert out["surface_pressure"].attrs["units"] == "Pa"
    np.testing.assert_allclose(out["precipitation"], [0.001, 0.002, 0.003])
    np.testing.assert_allclose(out["temperature"], [273.15, 283.15, 293.15])
    np.testing.assert_allclose(out["surface_pressure"], [100000, 100500, 101000])


def test_precipitation_flux_becomes_si_flux_not_amount():
    time = pd.date_range("2000-01-01", periods=2, freq="D")
    ds = xr.Dataset(
        {"pr": xr.DataArray([86.4, 172.8], dims="time", attrs={"units": "mm/day"})},
        coords={"time": time, "lat": 20.0, "lon": 85.0},
    )
    out = normalize_dataset(ds, dataset="cmip6")
    assert out["precipitation"].attrs["units"] == "kg m-2 s-1"
    assert out["precipitation"].attrs["climatevar:quantity"] == "precipitation_flux"
    np.testing.assert_allclose(out["precipitation"], [1e-3, 2e-3])


def test_frequency_detection():
    time = pd.date_range("2000-01-01", periods=24, freq="h")
    data = xr.DataArray(np.ones(24), coords={"time": time}, dims="time")
    assert infer_frequency(data) == "hourly"


def test_rectilinear_regrid():
    lat = xr.DataArray([0.0, 1.0], dims="latitude")
    lon = xr.DataArray([0.0, 1.0], dims="longitude")
    source = xr.DataArray([[0.0, 1.0], [1.0, 2.0]], coords={"latitude": lat, "longitude": lon}, dims=("latitude", "longitude"))
    target = xr.Dataset(coords={"latitude": [0.5], "longitude": [0.5]})
    out = regrid(source, target)
    assert np.isclose(out.item(), 1.0)


def test_direct_temperature_si_conversion():
    data = xr.DataArray([0.0], attrs={"units": "degC"})
    out = to_si(data, "temperature")
    assert out.attrs["units"] == "K"
    assert np.isclose(out.item(), 273.15)
