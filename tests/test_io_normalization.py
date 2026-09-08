import numpy as np
import pandas as pd
import xarray as xr

from climatevar.io import normalize_dataset, normalize_longitude, precipitation_to_mm, temperature_to_celsius
from climatevar.io.stations import normalize_station_dataframe


def test_era5_aliases():
    ds = xr.Dataset({"tp": (("time", "latitude", "longitude"), np.ones((2, 2, 2)))}, coords={"time": pd.date_range("2000-01-01", periods=2), "latitude": [20, 21], "longitude": [85, 86]})
    out = normalize_dataset(ds, dataset="era5")
    assert "precipitation" in out


def test_wrf_aliases_and_coordinates():
    ds = xr.Dataset({"RAINNC": (("Time", "south_north", "west_east"), np.ones((2, 2, 2))), "XLAT": (("Time", "south_north", "west_east"), np.ones((2, 2, 2))), "XLONG": (("Time", "south_north", "west_east"), np.ones((2, 2, 2)))}, coords={"Time": np.arange(2)})
    out = normalize_dataset(ds, dataset="wrf", variables={"precipitation": "RAINNC"}, strict=False)
    assert "precipitation" in out
    assert "latitude" in out and "longitude" in out


def test_units_and_longitude():
    rain = xr.DataArray([0.001, 0.002], dims="time", coords={"time": pd.date_range("2000-01-01", periods=2)}, attrs={"units": "m"})
    assert precipitation_to_mm(rain).isel(time=0).item() == 1
    temp = xr.DataArray([273.15], attrs={"units": "K"})
    assert temperature_to_celsius(temp).item() == 0
    ds = xr.Dataset(coords={"longitude": [0, 180, 270], "latitude": [20, 10]})
    assert normalize_longitude(ds).longitude.max().item() == 180


def test_station_aliases():
    df = pd.DataFrame({"Date": ["2000-01-01"], "LAT": [20], "LON": [85], "rainfall": [4]})
    out = normalize_station_dataframe(df)
    assert {"time", "latitude", "longitude", "precipitation"}.issubset(out.columns)
