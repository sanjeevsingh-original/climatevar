import numpy as np
import xarray as xr

from climatevar.io import normalize_dataset, standardize_variables


def test_imerg_aliases_are_normalized():
    ds = xr.Dataset(
        {"precipitation": (("time", "lat", "lon"), np.ones((2, 2, 3)))},
        coords={"time": [0, 1], "lat": [-1, 0], "lon": [70, 71, 72]},
    )
    out = normalize_dataset(ds, dataset="imerg")
    assert "latitude" in out.coords
    assert "longitude" in out.coords
    assert "precipitation" in out.data_vars


def test_explicit_variable_mapping_is_preferred():
    ds = xr.Dataset(
        {
            "RAINNC": (("south_north", "west_east"), np.ones((2, 3))),
            "RAINSH": (("south_north", "west_east"), np.ones((2, 3)) * 2),
        },
        coords={
            "XLAT": (("south_north", "west_east"), [[10, 10, 10], [11, 11, 11]]),
            "XLONG": (("south_north", "west_east"), [[70, 71, 72], [70, 71, 72]]),
        },
    )
    out = normalize_dataset(
        ds,
        dataset="wrf",
        variables={"precipitation": "RAINNC"},
    )
    assert "latitude" in out
    assert "longitude" in out
    assert "precipitation" in out.data_vars
    assert "RAINSH" in out.data_vars


def test_standardize_temperature_alias():
    ds = xr.Dataset({"t2m": (("time",), [290.0, 291.0])})
    out = standardize_variables(ds, dataset="era5")
    assert "temperature" in out.data_vars
