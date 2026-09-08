import numpy as np
import pandas as pd
import xarray as xr

from climatevar.precipitation import wrf_precipitation_amount, wrf_total_precipitation


def test_wrf_total_precipitation_combines_rainc_and_rainnc():
    time = pd.date_range("2000-01-01", periods=3, freq="h")
    ds = xr.Dataset({
        "RAINC": ("Time", [1.0, 3.0, 4.0]),
        "RAINNC": ("Time", [2.0, 4.0, 7.0]),
    }, coords={"Time": time})
    result = wrf_total_precipitation(ds)
    np.testing.assert_allclose(result, [3.0, 7.0, 11.0])
    assert result.attrs["climatevar:precipitation_kind"] == "accumulated"


def test_wrf_precipitation_amount_handles_restart_reset():
    time = pd.date_range("2000-01-01", periods=4, freq="h")
    ds = xr.Dataset({
        "RAINC": ("Time", [0.0, 1.0, 0.2, 1.2]),
        "RAINNC": ("Time", [0.0, 2.0, 0.3, 1.3]),
    }, coords={"Time": time})
    result = wrf_precipitation_amount(ds)
    np.testing.assert_allclose(result, [0.0, 3.0, 0.5, 1.0])
    assert result.attrs["units"] == "mm"
    assert result.attrs["climatevar:precipitation_kind"] == "interval_amount"


def test_wrf_precipitation_can_include_rainsh():
    ds = xr.Dataset({
        "RAINC": ("Time", [1.0, 2.0]),
        "RAINNC": ("Time", [2.0, 3.0]),
        "RAINSH": ("Time", [0.5, 1.0]),
    }, coords={"Time": pd.date_range("2000-01-01", periods=2, freq="h")})
    result = wrf_total_precipitation(ds, include_shallow=True)
    np.testing.assert_allclose(result, [3.5, 6.0])
