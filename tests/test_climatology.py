import numpy as np
import pandas as pd
import xarray as xr

from climatevar.climatology import anomaly, climatology


def test_monthly_climatology():
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    values = np.arange(24, dtype=float)
    data = xr.DataArray(values, coords={"time": time}, dims="time")

    result = climatology(data)

    assert result.sizes["month"] == 12
    np.testing.assert_allclose(result.sel(month=1), 6.0)
    np.testing.assert_allclose(result.sel(month=12), 17.0)


def test_monthly_anomaly():
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    values = np.arange(24, dtype=float)
    data = xr.DataArray(values, coords={"time": time}, dims="time")
    reference = climatology(data)

    result = anomaly(data, reference)

    np.testing.assert_allclose(result.values, np.r_[np.full(12, -6.0), np.full(12, 6.0)])
