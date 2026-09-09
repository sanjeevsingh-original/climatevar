import numpy as np
import pandas as pd
import xarray as xr
from climatevar.precipitation.indices import *

KW = dict(min_valid_fraction=0.0)

def make_data(units=None):
    time = pd.date_range("2000-01-01", periods=365, freq="D")
    values = np.zeros(365); values[100] = 30
    return xr.DataArray(values, coords={"time": time}, dims="time", attrs={} if units is None else {"units": units})

def test_missing_units_are_implicitly_treated_as_mm():
    data = make_data().drop_attrs()
    assert np.isclose(rx1day(data, **KW).item(), 30)

def test_daily_rate_is_accepted_implicitly():
    data = make_data("mm/day")
    result = rx1day(data, **KW)
    assert np.isclose(result.item(), 30)

def test_hourly_rate_is_aggregated_to_daily_amount():
    time = pd.date_range("2000-01-01", periods=48, freq="h")
    data = xr.DataArray(np.ones(48), coords={"time": time}, dims="time", attrs={"units": "mm/hr"})
    assert np.isclose(rx1day(data, **KW).item(), 24)
