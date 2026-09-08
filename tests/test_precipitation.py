import numpy as np
import pandas as pd
import xarray as xr

from climatevar.precipitation import cdd, cwd, prcptot, r10mm, r20mm, r95p, r99p, rx1day, rx5day


def make_data():
    time = pd.date_range("2000-01-01", periods=10, freq="D")
    values = np.array([0, 2, 10, 20, 5, 0, 0, 1, 30, 4], dtype=float)
    return xr.DataArray(values, coords={"time": time}, dims="time")


def test_rx1day():
    assert rx1day(make_data()).item() == 30


def test_rx5day():
    assert rx5day(make_data()).item() == 37


def test_prcptot_and_threshold_counts():
    data = make_data()
    assert prcptot(data).item() == 72
    assert r10mm(data).item() == 3
    assert r20mm(data).item() == 2


def test_percentile_indices_with_explicit_reference():
    data = make_data()
    threshold = xr.DataArray(9.0)
    assert r95p(data, reference=threshold).item() == 60
    assert r99p(data, reference=threshold).item() == 60


def test_consecutive_days():
    data = make_data()
    assert cwd(data).item() == 4
    assert cdd(data).item() == 2
