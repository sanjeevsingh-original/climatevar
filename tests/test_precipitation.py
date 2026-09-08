import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climatevar.precipitation import cdd, cwd, prcptot, r10mm, r20mm, r95p, r99p, rx1day, rx5day


def make_data(units="mm"):
    time = pd.date_range("2000-01-01", periods=10, freq="D")
    values = np.array([0, 2, 10, 20, 5, 0, 0, 1, 30, 4], dtype=float)
    return xr.DataArray(values, coords={"time": time}, dims="time", attrs={"units": units})


def test_rx1day():
    result = rx1day(make_data())
    assert result.item() == 30
    assert result.attrs["units"] == "mm"


def test_rx5day():
    result = rx5day(make_data())
    assert result.item() == 37
    assert result.attrs["units"] == "mm"


def test_prcptot_and_threshold_counts():
    data = make_data()
    assert prcptot(data).item() == 72
    assert r10mm(data).item() == 3
    assert r20mm(data).item() == 2


def test_percentile_indices_with_explicit_reference():
    data = make_data()
    threshold = xr.DataArray(9.0, attrs={"units": "mm"})
    assert r95p(data, reference=threshold).item() == 60
    assert r99p(data, reference=threshold).item() == 60


def test_consecutive_days():
    data = make_data()
    assert cwd(data).item() == 4
    assert cdd(data).item() == 2


def test_amount_units_are_normalized_to_mm():
    result = rx1day(make_data("m"))
    assert result.item() == 30000
    assert result.attrs["units"] == "mm"


def test_missing_units_are_rejected():
    data = make_data().drop_attrs()
    with pytest.raises(ValueError, match="explicit precipitation units"):
        rx1day(data)


def test_precipitation_rate_is_rejected():
    data = make_data("mm/day")
    with pytest.raises(ValueError, match="rates/fluxes"):
        rx1day(data)


def test_percentile_reference_must_be_mm():
    data = make_data()
    threshold = xr.DataArray(0.009, attrs={"units": "m"})
    with pytest.raises(ValueError, match="units='mm'"):
        r95p(data, reference=threshold)
