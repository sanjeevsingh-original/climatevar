import numpy as np
import pandas as pd
import xarray as xr

from climatevar.precipitation import (
    cdd, cwd, daily_amount, normalize_precipitation, prcptot, r10mm, r20mm,
    r95p, r99p, rx1day, rx5day,
)


def make_data(units="mm"):
    time = pd.date_range("2000-01-01", periods=10, freq="D")
    values = np.array([0, 2, 10, 20, 5, 0, 0, 1, 30, 4], dtype=float)
    return xr.DataArray(values, coords={"time": time}, dims="time", attrs={"units": units})

KW = {"min_valid_fraction": 0}


def test_rx1day():
    result = rx1day(make_data(), **KW); assert result.item() == 30; assert result.attrs["units"] == "mm"

def test_rx5day():
    result = rx5day(make_data(), **KW); assert result.item() == 37; assert result.attrs["units"] == "mm"

def test_prcptot_and_threshold_counts():
    data = make_data(); assert prcptot(data, **KW).item() == 72; assert r10mm(data, **KW).item() == 3; assert r20mm(data, **KW).item() == 2

def test_percentile_indices_with_explicit_reference():
    data = make_data(); threshold = xr.DataArray(9.0, attrs={"units": "mm"}); assert r95p(data, reference=threshold, **KW).item() == 60; assert r99p(data, reference=threshold, **KW).item() == 60

def test_consecutive_days():
    data = make_data(); assert cwd(data, **KW).item() == 4; assert cdd(data, **KW).item() == 2

def test_amount_units_are_normalized_to_mm():
    result = rx1day(make_data("m"), **KW); assert result.item() == 30000; assert result.attrs["units"] == "mm"

def test_missing_units_are_implicitly_treated_as_mm():
    data = make_data().drop_attrs(); assert rx1day(data, **KW).item() == 30

def test_daily_rate_is_accepted_implicitly():
    data = make_data("mm/day"); result = rx1day(data, **KW); assert np.isclose(result.item(), 30)

def test_hourly_rate_is_aggregated_to_daily_amount():
    time = pd.date_range("2000-01-01", periods=48, freq="h")
    data = xr.DataArray(np.ones(48), coords={"time": time}, dims="time", attrs={"units": "mm/hr"})
    daily = daily_amount(data); assert daily.attrs["units"] == "mm"; assert np.allclose(daily.values, [24.0, 24.0]); assert np.isclose(rx1day(data, **KW).item(), 24.0)

def test_native_normalization_uses_sampling_resolution():
    hourly = xr.DataArray(np.ones(24), coords={"time": pd.date_range("2000-01-01", periods=24, freq="h")}, dims="time", attrs={"units": "kg m-2 s-1"})
    daily = make_data("mm"); hourly_result = normalize_precipitation(hourly); daily_result = normalize_precipitation(daily)
    assert hourly_result.attrs["units"] == "mm/hr"; assert daily_result.attrs["units"] == "mm/day"; assert np.allclose(hourly_result.values, 3600.0); assert np.allclose(daily_result.values, daily.values)

def test_percentile_reference_accepts_common_amount_units():
    data = make_data(); threshold = xr.DataArray(0.009, attrs={"units": "m"}); assert r95p(data, reference=threshold, **KW).item() == 60

def test_unsupported_precipitation_units_raise():
    data = make_data("inches")
    try: rx1day(data, **KW)
    except ValueError as exc: assert "Unsupported precipitation units" in str(exc)
    else: raise AssertionError("Unsupported precipitation units should raise ValueError")
