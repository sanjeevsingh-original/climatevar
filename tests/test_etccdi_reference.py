"""Independent regression fixtures for core ETCCDI precipitation indices.

The expected values in this module are calculated directly from the published
index definitions rather than by reusing climatevar internals.  This makes the
suite useful as a reference layer for future refactors and for comparisons with
RClimDex/ClimDex-style implementations.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climatevar.precipitation import (
    cdd,
    cwd,
    prcptot,
    r10mm,
    r20mm,
    r95p,
    r95p_fraction,
    r99p,
    rx1day,
    rx5day,
    sdii,
    wet_day_count,
)


def _fixture():
    # Deliberately asymmetric sequence so thresholds, runs and rolling sums
    # cannot pass because of a trivial constant-value implementation.
    values = np.array(
        [0, 1, 2, 9, 10, 20, 0, 0, 30, 1, 11, 21, 0, 4, 5, 0, 0, 0, 40, 2],
        dtype=float,
    )
    time = pd.date_range("2001-01-01", periods=values.size, freq="D")
    return xr.DataArray(values, coords={"time": time}, dims="time", attrs={"units": "mm"})


def test_core_indices_match_independent_definition_fixture():
    data = _fixture()
    kw = {"min_valid_fraction": 0}

    # Directly computed from the ETCCDI/RClimDex definitions:
    # wet day >= 1 mm; dry day < 1 mm; annual January-December aggregation.
    wet = data.values[data.values >= 1]
    expected_prcptot = wet.sum()
    expected_sdii = wet.mean()
    expected_r10 = np.count_nonzero(data.values >= 10)
    expected_r20 = np.count_nonzero(data.values >= 20)

    dry_runs = []
    wet_runs = []
    for condition, target in [(data.values < 1, dry_runs), (data.values >= 1, wet_runs)]:
        current = 0
        for flag in condition:
            current = current + 1 if flag else 0
            target.append(current)
    expected_cdd = max(dry_runs)
    expected_cwd = max(wet_runs)

    expected_rx1 = data.values.max()
    expected_rx5 = max(np.sum(data.values[i : i + 5]) for i in range(data.size - 4))

    assert prcptot(data, **kw).item() == expected_prcptot
    assert np.isclose(sdii(data, **kw).item(), expected_sdii)
    assert wet_day_count(data, **kw).item() == wet.size
    assert r10mm(data, **kw).item() == expected_r10
    assert r20mm(data, **kw).item() == expected_r20
    assert cdd(data, **kw).item() == expected_cdd
    assert cwd(data, **kw).item() == expected_cwd
    assert rx1day(data, **kw).item() == expected_rx1
    assert rx5day(data, **kw).item() == expected_rx5


def test_percentile_and_fraction_match_independent_reference():
    data = _fixture()
    kw = {"min_valid_fraction": 0}
    wet = data.values[data.values >= 1]
    p95 = float(np.quantile(wet, 0.95))
    p99 = float(np.quantile(wet, 0.99))
    expected95 = data.values[data.values > p95].sum()
    expected99 = data.values[data.values > p99].sum()
    total = wet.sum()

    assert np.isclose(r95p(data, reference=p95, **kw).item(), expected95)
    assert np.isclose(r99p(data, reference=p99, **kw).item(), expected99)
    assert np.isclose(r95p_fraction(data, reference=p95, **kw).item(), 100 * expected95 / total)


def test_exact_wet_day_boundary_is_included():
    data = xr.DataArray(
        [0.999999, 1.0, 1.000001, 9.999999, 10.0, 20.0],
        coords={"time": pd.date_range("2001-01-01", periods=6)},
        dims="time",
        attrs={"units": "mm"},
    )
    kw = {"min_valid_fraction": 0}
    assert wet_day_count(data, **kw).item() == 5
    assert prcptot(data, **kw).item() == pytest.approx(42.0)
    assert r10mm(data, **kw).item() == 2
    assert r20mm(data, **kw).item() == 1


def test_annual_completeness_threshold_is_explicit_and_reproducible():
    time = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    data = xr.DataArray(np.ones(time.size), coords={"time": time}, dims="time")

    # 37 missing days leaves 328/365 = 0.8986, below the default 0.90.
    incomplete = data.copy()
    incomplete["time"] = time
    incomplete.values[:37] = np.nan

    assert np.isfinite(prcptot(data).item())
    assert np.isnan(prcptot(incomplete).item())
    assert np.isfinite(prcptot(incomplete, min_valid_fraction=0.89).item())


def test_leap_year_is_not_penalized_for_having_366_days():
    time = pd.date_range("2000-01-01", "2000-12-31", freq="D")
    data = xr.DataArray(np.ones(time.size), coords={"time": time}, dims="time")
    assert data.size == 366
    assert prcptot(data).item() == 366


def test_noleap_calendar_uses_365_day_year():
    time = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    data = xr.DataArray(np.ones(time.size), coords={"time": time}, dims="time")
    data["time"].attrs["calendar"] = "noleap"
    assert prcptot(data).item() == 365


def test_360_day_calendar_if_cftime_is_available():
    cftime = pytest.importorskip("cftime")
    time = xr.cftime_range("2001-01-01", periods=360, freq="D", calendar="360_day")
    data = xr.DataArray(np.ones(360), coords={"time": time}, dims="time")
    assert data.time.values[59].month == 2
    assert data.time.values[359].month == 12
    assert prcptot(data).item() == 360
