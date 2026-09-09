import numpy as np
import xarray as xr
from climatevar.metrics import (
    bias_ratio, brier_score, equitable_threat_score, fractions_skill_score,
    frequency_bias, reliability_components, threat_score,
)

def test_event_scores_perfect_forecast():
    o = xr.DataArray([0, 2, 0, 5], dims="time")
    assert threat_score(o, o).item() == 1
    assert equitable_threat_score(o, o).item() == 1
    assert frequency_bias(o, o).item() == 1

def test_bias_ratio_and_brier():
    o = xr.DataArray([0., 1., 2.], dims="time")
    p = xr.DataArray([0., 2., 4.], dims="time")
    assert np.isclose(bias_ratio(o, p, dim="time"), 2)
    y = xr.DataArray([0., 1., 1.], dims="time")
    prob = xr.DataArray([0., 1., 0.5], dims="time")
    assert np.isclose(brier_score(y, prob, dim="time"), 1/12)

def test_reliability_components_and_fss():
    y = xr.DataArray([0., 1., 1., 0.], dims="time")
    p = xr.DataArray([0.1, 0.9, 0.9, 0.1], dims="time")
    out = reliability_components(y, p, bins=2)
    assert np.isclose(out.reliability, 0.01)
    field = xr.DataArray(np.ones((3,3)), dims=("latitude","longitude"))
    assert np.isclose(fractions_skill_score(field, field, threshold=0.5).item(), 1)
