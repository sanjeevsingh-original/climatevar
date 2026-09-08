import numpy as np
import xarray as xr

from climatevar.trends import mann_kendall, sens_slope


def test_increasing_series_has_positive_slope_and_tau():
    data = xr.DataArray(np.arange(10, dtype=float), dims="time")
    result = mann_kendall(data)
    assert result.n.item() == 10
    assert result.tau.item() == 1.0
    assert result.pvalue.item() < 0.01
    assert sens_slope(data).item() == 1.0


def test_constant_series_has_zero_sen_slope():
    data = xr.DataArray(np.ones(8), dims="time")
    assert sens_slope(data).item() == 0.0
