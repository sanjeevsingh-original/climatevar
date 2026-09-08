import numpy as np
import xarray as xr

from climatevar.trends import field_significance, regional_mean, regional_trend


def _field():
    time = np.arange(16)
    lat = np.array([10.0, 20.0])
    lon = np.array([80.0, 90.0])
    base = np.arange(16, dtype=float)[:, None, None]
    return xr.DataArray(
        base + np.zeros((16, 2, 2)),
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": lat, "longitude": lon},
        name="rainfall",
    )


def test_regional_mean_and_trend():
    data = _field()
    mean = regional_mean(data)
    assert mean.dims == ("time",)
    result = regional_trend(data)
    assert float(result["pvalue"]) < 0.05
    assert float(result["sen_slope"]) > 0


def test_field_significance_detects_strong_common_trend():
    data = _field()
    result = field_significance(data, block_length=3, n_resamples=25, random_state=1)
    assert int(result["observed_count"]) == 4
    assert float(result["field_pvalue"]) < 0.05
    assert result.attrs["spatial_dependence"]


def test_field_significance_preserves_field_shape():
    data = _field()
    result = field_significance(data, block_length=2, n_resamples=10, random_state=2)
    assert result["pvalue"].dims == ("latitude", "longitude")
    assert result["significant_fdr"].shape == (2, 2)
    assert result["null_counts"].dims == ("resample",)
