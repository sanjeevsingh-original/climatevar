import numpy as np
import xarray as xr

from climatevar.trends import regional_slope_ci, sens_slope_ci


def test_sens_slope_ci_reproducible_and_contains_known_trend():
    rng = np.random.default_rng(42)
    t = np.arange(40, dtype=float)
    y = 0.5 * t + rng.normal(0, 0.3, size=t.size)
    da = xr.DataArray(y, dims="time")

    a = sens_slope_ci(da, n_resamples=200, random_state=7)
    b = sens_slope_ci(da, n_resamples=200, random_state=7)

    assert np.isclose(a.sen_slope, b.sen_slope)
    assert np.isclose(a.ci_lower, b.ci_lower)
    assert np.isclose(a.ci_upper, b.ci_upper)
    assert float(a.ci_lower) <= 0.5 <= float(a.ci_upper)
    assert a.attrs["confidence_level"] == 0.95


def test_regional_slope_ci_uses_area_weighted_series():
    t = np.arange(30, dtype=float)
    lat = xr.DataArray([0.0, 60.0], dims="latitude")
    lon = xr.DataArray([0.0, 1.0], dims="longitude")
    base = 0.2 * t[:, None, None]
    field = xr.DataArray(
        base + np.array([[0.0, 0.0], [1.0, 1.0]])[None, :, :],
        dims=("time", "latitude", "longitude"),
        coords={"time": t, "latitude": lat, "longitude": lon},
    )

    result = regional_slope_ci(field, n_resamples=100, random_state=3)
    assert np.isfinite(result.sen_slope)
    assert float(result.ci_lower) <= float(result.sen_slope) <= float(result.ci_upper)
    assert result.attrs["weighting"] == "cosine latitude"
