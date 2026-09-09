import numpy as np
import xarray as xr

from climatevar.verification import ExperimentConfig, ProductSpec, load_precipitation, run_experiment


def _ds(values, *, units="mm/day", var="pr"):
    time = xr.date_range("2000-06-01", periods=6, freq="D")
    da = xr.DataArray(np.asarray(values)[:, None, None], dims=("time", "lat", "lon"), coords={"time": time, "lat": [20.0], "lon": [85.0]}, attrs={"units": units})
    return xr.Dataset({var: da})


def test_generic_cmip6_adapter_normalizes_rate():
    values = np.ones((6, 1, 1)) * 86.4
    ds = _ds(values, units="mm/day")
    da = load_precipitation(ds, family="cmip6")
    assert da.dims == ("time", "lat", "lon")
    assert da.attrs["units"] == "mm/day"
    assert np.allclose(da.values, values)


def test_experiment_runner_produces_all_outputs():
    obs = _ds(np.array([0, 2, 0, 5, 10, 20], float))
    imerg = _ds(np.array([0, 1.8, 0, 4, 8, 18], float))
    imdaa = _ds(np.array([0, 2.2, 0, 6, 11, 21], float))
    cfg = ExperimentConfig(
        reference=ProductSpec("IMD", "imd", obs, variable="pr"),
        products=(ProductSpec("IMERG", "imerg", imerg, variable="pr"), ProductSpec("IMDAA", "imdaa", imdaa, variable="pr")),
        seasons={"JJAS": (6, 7, 8, 9)},
    )
    result = run_experiment(cfg)
    assert set(result.overall["dataset"]) == {"IMERG", "IMDAA"}
    assert set(result.seasonal["period"]) == {"JJAS"}
    assert {"bias", "rmse", "pod"}.issubset(result.spatial.data_vars)
    assert "error_class" in result.error_features
    assert list(result.ranking["rank"]) == [1, 2]


def test_wrf_adapter_uses_total_precipitation():
    time = xr.date_range("2000-06-01", periods=3, freq="D")
    ds = xr.Dataset({"RAINC": (("time", "lat", "lon"), np.array([[[0.]], [[2.]], [[5.]]])), "RAINNC": (("time", "lat", "lon"), np.array([[[0.]], [[3.]], [[7.]]]))}, coords={"time": time, "lat": [20.], "lon": [85.]})
    da = load_precipitation(ds, family="wrf")
    assert np.allclose(da.values.ravel(), [0., 5., 7.])
