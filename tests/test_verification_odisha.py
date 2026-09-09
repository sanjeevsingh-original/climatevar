import numpy as np
import xarray as xr

from climatevar.verification import ProductSpec, odisha_experiment, run_experiment


def _ds(values, name="pr"):
    time = xr.date_range("2005-06-01", periods=30, freq="D")
    return xr.Dataset(
        {name: (("time", "lat", "lon"), np.asarray(values)[:, None, None])},
        coords={"time": time, "lat": [20.0], "lon": [85.0]},
    )


def _wrf_ds(interval_amount):
    time = xr.date_range("2005-06-01", periods=30, freq="D")
    cumulative = np.cumsum(np.asarray(interval_amount, dtype=float) * 0.9)
    return xr.Dataset(
        {
            "RAINC": (("time", "lat", "lon"), np.zeros((30, 1, 1))),
            "RAINNC": (("time", "lat", "lon"), cumulative[:, None, None]),
        },
        coords={"time": time, "lat": [20.0], "lon": [85.0]},
    )


def test_odisha_configuration_and_block_bootstrap():
    obs = _ds(np.arange(30, dtype=float) % 11)
    candidate = _ds(obs["pr"].values * 0.9)
    wrf = _wrf_ds(obs["pr"].values)
    cfg = odisha_experiment(
        imd=obs,
        era5=candidate,
        imerg=candidate,
        imdaa=candidate,
        wrf=wrf,
        cmip6=candidate,
        imd_variable="pr",
        era5_variable="pr",
        imerg_variable="pr",
        imdaa_variable="pr",
        cmip6_variable="pr",
        start="2005-06-01",
        end="2005-06-30",
        regrid_method=None,
        uncertainty_resamples=20,
        block_length=5,
    )
    result = run_experiment(cfg)
    assert set(result.overall["dataset"]) == {"ERA5", "IMERG", "IMDAA", "WRF", "CMIP6"}
    assert set(result.seasonal["period"]) == {"JJAS"}
    assert not result.uncertainty.empty
    assert {"lower_95", "median", "upper_95"}.issubset(result.uncertainty.columns)
    assert result.spatial.sizes["lat"] == 1
