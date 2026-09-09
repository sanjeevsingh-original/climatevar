import numpy as np
import xarray as xr

from climatevar.verification import ProductSpec, odisha_experiment, run_experiment


def _ds(values, name="pr"):
    time = xr.date_range("2005-06-01", periods=30, freq="D")
    return xr.Dataset(
        {name: (("time", "lat", "lon"), np.asarray(values)[:, None, None])},
        coords={"time": time, "lat": [20.0], "lon": [85.0]},
    )


def test_odisha_configuration_and_block_bootstrap():
    obs = _ds(np.arange(30, dtype=float) % 11)
    era5 = _ds(obs["pr"].values * 0.9)
    cfg = odisha_experiment(
        imd=obs,
        era5=era5,
        imerg=era5,
        imdaa=era5,
        wrf=era5,
        cmip6=era5,
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
