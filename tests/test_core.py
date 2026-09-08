import numpy as np
import xarray as xr

from climatevar.spatial import area_weighted_mean, cosine_latitude_weights
from climatevar.thermodynamics import potential_temperature, saturation_vapor_pressure, specific_humidity_from_rh
from climatevar.wrf import nested_grid_spacing


def test_potential_temperature():
    assert np.isclose(potential_temperature(300.0, 100000.0), 300.0)


def test_moisture_round_trip():
    t = 300.0
    p = 100000.0
    rh = 0.5
    q = specific_humidity_from_rh(t, p, rh)
    assert q > 0
    assert saturation_vapor_pressure(t) > 0


def test_area_weighted_mean():
    data = xr.DataArray([1.0, 3.0], coords={"lat": [-60.0, 60.0]}, dims="lat")
    result = area_weighted_mean(data, spatial_dims=("lat",))
    assert np.isclose(result.item(), 2.0)
    assert np.isclose(cosine_latitude_weights(data.lat).mean().item(), 1.0)


def test_nested_spacing():
    assert nested_grid_spacing(9000, [3, 3]) == [9000.0, 3000.0, 1000.0]
