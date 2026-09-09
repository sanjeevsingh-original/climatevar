import numpy as np
import xarray as xr

from climatevar.error_tagging import (
    build_error_features, error_class, event_group_split,
    spatial_block_split, temporal_split, validate_no_overlap,
)


def test_error_classes_and_zero_rainfall():
    obs = xr.DataArray([0.0, 10.0, 10.0, 10.0], dims="time")
    pred = xr.DataArray([0.0, 2.0, 7.0, 20.0], dims="time")
    out = error_class(obs, pred)
    assert out.values.tolist() == ["near_zero", "underestimate_severe", "underestimate_moderate", "overestimate_severe"]


def test_feature_builder_preserves_target_and_metadata():
    time = xr.date_range("2020-01-01", periods=3, freq="D", use_cftime=False)
    obs = xr.DataArray([0., 5., 20.], dims="time", coords={"time": time})
    pred = xr.DataArray([0., 10., 10.], dims="time", coords={"time": time})
    ds = build_error_features(obs, pred)
    assert {"observation", "prediction", "error", "absolute_error", "relative_error", "error_class"}.issubset(ds.data_vars)
    assert ds["month"].values.tolist() == [1, 1, 1]
    assert ds["observed_wet"].values.tolist() == [0, 1, 1]


def test_temporal_and_group_splits_do_not_overlap():
    splits = temporal_split(np.arange(20), train_fraction=0.6, validation_fraction=0.2)
    assert splits["train"][-1] < splits["validation"][0] < splits["test"][0]
    groups = np.repeat(np.arange(10), 2)
    gs = event_group_split(groups, train_fraction=0.6, validation_fraction=0.2)
    assert validate_no_overlap(gs, groups)


def test_spatial_blocks_hold_out_complete_blocks():
    lat = np.array([0.1, 0.2, 0.3, 2.1, 2.2, 2.3])
    lon = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3])
    split = spatial_block_split(lat, lon, lat_block=1, lon_block=1, test_fraction=0.5, random_state=0)
    assert len(np.intersect1d(split["train"], split["test"])) == 0
    assert len(np.unique(np.floor(lat[split["test"]]))) == 1
