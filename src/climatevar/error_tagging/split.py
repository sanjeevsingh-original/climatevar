"""Leakage-safe train/validation/test splitting for climate ML."""
from __future__ import annotations

import numpy as np
import xarray as xr


def temporal_split(time, *, train_fraction=0.70, validation_fraction=0.15):
    """Return chronological train/validation/test index arrays."""
    n = len(time)
    if n < 3 or not (0 < train_fraction < 1) or not (0 <= validation_fraction < 1):
        raise ValueError("invalid split fractions")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1")
    i1 = int(np.floor(n * train_fraction))
    i2 = int(np.floor(n * (train_fraction + validation_fraction)))
    return {"train": np.arange(0, i1), "validation": np.arange(i1, i2), "test": np.arange(i2, n)}


def event_group_split(groups, *, train_fraction=0.70, validation_fraction=0.15):
    """Split whole events/groups, never individual rows within an event."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    splits = temporal_split(unique, train_fraction=train_fraction, validation_fraction=validation_fraction)
    return {name: np.flatnonzero(np.isin(groups, unique[idx])) for name, idx in splits.items()}


def spatial_block_split(lat, lon, *, lat_block=2.0, lon_block=2.0, test_fraction=0.20, random_state=0):
    """Hold out complete latitude/longitude blocks to test spatial transfer."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    if lat.shape != lon.shape:
        raise ValueError("lat and lon must have the same shape")
    if not (0 < test_fraction < 1) or lat_block <= 0 or lon_block <= 0:
        raise ValueError("invalid spatial split parameters")
    blocks = np.column_stack((np.floor(lat / lat_block), np.floor(lon / lon_block)))
    unique = np.unique(blocks, axis=0)
    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(unique))
    n_test = max(1, int(np.ceil(len(unique) * test_fraction)))
    test_blocks = unique[order[:n_test]]
    is_test = np.zeros(len(lat), dtype=bool)
    for block in test_blocks:
        is_test |= np.all(blocks == block, axis=1)
    return {"train": np.flatnonzero(~is_test), "test": np.flatnonzero(is_test)}


def validate_no_overlap(splits, groups):
    """Raise if a group appears in more than one split; return True otherwise."""
    groups = np.asarray(groups)
    seen = {}
    for name, idx in splits.items():
        for group in np.unique(groups[np.asarray(idx, dtype=int)]):
            if group in seen:
                raise ValueError(f"group {group!r} occurs in both {seen[group]!r} and {name!r}")
            seen[group] = name
    return True


def assign_split(ds: xr.Dataset, splits, *, dim="sample") -> xr.DataArray:
    """Create a split-label coordinate from index dictionaries."""
    n = ds.sizes[dim]
    labels = np.full(n, "unused", dtype=object)
    for name, idx in splits.items():
        idx = np.asarray(idx, dtype=int)
        if np.any((idx < 0) | (idx >= n)):
            raise IndexError("split index outside dataset bounds")
        labels[idx] = name
    return xr.DataArray(labels, dims=(dim,), coords={dim: ds[dim]}, name="split")
