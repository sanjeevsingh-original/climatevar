"""Dataset loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import xarray as xr


def open_dataset(path: str | Path, **kwargs: Any) -> xr.Dataset:
    """Open a NetCDF dataset with xarray.

    The helper intentionally adds no silent unit conversion or coordinate
    renaming. Scientific workflows should make those transformations explicit.
    Extra keyword arguments are passed directly to ``xarray.open_dataset``.
    """
    return xr.open_dataset(path, **kwargs)
