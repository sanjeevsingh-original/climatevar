"""Core climatology utilities.

The functions in this module are deliberately small and transparent. They
operate on xarray objects and preserve coordinate metadata where possible.
"""

from __future__ import annotations

import xarray as xr


def climatology(data: xr.DataArray, dim: str = "time", group: str = "month") -> xr.DataArray:
    """Calculate a grouped climatological mean.

    Parameters
    ----------
    data:
        Input climate variable as an :class:`xarray.DataArray`.
    dim:
        Dimension containing the time coordinate.
    group:
        Grouping frequency. Currently ``"month"`` and ``"dayofyear"`` are
        supported.

    Returns
    -------
    xarray.DataArray
        Grouped climatological mean.

    Raises
    ------
    ValueError
        If the requested dimension or grouping is unavailable.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data.")
    if dim not in data.coords:
        raise ValueError(f"Dimension {dim!r} must have a coordinate for climatology.")

    if group == "month":
        return data.groupby(f"{dim}.month").mean(dim=dim, skipna=True)
    if group == "dayofyear":
        return data.groupby(f"{dim}.dayofyear").mean(dim=dim, skipna=True)

    raise ValueError("group must be either 'month' or 'dayofyear'.")


def anomaly(data: xr.DataArray, reference: xr.DataArray, group: str = "month") -> xr.DataArray:
    """Calculate anomalies relative to a climatological reference.

    ``reference`` should normally be produced by :func:`climatology`.
    """
    if group == "month":
        return data.groupby("time.month") - reference
    if group == "dayofyear":
        return data.groupby("time.dayofyear") - reference
    raise ValueError("group must be either 'month' or 'dayofyear'.")
