"""WRF precipitation utilities.

WRF commonly stores RAINC and RAINNC as cumulative precipitation fields.
These helpers combine the configured precipitation components and convert
cumulative fields to interval amounts without requiring users to perform the
bookkeeping themselves.
"""
from __future__ import annotations

import xarray as xr


def wrf_total_precipitation(
    ds: xr.Dataset,
    *,
    include_shallow: bool = False,
    convective: str = "RAINC",
    nonconvective: str = "RAINNC",
    shallow: str = "RAINSH",
) -> xr.DataArray:
    """Return total accumulated WRF precipitation.

    By default total precipitation is ``RAINC + RAINNC``. ``RAINSH`` is
    included only when ``include_shallow=True`` because its interpretation
    depends on the WRF physics configuration.
    """
    missing = [name for name in (convective, nonconvective) if name not in ds]
    if missing:
        raise KeyError(f"Required WRF precipitation variable(s) missing: {', '.join(missing)}")
    total = ds[convective] + ds[nonconvective]
    if include_shallow:
        if shallow not in ds:
            raise KeyError(f"Requested shallow-convection variable {shallow!r} is missing.")
        total = total + ds[shallow]
    total = total.copy()
    total.name = "precipitation_accumulated"
    total.attrs = dict(total.attrs)
    total.attrs["climatevar:precipitation_kind"] = "accumulated"
    total.attrs["climatevar:source_components"] = ",".join(
        [convective, nonconvective] + ([shallow] if include_shallow else [])
    )
    return total


def wrf_precipitation_amount(
    ds: xr.Dataset,
    *,
    include_shallow: bool = False,
    convective: str = "RAINC",
    nonconvective: str = "RAINNC",
    shallow: str = "RAINSH",
    dim: str = "Time",
) -> xr.DataArray:
    """Return WRF precipitation increments in mm.

    Negative differences are treated as accumulation resets/restarts and the
    current cumulative value is retained for that interval.
    """
    accumulated = wrf_total_precipitation(
        ds,
        include_shallow=include_shallow,
        convective=convective,
        nonconvective=nonconvective,
        shallow=shallow,
    )
    if dim not in accumulated.dims:
        if "time" in accumulated.dims:
            dim = "time"
        else:
            raise ValueError(f"Time dimension {dim!r} was not found in WRF precipitation data.")
    diff = accumulated.diff(dim, label="upper")
    first = accumulated.isel({dim: 0})
    first = first.expand_dims({dim: [accumulated[dim].values[0]]})
    first = first.assign_coords({dim: [accumulated[dim].values[0]]})
    increments = xr.concat([first, diff], dim=dim)
    increments = increments.where(increments >= 0, accumulated)
    increments.name = "precipitation"
    increments.attrs = dict(accumulated.attrs)
    increments.attrs["units"] = "mm"
    increments.attrs["climatevar:precipitation_kind"] = "interval_amount"
    increments.attrs["climatevar:source_accumulation"] = "WRF cumulative precipitation"
    return increments
