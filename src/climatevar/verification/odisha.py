"""Ready-to-use Odisha precipitation comparison configuration.

The paths are intentionally user-supplied: the scientific protocol is fixed,
while local storage locations and provider-specific variable names remain
explicit and auditable.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .datasets import ProductSpec
from .experiment import ExperimentConfig

# Approximate geographic envelope of Odisha. Users can replace this with a
# watershed/station-derived mask for publication analyses.
ODISHA_BBOX = (17.5, 22.75, 81.25, 87.75)  # south, north, west, east


def odisha_experiment(
    *,
    imd,
    era5,
    imerg,
    imdaa,
    wrf,
    cmip6,
    start="2000-06-01",
    end="2018-12-31",
    imd_variable="rainfall",
    era5_variable="tp",
    imerg_variable="precipitationCal",
    imdaa_variable="pr",
    cmip6_variable="pr",
    wrf_variable=None,
    regrid_method="conservative",
):
    """Build the standard IMD-vs-five-product Odisha experiment.

    ``2000-2018`` is the default because IMDAA is available for 1979-2018;
    users should change the period when their IMDAA archive or other products
    cover a different common period.
    """
    return ExperimentConfig(
        reference=ProductSpec("IMD", "imd", imd, variable=imd_variable),
        products=(
            ProductSpec("ERA5", "era5", era5, variable=era5_variable),
            ProductSpec("IMERG", "imerg", imerg, variable=imerg_variable),
            ProductSpec("IMDAA", "imdaa", imdaa, variable=imdaa_variable),
            ProductSpec("WRF", "wrf", wrf, variable=wrf_variable),
            ProductSpec("CMIP6", "cmip6", cmip6, variable=cmip6_variable),
        ),
        start=start,
        end=end,
        threshold=1.0,
        seasons={"MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9), "ON": (10, 11)},
        common_grid_method=None,
        regrid_method=regrid_method,
        region="odisha",
    )


def subset_odisha(data, bbox=ODISHA_BBOX):
    """Subset a standardized ``lat/lon`` field to the Odisha envelope."""
    south, north, west, east = bbox
    lat_slice = slice(south, north) if float(data.lat[0]) < float(data.lat[-1]) else slice(north, south)
    lon_slice = slice(west, east) if float(data.lon[0]) < float(data.lon[-1]) else slice(east, west)
    return data.sel(lat=lat_slice, lon=lon_slice)


def validate_input_paths(**paths):
    """Return missing local file paths before a long experiment is started."""
    missing = {name: str(path) for name, path in paths.items() if isinstance(path, (str, Path)) and not Path(path).exists()}
    if missing:
        raise FileNotFoundError("Missing experiment inputs: " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    return True
