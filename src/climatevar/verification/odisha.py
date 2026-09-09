"""Ready-to-use Odisha precipitation comparison configuration."""
from __future__ import annotations

from pathlib import Path

from .datasets import ProductSpec
from .experiment import ExperimentConfig

# Approximate geographic envelope; replace with an authoritative polygon mask
# when publication results require administrative-boundary exactness.
ODISHA_BBOX = (17.7, 22.6, 81.3, 87.5)  # south, north, west, east


def odisha_experiment(*, imd, era5, imerg, imdaa, wrf, cmip6,
                      start="2000-06-01", end="2018-12-31",
                      imd_variable="rainfall", era5_variable="tp",
                      imerg_variable="precipitationCal", imdaa_variable="pr",
                      cmip6_variable="pr", wrf_variable=None,
                      regrid_method="conservative", uncertainty_resamples=1000,
                      block_length=7, random_state=0):
    """Build the standard IMD-vs-five-product Odisha experiment.

    The default period ends in 2018 because the IMDAA archive spans 1979-2018.
    Paths, variable names and period remain explicit so the provenance of a
    real analysis is auditable.
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
        start=start, end=end, threshold=1.0,
        seasons={"MAM": (3, 4, 5), "JJAS": (6, 7, 8, 9), "ON": (10, 11)},
        regrid_method=regrid_method, region="odisha",
        uncertainty_resamples=uncertainty_resamples,
        block_length=block_length, random_state=random_state,
    )


def subset_odisha(data, bbox=ODISHA_BBOX):
    """Subset a standardized 1-D ``lat/lon`` field to the Odisha envelope."""
    south, north, west, east = bbox
    lat_slice = slice(south, north) if float(data.lat[0]) < float(data.lat[-1]) else slice(north, south)
    lon_slice = slice(west, east) if float(data.lon[0]) < float(data.lon[-1]) else slice(east, west)
    return data.sel(lat=lat_slice, lon=lon_slice)


def validate_input_paths(**paths):
    """Fail early when local experiment inputs are missing."""
    missing = {name: str(path) for name, path in paths.items() if isinstance(path, (str, Path)) and not Path(path).exists()}
    if missing:
        raise FileNotFoundError("Missing experiment inputs: " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    return True
