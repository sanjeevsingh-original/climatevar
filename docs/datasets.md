# Heterogeneous climate datasets

`climatevar` uses **xarray** as its internal data model, but real climate datasets do not use one universal naming convention. The IO normalization layer converts common coordinate and variable names to a small canonical vocabulary before analysis.

## Supported conventions

| Source | Typical coordinates | Common precipitation variable | Typical format |
|---|---|---|---|
| ERA5 | `latitude`, `longitude`, `time` | `tp` | NetCDF / GRIB |
| IMD | `lat`, `lon`, `time` (product dependent) | `rain`, `rf`, `precip` (product dependent) | NetCDF / text / binary |
| IMERG | `lat`, `lon`, `time` | `precipitation` or `precipitationCal` depending on product/version | NetCDF / HDF5 |
| WRF | `XLAT`, `XLONG`, `Times`/`Time` | `RAINNC` + `RAINC` | NetCDF |

The exact variable names can change between product versions, so **explicit mappings remain the preferred scientific practice**. IMERG V07 daily data, for example, documents `precipitation` as the daily precipitation-rate variable and `lat`/`lon` as coordinates. citeturn0search3 WRF output commonly uses `XLAT` and `XLONG` for mass-grid latitude/longitude and may contain staggered-grid coordinate variables such as `XLAT_U`/`XLONG_U`. citeturn0search2 ERA5 GRIB opened through `cfgrib` exposes `latitude` and `longitude` directly to xarray. citeturn0search0

## Canonicalize a dataset

```python
import xarray as xr
from climatevar.io import normalize_dataset

# ERA5 NetCDF/GRIB already using standard coordinate names
raw = xr.open_dataset("era5.nc")
ds = normalize_dataset(raw, dataset="era5")

# The analysis layer can now use stable names.
precip = ds["precipitation"]
lat = ds["latitude"]
lon = ds["longitude"]
```

## IMERG

```python
raw = xr.open_dataset("IMERG_daily.nc")
ds = normalize_dataset(raw, dataset="imerg")
precip = ds["precipitation"]
```

## WRF

WRF is more complicated because its latitude/longitude coordinates can be 2-D and its rainfall is often accumulated in separate fields. The normalization layer handles the coordinate naming, but it **does not silently add `RAINC` and `RAINNC`** because doing so without checking the model output convention could change the scientific meaning.

```python
raw = xr.open_dataset("wrfout_d01_2020-06-01_00:00:00")
ds = normalize_dataset(
    raw,
    dataset="wrf",
    variables={"precipitation": "RAINNC"},
)
```

If total accumulated convective + non-convective rainfall is intended, construct it explicitly:

```python
total_rain = raw["RAINC"] + raw["RAINNC"]
```

## Explicit mappings for IMD or custom products

When a product uses an unusual name, pass the mapping instead of relying on aliases:

```python
ds = normalize_dataset(
    raw,
    variables={"precipitation": "rf"},
    strict=True,
)
```

The normalization history is stored in `ds.attrs["climatevar:normalization"]`, so transformations remain auditable.

## Important scientific limitation

Normalization solves **naming and structural interoperability**. It does not automatically solve differences in units, temporal accumulation/averaging, calendars, grids, missing-value conventions, or spatial resolution. Those require dataset-aware transformations and validation. This distinction is deliberate: a library for research should not silently convert a rate to an accumulation or combine incompatible rainfall definitions.
