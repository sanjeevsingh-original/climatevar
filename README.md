# climatevar

A research-oriented Python library for reproducible climate and atmospheric science analysis.

## Why climatevar?

`climatevar` is designed around a simple principle: **scientific calculations should be explicit, testable, reproducible, and easy to audit.** The library uses `xarray` as its core data model so that dimensions, coordinates, attributes, and multidimensional climate fields remain visible throughout an analysis.

## Current capabilities

### Climate variability
- Grouped monthly and day-of-year climatologies
- Anomaly calculations
- Data completeness and validity diagnostics

### Precipitation extremes
- Rx1day — annual maximum 1-day precipitation
- Rx5day — annual maximum consecutive 5-day precipitation
- PRCPTOT — wet-day precipitation total
- R10mm / R20mm — heavy precipitation day counts
- R95p / R99p — precipitation above percentile thresholds
- CWD / CDD — consecutive wet/dry day diagnostics
- **All precipitation amount indices explicitly use mm**
- Daily precipitation indices reject rates/fluxes unless they have first been converted to daily amounts

### Trend analysis
- Mann-Kendall statistic and Kendall tau
- Sen's slope estimator
- Explicit valid-sample counts

### Extreme-value statistics
- Generalized Extreme Value (GEV) maximum-likelihood fitting
- GEV return-level calculation
- Explicit SciPy shape-parameter convention

### Atmospheric thermodynamics
- Potential temperature
- Bolton-style equivalent potential temperature
- Saturation vapor pressure
- Specific humidity from relative humidity

### Spatial analysis
- Cosine-latitude weights
- Area-weighted means for regular latitude/longitude grids

### Dataset interoperability
- Canonical `latitude`, `longitude`, and `time` coordinate names
- Alias handling for common ERA5, IMD, IMDAA, IMERG, and WRF conventions
- Canonical variable names such as `precipitation`, `temperature`, `surface_pressure`, `relative_humidity`, `u_wind`, and `v_wind`
- Explicit user mappings for unusual or ambiguous products
- WRF `Times` decoding foundation
- Optional NetCDF/HDF5/GRIB dependencies
- Normalization history stored in dataset metadata for auditability
- Precipitation amounts normalized to **mm**; precipitation fluxes remain explicit fluxes

### Data and modelling utilities
- Explicit xarray NetCDF loading helper
- WRF nested-domain grid-spacing calculations
- Lightweight xarray plotting helpers

## Design principles

1. **xarray-first:** preserve dimensions and coordinates.
2. **No silent transformations:** units, calendars, masking, temporal accumulation, and regridding should be explicit.
3. **Research transparency:** document statistical assumptions and conventions.
4. **Test-driven development:** scientific functions should have regression tests.
5. **Optional dependencies:** statistical, plotting, and specialized IO features remain separated from the core installation.
6. **Reproducibility:** deterministic calculations and documented workflows are preferred over opaque convenience functions.

## Installation for development

```bash
git clone https://github.com/sanjeevsingh-original/climatevar.git
cd climatevar
python -m pip install -e ".[dev]"
pytest
```

For NetCDF/HDF5/GRIB interoperability:

```bash
python -m pip install -e ".[io]"
```

## Example: normalize heterogeneous datasets

```python
import xarray as xr
from climatevar.io import normalize_dataset

# ERA5: latitude/longitude/tp/t2m/sp conventions
era5 = normalize_dataset(xr.open_dataset("era5.nc"), dataset="era5")

# IMERG: lat/lon/precipitation conventions
imerg = normalize_dataset(xr.open_dataset("IMERG_daily.nc"), dataset="imerg")

# WRF: XLAT/XLONG and an explicitly selected precipitation field
wrf = normalize_dataset(
    xr.open_dataset("wrfout_d01_2020-06-01_00:00:00"),
    dataset="wrf",
    variables={"precipitation": "RAINNC"},
)

# The analysis layer can now use stable names.
print(era5["latitude"], era5["longitude"], era5["precipitation"])
```

`normalize_dataset(..., si=True)` is the default. Recognized precipitation **amounts are normalized to mm**, while precipitation rates/fluxes remain explicit `kg m-2 s-1` quantities. Rates are never silently treated as daily rainfall amounts.

See [`docs/datasets.md`](docs/datasets.md) and [`docs/interoperability.md`](docs/interoperability.md) for dataset-specific guidance.

## Example: precipitation indices

```python
from climatevar.precipitation import rx1day, rx5day

rain = era5["precipitation"]  # precipitation amount in mm
annual_rx1day = rx1day(rain)  # mm
annual_rx5day = rx5day(rain)  # mm
```

Precipitation indices require explicit amount units. Inputs in `m`, `cm`, or `mm` are normalized to `mm`; inputs without units or with rate/flux units are rejected to prevent physically incorrect rainfall indices.

## Scientific scope

The first release is intentionally conservative. The library is not intended to replace domain-specific packages such as `xarray`, `scipy`, `wrf-python`, or specialized extreme-value software. Instead, it provides tested, composable analysis primitives for climate research workflows.

ETCCDI-inspired precipitation indices should be interpreted with their data-frequency, missing-data, threshold, calendar, and baseline-period assumptions in mind. For publication-grade analyses, users should verify the exact index definition required by their study and document all methodological choices.

## Development status

🚧 **Pre-alpha — active development.** APIs may change before the first stable release.

## Roadmap

- [x] Package structure and build system
- [x] Climatology and anomaly core
- [x] Precipitation extreme indices
- [x] Trend diagnostics
- [x] GEV tools
- [x] Thermodynamic diagnostics
- [x] Spatial weighting
- [x] WRF utility foundation
- [x] Automated test workflow
- [x] Dataset normalization foundation for ERA5 / IMD / IMDAA / IMERG / WRF
- [ ] Formal calendar-aware climate indices
- [ ] Dask-scale benchmarking and chunk-aware algorithms
- [ ] GPD/POT extremes and bootstrap uncertainty
- [ ] Autocorrelation-aware trend significance
- [ ] Formal SPI/SPEI implementation
- [ ] ERA5 / IMERG / IMDAA reader adapters and metadata validation
- [ ] WRF diagnostics and domain utilities
- [ ] Bias/error diagnostics and ML/DL-ready rainfall error tagging
- [ ] Publication-quality visualization and worked scientific examples
- [ ] API stability review
- [ ] PyPI release and DOI/research citation metadata

## License

MIT License. See `LICENSE`.
