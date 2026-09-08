# climatevar

A research-oriented Python library for reproducible climate and atmospheric science analysis.

## Why climatevar?

`climatevar` is designed around a simple principle: **scientific calculations should be explicit, testable, reproducible, and easy to audit.** The library uses `xarray` as its core data model so that dimensions, coordinates, attributes, and multidimensional climate fields remain visible throughout an analysis.

## Current capabilities

### Climate variability
- Grouped monthly and day-of-year climatologies
- Anomaly calculations
- Data completeness and validity diagnostics
- Calendar-aware time and seasonal handling

### Precipitation extremes
- Rx1day, Rx5day, PRCPTOT, R10mm, R20mm
- R95p / R99p percentile-based indices
- CWD / CDD consecutive wet/dry diagnostics
- Precipitation amount indices explicitly use mm
- Daily precipitation indices reject rates/fluxes unless converted to amounts

### Trend analysis
- Classical Mann-Kendall and Kendall tau
- Sen's slope
- Yue-Wang-style effective-sample-size modified MK
- Trend-free pre-whitening
- Calendar-aware seasonal aggregation for monthly, daily and regular sub-daily data
- Spatial trend fields with Benjamini-Hochberg FDR
- Area-weighted regional trend summaries
- Moving-block bootstrap field significance that preserves spatial covariance through common temporal resampling

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
- Canonical atmospheric variable names
- Explicit user mappings for unusual or ambiguous products
- WRF `Times` decoding foundation
- Optional NetCDF/HDF5/GRIB dependencies
- Normalization history stored in dataset metadata for auditability

### Data and modelling utilities
- Explicit xarray NetCDF loading helper
- WRF nested-domain grid-spacing calculations
- Lightweight xarray plotting helpers

## Design principles

1. **xarray-first:** preserve dimensions and coordinates.
2. **No silent transformations:** units, calendars, masking, temporal accumulation, and regridding should be explicit.
3. **Research transparency:** document statistical assumptions and conventions.
4. **Test-driven development:** scientific functions should have regression tests.
5. **Optional dependencies:** specialized IO and statistical features remain separated from the core installation where practical.
6. **Reproducibility:** deterministic calculations and documented workflows are preferred over opaque convenience functions.

## Installation for development

```bash
git clone https://github.com/sanjeevsingh-original/climatevar.git
cd climatevar
python -m pip install -e ".[dev]"
pytest
```

## Scientific scope

`climatevar` is not intended to replace domain-specific packages such as `xarray`, `scipy`, `wrf-python`, or specialized extreme-value software. It provides tested, composable analysis primitives for climate research workflows.

For publication-grade analyses, users should verify the exact index definition required by their study and document dataset/version, temporal aggregation, units, calendar, completeness rule, baseline period, spatial weighting, statistical method, autocorrelation treatment, significance level, multiple-testing correction, and uncertainty/resampling design.

## Development status

🚧 **Pre-alpha — active development.** APIs may change before the first stable release.

## Roadmap

### Completed foundations
- [x] Package structure and build system
- [x] Climatology and anomaly core
- [x] Precipitation extreme indices
- [x] GEV tools
- [x] Thermodynamic diagnostics
- [x] Spatial weighting
- [x] WRF utility foundation
- [x] Automated test workflow
- [x] Dataset normalization foundation for ERA5 / IMD / IMDAA / IMERG / WRF
- [x] Classical, modified and pre-whitened trend diagnostics
- [x] Calendar-aware seasonal trend aggregation
- [x] Spatial FDR, regional trends and block-bootstrap field significance
- [x] Formal SPI/SPEI foundation

### Near-term scientific priorities
- [ ] Complete formal calendar-aware ETCCDI-style index definitions and edge-case validation
- [ ] Strengthen precipitation time-bound handling and accumulation semantics
- [ ] Improve missing-data and calendar completeness diagnostics across all index families
- [ ] Validate trend methods against published reference implementations and synthetic persistence tests
- [ ] Add uncertainty intervals for Sen slopes and regional trends
- [ ] Add GPD/POT extremes with bootstrap uncertainty and diagnostics

### Dataset and modelling priorities
- [ ] Robust ERA5 / ERA5-Land / IMERG / IMDAA reader adapters and metadata validation
- [ ] WRF precipitation and diagnostic adapters, including nested-domain workflows
- [ ] CMIP6 preprocessing, calendar handling and ensemble utilities
- [ ] Bias/error diagnostics and ML/DL-ready rainfall error tagging
- [ ] Dask-scale benchmarking and chunk-aware algorithms

### Reproducibility and usability
- [ ] Publication-quality visualization and worked scientific examples
- [ ] Example notebooks for Indian Summer Monsoon and precipitation extremes
- [ ] Validation datasets and expected-value regression fixtures
- [ ] API stability review and deprecation policy
- [ ] PyPI release, versioned documentation and DOI/research citation metadata
