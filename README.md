# climatevar

A research-oriented Python library for reproducible climate and atmospheric science analysis.

## Why climatevar?

`climatevar` is designed around a simple principle: **scientific calculations should be explicit, testable, reproducible, and easy to audit.** The library uses `xarray` as its core data model so that dimensions, coordinates, attributes, and multidimensional climate fields remain visible throughout an analysis.

## Current capabilities

### Climate variability
- Grouped monthly and day-of-year climatologies
- Anomaly calculations

### Precipitation extremes
- Rx1day — annual maximum 1-day precipitation
- Rx5day — annual maximum consecutive 5-day precipitation
- PRCPTOT — wet-day precipitation total
- R10mm / R20mm — heavy precipitation day counts
- R95p / R99p — precipitation above percentile thresholds
- CWD / CDD — consecutive wet/dry day diagnostics

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

### Data and modelling utilities
- Explicit xarray NetCDF loading helper
- WRF nested-domain grid-spacing calculations
- Lightweight xarray plotting helpers

## Design principles

1. **xarray-first:** preserve dimensions and coordinates.
2. **No silent transformations:** units, calendars, masking, and regridding should be explicit.
3. **Research transparency:** document statistical assumptions and conventions.
4. **Test-driven development:** scientific functions should have regression tests.
5. **Optional dependencies:** statistical and plotting features remain separated from the core installation.
6. **Reproducibility:** deterministic calculations and documented workflows are preferred over opaque convenience functions.

## Installation for development

```bash
git clone https://github.com/sanjeevsingh-original/climatevar.git
cd climatevar
python -m pip install -e ".[dev]"
pytest
```

## Example

```python
import xarray as xr
from climatevar.precipitation import rx1day, rx5day

rain = xr.open_dataset("daily_precipitation.nc")["pr"]

annual_rx1day = rx1day(rain)
annual_rx5day = rx5day(rain)
```

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
- [ ] Robust calendar-aware climate indices
- [ ] Dask-scale benchmarking and chunk-aware algorithms
- [ ] ERA5 / IMERG / GRIB adapters
- [ ] WRF diagnostics and domain utilities
- [ ] Bias/error diagnostics for model precipitation
- [ ] ML/DL-ready feature engineering
- [ ] Documentation site and worked scientific examples
- [ ] API stability review
- [ ] PyPI release and DOI/research citation metadata

## License

MIT License. See `LICENSE`.
