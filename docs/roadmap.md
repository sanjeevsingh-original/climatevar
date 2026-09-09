# climatevar scientific roadmap

This roadmap prioritizes scientific correctness, reproducibility and research usefulness before API expansion.

## Phase 1 — Core research foundations

- [x] xarray-first data model and package structure
- [x] Climatology, anomalies and completeness diagnostics
- [x] Precipitation indices and explicit unit handling
- [x] GEV fitting and return levels
- [x] Thermodynamic diagnostics
- [x] Spatial weighting
- [x] Dataset normalization foundations
- [x] WRF utility foundation
- [x] Automated tests and documentation workflow

## Phase 2 — Robust trend and drought analysis

- [x] Classical Mann-Kendall and Sen slope
- [x] Autocorrelation-aware modified Mann-Kendall
- [x] Trend-free pre-whitening
- [x] Calendar-aware seasonal aggregation for monthly, daily and regular sub-daily data
- [x] Spatial trend fields and Benjamini-Hochberg FDR
- [x] Area-weighted regional trends
- [x] Moving-block bootstrap field significance
- [x] Formal SPI/SPEI foundation
- [x] Benchmark classical MK, Sen slope and modified MK against independent reference calculations
- [x] Add moving-block bootstrap confidence intervals for Sen slopes and regional trends
- [x] Add DFA-based persistence/long-memory diagnostics

### Phase 2 scientific status
Trend methods now include independent numerical reference checks. Sen's slope uses the pairwise-slope definition implemented by SciPy's `theilslopes`. Modified MK remains explicitly documented as a Yue-Wang-style effective-sample-size correction rather than a Hamed-Rao implementation. DFA is a persistence/scaling diagnostic and is not treated as a standalone hypothesis test for long-range dependence.

## Phase 3 — Publication-grade climate indices

- [x] Calendar-aware ETCCDI-style precipitation definitions for core rainfall indices
- [x] Wet-day threshold conventions and explicit annual completeness controls
- [x] Fixed percentile baseline support with 1961–1990 as the ETCCDI default
- [x] CF time bounds for precipitation accumulation where available
- [x] Sub-daily-to-daily regular-sampling and coverage safeguards
- [ ] Validate wet-day, percentile-baseline and completeness conventions against independent reference datasets
- [ ] Add uncertainty and sensitivity options for percentile thresholds

### Phase 3 current focus
The precipitation indices now use the ETCCDI wet-day convention (RR >= 1 mm; dry day RR < 1 mm), default R95p/R99p percentile calibration to 1961–1990, and expose `min_valid_fraction` for annual completeness plus `min_daily_valid_fraction` for sub-daily coverage. Irregular sub-daily data are rejected for ETCCDI indices rather than silently producing potentially biased daily totals. Explicit reference thresholds remain available for reproducible experiments and non-standard baselines.

## Phase 4 — Extremes and uncertainty

- [ ] GPD/POT framework
- [ ] Bootstrap confidence intervals for extreme-value parameters and return levels
- [ ] Non-stationary extreme-value models
- [ ] Diagnostic plots and goodness-of-fit tests
- [ ] Threshold-selection diagnostics

## Phase 5 — Dataset interoperability

- [ ] Robust ERA5 / ERA5-Land adapters
- [ ] IMERG/GPM product adapters
- [ ] IMD/IMDAA station and gridded-data adapters
- [ ] CMIP6 calendar and ensemble utilities
- [ ] WRF precipitation and diagnostic adapters
- [ ] Metadata validation and provenance reporting

## Phase 6 — Model evaluation and rainfall error tagging

- [ ] Bias, MAE, RMSE, correlation and NSE extensions
- [ ] Event-based precipitation verification
- [ ] Spatial and conditional error diagnostics
- [ ] Error decomposition by intensity, season and synoptic regime
- [ ] ML/DL-ready rainfall error-tagging datasets
- [ ] Explainable ML diagnostics for model rainfall errors

## Phase 7 — Research-scale computation

- [ ] Dask-aware trend and bootstrap algorithms
- [ ] Chunk-aware spatial statistics
- [ ] Performance benchmarks for regional and continental grids
- [ ] Memory-safe large ensemble workflows
- [ ] Optional accelerated numerical backends where justified

## Phase 8 — Scientific usability

- [ ] Publication-quality plotting utilities
- [ ] Reproducible worked examples for Indian Summer Monsoon research
- [ ] Example workflows for Odisha precipitation extremes
- [ ] Notebook suite covering ERA5, IMERG, IMDAA, WRF and CMIP6
- [ ] Validation fixtures and expected-value regression datasets
- [ ] Research-method reporting helpers

## Phase 9 — Stable release

- [ ] API stability review
- [ ] Deprecation and compatibility policy
- [ ] Versioned documentation
- [ ] Comprehensive API reference
- [ ] PyPI release
- [ ] DOI and citation metadata
- [ ] Reproducible release benchmarks

## Scientific quality gate

A feature should not be considered publication-ready merely because its numerical output looks plausible. Before a major method is promoted to a stable API, `climatevar` should have:

1. an explicit mathematical/statistical definition;
2. documented assumptions and limitations;
3. synthetic-data tests with known behavior;
4. comparison against an independent reference where available;
5. clear treatment of missing data and calendars;
6. reproducible random seeds for stochastic procedures;
7. appropriate uncertainty or significance reporting; and
8. documentation showing how the method should be reported in a scientific paper.
