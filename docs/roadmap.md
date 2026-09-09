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
- [x] Independent numerical reference checks for trend methods
- [x] Moving-block bootstrap confidence intervals for Sen slopes and regional trends
- [x] DFA-based persistence/long-memory diagnostic

## Phase 3 — Publication-grade climate indices
- [x] Calendar-aware ETCCDI-style precipitation definitions for core rainfall indices
- [x] Wet-day threshold conventions and explicit annual completeness controls
- [x] Fixed percentile baseline support with 1961–1990 as the ETCCDI default
- [x] CF time bounds for precipitation accumulation where available
- [x] Sub-daily-to-daily regular-sampling and coverage safeguards
- [x] Independent regression fixtures for core ETCCDI precipitation definitions
- [x] Leap-year, no-leap and 360-day calendar validation
- [x] Missing-day sensitivity and explicit completeness-threshold tests
- [x] Cross-check against published RClimDex/ETCCDI definitions and threshold conventions
- [ ] Validate against external RClimDex/Climpact numerical output files
- [ ] Add uncertainty and sensitivity options for percentile thresholds

## Phase 4 — Extremes and uncertainty
- [x] GPD/POT framework
- [x] Parametric bootstrap confidence intervals for GEV return levels
- [x] POT threshold diagnostics and sensitivity
- [x] Runs-based declustering
- [x] Bootstrap confidence intervals and bootstrap GOF calibration
- [x] Non-stationary GPD and model-comparison diagnostics
- [x] Publication-quality EVT plotting and diagnostic reports
- [ ] Formal Bayesian/model-averaged threshold-selection uncertainty
- [ ] Robust non-stationary model comparison across multiple covariates and dependence structures
- [ ] External independent EVT reference validation

## Phase 5 — Dataset interoperability
- [x] Robust ERA5 / ERA5-Land adapters
- [x] IMERG/GPM product adapters
- [x] IMD/IMDAA gridded-data adapters
- [x] CMIP6 precipitation variable and calendar-compatible loading foundation
- [x] WRF precipitation adapter using cumulative-component differencing and restart handling
- [x] Metadata validation and explicit variable override/provenance fields
- [ ] Station-level IMD ingestion and station-to-grid matchup utilities
- [ ] Full CF cell-bound/bounds validation for conservative regridding

## Phase 6 — Model evaluation and rainfall error tagging
- [x] Bias, MAE, RMSE, correlation and NSE extensions
- [x] Event-based precipitation verification
- [x] Spatial and conditional error diagnostics
- [x] Error decomposition by intensity, season and synoptic-ready feature hooks
- [x] ML/DL-ready rainfall error-tagging datasets
- [x] Random Forest / histogram-gradient-boosting baselines and optional XGBoost
- [x] Class-imbalance-aware training, probability calibration and leakage-aware CV
- [x] Comparative ERA5/IMERG/IMDAA/WRF/CMIP6 scorecard and transparent ranking
- [x] Automated multi-product experiment runner with seasonal/intensity scorecards, spatial NetCDF fields and ML-ready error outputs
- [ ] Calibrated model-specific error-class thresholds and sensitivity analysis
- [ ] Deep CNN/ConvLSTM/Transformer rainfall-error classifier

### Phase 6 current focus
`climatevar.verification` now provides dataset-family adapters, explicit reference/product specifications, common-period alignment, optional linear/nearest or xESMF regridding, deterministic and categorical scorecards, seasonal and rainfall-intensity conditioning, India/Odisha regional subsetting, spatial metric fields, transparent weighted ranking and ML/error-tagging outputs. The remaining research step is independent numerical validation on real ERA5/IMERG/IMDAA/WRF/CMIP6 files and calibrated uncertainty/sensitivity analysis.

## Phase 7 — Research-scale computation
- [ ] Dask-aware trend and bootstrap algorithms
- [ ] Chunk-aware spatial statistics
- [ ] Performance benchmarks for regional and continental grids
- [ ] Memory-safe large ensemble workflows
- [ ] Optional accelerated numerical backends where justified

## Phase 8 — Scientific usability
- [ ] Reproducible worked examples for Indian Summer Monsoon research
- [ ] Example workflows for Odisha precipitation extremes
- [ ] Notebook suite covering ERA5, IMERG, IMDAA, WRF and CMIP6
- [x] Validation fixtures and expected-value regression datasets for core precipitation indices
- [x] EVT diagnostic and research-method reporting helpers

## Phase 9 — Stable release
- [ ] API stability review
- [ ] Deprecation and compatibility policy
- [ ] Versioned documentation
- [ ] Comprehensive API reference
- [ ] PyPI release
- [ ] DOI and citation metadata
- [ ] Reproducible release benchmarks

## Scientific quality gate
Before a major method is promoted to a stable API, `climatevar` should have: (1) an explicit mathematical/statistical definition; (2) documented assumptions and limitations; (3) synthetic-data tests; (4) independent reference comparison where available; (5) clear missing-data and calendar treatment; (6) reproducible stochastic procedures; (7) appropriate uncertainty/significance reporting; and (8) documentation showing how the method should be reported scientifically.
