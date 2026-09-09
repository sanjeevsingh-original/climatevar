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

### Phase 2 scientific status
Trend methods include independent numerical reference checks. Sen's slope follows the pairwise-slope definition implemented by SciPy's `theilslopes`. Modified MK is explicitly a Yue-Wang-style effective-sample-size correction rather than a Hamed-Rao implementation. DFA is available as a persistence/scaling diagnostic and should not be interpreted alone as proof of long-range dependence.

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

### Phase 3 current focus
The precipitation indices use the ETCCDI wet-day convention (RR >= 1 mm; dry day RR < 1 mm), default R95p/R99p calibration to 1961–1990, explicit completeness controls, CF time bounds where available, and safeguards against irregular sub-daily aggregation. The external RClimDex harness remains non-blocking because the legacy PCICt/climdex.pcic dependency cannot currently be installed reliably on the modern CI environment.

## Phase 4 — Extremes and uncertainty
- [x] GPD/POT framework
- [x] Parametric bootstrap confidence intervals for GEV return levels
- [x] POT threshold diagnostics: exceedance rate, mean excess and parameter stability
- [x] Runs-based declustering for dependent exceedances
- [x] Bootstrap confidence intervals for POT return levels
- [x] POT threshold sensitivity workflow
- [x] Descriptive KS/Anderson-Darling and PIT/QQ diagnostics
- [x] Fitted-parameter parametric-bootstrap calibration for POT goodness-of-fit statistics
- [x] Empirical threshold-selection uncertainty envelope
- [x] Bootstrap propagation through an explicit AIC threshold-selection rule
- [x] First non-stationary GPD model with covariate-dependent scale
- [x] Stationary vs non-stationary GPD comparison with AIC/BIC and LR diagnostics
- [x] Parametric-bootstrap LR inference for non-stationarity
- [x] Bootstrap parameter intervals for non-stationary GPD
- [x] Publication-quality EVT plotting utilities
- [x] Automated HTML POT diagnostic reports
- [ ] Formal Bayesian/model-averaged threshold-selection uncertainty
- [ ] Robust non-stationary model comparison across multiple covariates and dependence structures
- [ ] External independent EVT reference validation

### Phase 4 current focus
POT now provides threshold exploration, GPD fitting, declustering, return levels, sampling uncertainty, calibrated bootstrap GOF diagnostics, threshold sensitivity, bootstrap propagation through an explicit threshold-selection rule, and conservative non-stationary inference. Advanced results remain conditional on the documented candidate thresholds, selection rule, covariates and stochastic assumptions.

## Phase 5 — Dataset interoperability
- [ ] Robust ERA5 / ERA5-Land adapters
- [ ] IMERG/GPM product adapters
- [ ] IMD/IMDAA station and gridded-data adapters
- [ ] CMIP6 calendar and ensemble utilities
- [ ] WRF precipitation and diagnostic adapters
- [ ] Metadata validation and provenance reporting

## Phase 6 — Model evaluation and rainfall error tagging
- [x] Bias, MAE, RMSE, correlation and NSE extensions
- [x] Event-based precipitation verification
- [x] Spatial and conditional error diagnostics
- [x] Error decomposition by intensity, season and synoptic-ready feature hooks
- [x] ML/DL-ready rainfall error-tagging datasets
- [x] Baseline Random Forest and histogram gradient-boosting classifiers
- [x] Optional XGBoost classifier interface
- [x] Class-imbalance-aware training and calibrated probabilities
- [x] Time-, group- and spatial-block-aware evaluation
- [x] Validation-set permutation importance
- [x] Optional TreeSHAP explanations
- [ ] Calibrated model-specific error-class thresholds and sensitivity analysis
- [ ] Deep CNN/ConvLSTM/Transformer rainfall-error classifier

### Phase 6 current focus
`climatevar.error_tagging` now separates post-hoc diagnostic error tagging from prospective error prediction, provides configurable rainfall-intensity and error classes, preserves traceable verification features, and provides chronological, event-group and spatial-block split utilities. Optional ML tooling provides reproducible Random Forest and histogram-gradient-boosting baselines, optional XGBoost, class-balanced training, probability calibration, leakage-aware cross-validation, permutation importance and TreeSHAP. Deep neural models remain deliberately deferred until the tabular benchmark and evaluation protocol are established.

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
