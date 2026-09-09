# Rainfall error tagging and ML-ready datasets

`climatevar.error_tagging` provides a reproducible foundation for diagnosing and modelling rainfall-model errors across ERA5, IMERG/GPM, IMDAA/IMD, WRF and CMIP6 workflows.

## Two scientifically different tasks

### 1. Diagnostic error tagging

Use observations and model output together to label the realised error:

```python
from climatevar.error_tagging import error_class

labels = error_class(observation, prediction)
```

The default classes are `underestimate_severe`, `underestimate_moderate`, `near_zero`, `overestimate_moderate`, and `overestimate_severe`. Thresholds are configurable. This is appropriate for post-hoc attribution and model evaluation.

### 2. Prospective error prediction

If the goal is to predict whether a model will fail before the verifying observation is available, **do not use observation-derived features as predictors**. In that case, the observed error class is the target and predictors must come from model-side meteorology, static geography, and information available at forecast/analysis time.

## Feature generation

```python
from climatevar.error_tagging import build_error_features, to_ml_table

features = build_error_features(observation, prediction, elevation=elevation)
table = to_ml_table(features)
```

The feature dataset preserves observation, prediction, signed error, absolute error, relative error, log-ratio, rainfall intensity class, wet/dry status, calendar variables and spatial coordinates. The target `error_class` is included by default.

For prospective ML, use `include_target=False` and explicitly construct the predictor list without `observation`, `error`, `absolute_error`, `relative_error`, `log_ratio`, or any other truth-derived quantity.

## Leakage-safe splitting

Rainfall observations are temporally correlated, and nearby grid cells are spatially dependent. Random row-wise splitting can therefore give overly optimistic generalisation estimates. The package provides:

```python
from climatevar.error_tagging import temporal_split, event_group_split, spatial_block_split

# chronological generalisation
splits = temporal_split(time, train_fraction=0.70, validation_fraction=0.15)

# keep every observation from an event in one partition
event_splits = event_group_split(event_id)

# test spatial transfer to held-out blocks
space_splits = spatial_block_split(lat, lon, lat_block=2.0, lon_block=2.0)
```

For time-ordered data, chronological validation is preferable to random shuffling because nearby observations can be correlated in time. Scikit-learn likewise documents `TimeSeriesSplit` for time series and group-based splitters for keeping related samples together. citeturn0search0turn0search3

## Recommended Odisha/Indian-monsoon experiment

1. Harmonise the observation and model datasets to the same grid, accumulation period and units.
2. Generate continuous verification metrics before classification.
3. Define error classes using thresholds fixed **before** inspecting test performance.
4. Add physically meaningful predictors: rainfall intensity, season, latitude/longitude, elevation, distance from coast, orography and model-side thermodynamic/dynamical fields.
5. Split chronologically by year or monsoon season; use event grouping for individual rainfall events.
6. For spatial-transfer experiments, hold out complete geographic blocks rather than random grid cells.
7. Fit imputation, scaling, feature selection and class-balancing procedures using training data only.
8. Evaluate class-wise precision, recall, F1, balanced accuracy and confusion matrices, together with continuous rainfall verification metrics.
9. Preserve the original model value, verifying observation, timestamps and coordinates so every ML label remains auditable.

A recent rainfall-ML study similarly used chronological validation and restricted preprocessing/resampling to training folds to avoid leakage. citeturn0search1turn0search2

## Important interpretation

The error classes are diagnostic categories, not universal physical standards. The default rainfall-intensity and error thresholds should be reported in a paper and adapted to the temporal resolution, region and scientific question. For publication, compare conclusions under sensitivity ranges rather than relying on a single arbitrary threshold.
