# ML/DL rainfall error tagging

`climatevar` provides an optional machine-learning layer for classifying rainfall-model errors after verification. The core package does not depend on scikit-learn, XGBoost or SHAP.

## Scientific design

The workflow is explicitly separated into two tasks:

1. **Diagnostic/post-hoc tagging** — observations define the error target. This answers *where, when and under what rainfall regimes a model fails*.
2. **Prospective error prediction** — a classifier predicts the likely error class using only information available before the verifying observation. Observation-derived variables must not enter the predictor matrix.

For rainfall grids, random row-wise splitting is discouraged because nearby samples can be correlated in time and space. Use chronological or group/spatial holdouts. `TimeSeriesSplit` is intended for ordered observations, while `GroupKFold` keeps groups out of both training and test folds; `StratifiedGroupKFold` additionally attempts to preserve class proportions. citeturn0search0turn0search1turn0search2

## Baseline models

Install the optional ML stack:

```bash
pip install "climatevar[ml]"
```

Available baselines:

- `hist_gradient_boosting` — fast nonlinear baseline for large tabular datasets.
- `random_forest` — robust tree ensemble baseline.
- `xgboost` — optional external gradient-boosting implementation.

Class-balanced weighting is enabled by default where supported, which is important when severe-error classes are rare. Scikit-learn's histogram gradient boosting supports `class_weight="balanced"`. citeturn0search11

## Example

```python
from climatevar.error_tagging import (
    build_error_features,
    to_ml_table,
    fit_classifier,
    classification_metrics,
)

features = build_error_features(observation, prediction, elevation=elevation)
table = to_ml_table(features)

# For prospective prediction, select only predictor columns that do not use truth.
X = table[["prediction", "month", "latitude", "longitude", "elevation"]]
y = table["error_class"]

model = fit_classifier(X, y, model="hist_gradient_boosting")
pred = model.predict(X)
prob = model.predict_proba(X)
print(classification_metrics(y, pred, prob))
```

For a scientifically defensible experiment, the example above should be evaluated on held-out time periods or independent spatial/event groups rather than on the training rows.

## Probability calibration

`fit_classifier(..., calibrate=True)` applies sigmoid calibration. Report both discrimination and probability quality; accuracy alone is insufficient for a multi-class rainfall-error problem.

Recommended metrics include balanced accuracy, macro-F1, macro precision/recall, Matthews correlation coefficient, log loss and one-vs-rest ROC-AUC when estimable.

## Explainability

Two levels are provided:

- `permutation_importance(...)` for model-agnostic validation-set importance.
- `shap_values(...)` for optional TreeSHAP explanations.

Permutation importance should preferentially be computed on held-out data because tree impurity importance can be biased and can overstate features that do not generalize. citeturn0search5

## Dataset comparison plan

The framework is designed for a common observation/reference target with model-specific predictors, allowing the same experiment to be repeated for:

`ERA5 → IMERG/GPM → IMDAA/IMD → WRF → CMIP6`

For each dataset, retain provenance fields identifying source, resolution, accumulation period, interpolation/regridding method and verification reference. Compare models using identical spatial domain, temporal overlap, rainfall units, event definition, class thresholds and leakage-safe folds.

## Recommended research protocol

1. Harmonize units and temporal accumulation before verification.
2. Regrid only after defining the independent reference and document the interpolation method.
3. Compute deterministic, event-based and spatial precipitation scores.
4. Generate error classes independently of model fitting.
5. Build prospective predictors without observed-rainfall information.
6. Use time-, event- or spatial-blocked validation.
7. Tune hyperparameters only inside the training portion of each fold.
8. Calibrate probabilities using training/validation data only.
9. Report class-wise metrics and uncertainty, not only overall accuracy.
10. Apply permutation/SHAP diagnostics only after the final leakage-safe evaluation.
11. Preserve the original observation, prediction, error definition and dataset provenance for every sample.

Deep neural networks should be added only after these tabular baselines establish a reproducible benchmark. CNN/ConvLSTM/Transformer models can then use spatial-temporal neighborhoods while retaining the same held-out evaluation protocol.
