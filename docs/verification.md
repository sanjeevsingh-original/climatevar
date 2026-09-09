# Precipitation verification

`climatevar.metrics` provides continuous, categorical, probabilistic, and spatial verification tools for comparing observations with reanalysis, satellite, model, or forecast precipitation.

## Recommended workflow

For Indian monsoon applications, first harmonize units and temporal resolution, then put datasets on a common grid and compare against an independent reference. Do not assume that one product is universally superior: recent India-wide evaluations show performance varies with region, scale, season, and precipitation intensity. IMDAA is a high-resolution Indian reanalysis, while ERA5 provides a global reference; both should be evaluated rather than treated as truth.

```python
from climatevar.io import normalize_dataset
from climatevar.metrics import bias, correlation, rmse, threat_score, equitable_threat_score

obs = normalize_dataset(obs)
pred = normalize_dataset(pred)

mean_bias = bias(obs.pr, pred.pr, dim="time")
r = correlation(obs.pr, pred.pr, dim="time")
error = rmse(obs.pr, pred.pr, dim="time")
ts = threat_score(obs.pr, pred.pr, threshold=20, dim="time")
ets = equitable_threat_score(obs.pr, pred.pr, threshold=20, dim="time")
```

## Metric families

- **Continuous:** bias, MAE, RMSE, correlation, NSE.
- **Categorical:** POD, FAR, F1, HSS, threat score, ETS, frequency bias.
- **Probabilistic:** Brier score and reliability/resolution/uncertainty components.
- **Spatial:** Fractions Skill Score for regular latitude/longitude grids.
- **Precipitation distribution/extremes:** ETCCDI indices and EVT tools should be evaluated separately from mean-error metrics.

For event thresholds, report several scientifically meaningful thresholds rather than one arbitrary cutoff (for example 1, 10, 20, 50 and 100 mm/day where sample size permits). For extremes, retain the native temporal resolution until the analysis explicitly defines the accumulation period.

## Independent-reference principle

IMD gridded rainfall is commonly used as a reference in Indian precipitation evaluations, but it is still a gridded observational product with its own representativeness and interpolation uncertainty. Satellite, reanalysis and model products should therefore be described as alternative estimates, not automatically as truth.

## ML/DL error-tagging preparation

The verification layer can provide features for later ML/DL error tagging:

1. Calculate grid-cell bias, RMSE, correlation and event scores.
2. Add precipitation intensity, season, elevation, distance from coast and rainfall regime as predictors.
3. Define error classes independently of the predictors to avoid leakage.
4. Split training/validation/test data by time or event, not randomly across individual grid cells when spatial-temporal dependence matters.
5. Preserve the original forecast/model value and verification reference so every ML label remains traceable.
