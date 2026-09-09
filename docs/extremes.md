# Extreme-value analysis

`climatevar` provides both block-maxima GEV analysis and peaks-over-threshold (POT) analysis. The POT workflow is deliberately diagnostic-first: threshold choice is exposed to the researcher rather than selected silently.

## POT workflow

```python
from climatevar.extremes import (
    pot_exceedances, gpd_fit, threshold_diagnostics,
    decluster_exceedances, pot_threshold_sensitivity,
    gpd_goodness_of_fit, gpd_goodness_of_fit_bootstrap,
    pot_threshold_uncertainty, pot_return_level_ci,
)

rain = xr.open_dataarray("daily_rainfall.nc")
u = 100.0  # mm/day; justify for the study region

excess = pot_exceedances(rain, u)
fit = gpd_fit(excess)

diag = threshold_diagnostics(rain, [80, 90, 100, 110, 120])
peaks = decluster_exceedances(rain, u, run_length=3)

sensitivity = pot_threshold_sensitivity(rain, [80, 90, 100, 110, 120], 100 * 365.25)
gof = gpd_goodness_of_fit(excess)
gof_boot = gpd_goodness_of_fit_bootstrap(excess, n_resamples=2000, random_state=42)
threshold_range = pot_threshold_uncertainty(rain, [80, 90, 100, 110, 120], 100 * 365.25)
ci = pot_return_level_ci(rain, u, 100 * 365.25, decluster_run_length=3,
                         n_resamples=2000, random_state=42)
```

## Threshold selection

A POT threshold should be high enough for the GPD approximation to be credible while retaining enough exceedances for stable estimation. Use the combination of:

- exceedance count/rate;
- mean residual life (`mean_excess`);
- GPD shape stability;
- scale stability;
- return-level stability;
- scientific knowledge of the variable and event process.

`threshold_diagnostics` and `pot_threshold_sensitivity` provide the numerical evidence needed for this assessment. They do not automatically choose a threshold.

## Dependence and declustering

Daily rainfall extremes can occur in temporal clusters because a single storm system may produce several consecutive exceedances. `decluster_exceedances` applies a runs rule and retains the maximum observation from each cluster. The `run_length` should be justified from the dependence structure and sampling interval.

## Goodness of fit

`gpd_goodness_of_fit` provides probability-integral-transform, QQ and descriptive KS/Anderson-Darling diagnostics. Because the GPD parameters are estimated from the same observations, the ordinary test p-values should not be interpreted as calibrated goodness-of-fit inference.

`gpd_goodness_of_fit_bootstrap` addresses this limitation by repeatedly simulating from the fitted GPD, refitting the parameters, and comparing the observed statistic with its fitted-parameter bootstrap distribution. The resulting p-values are simulation-calibrated under the fitted stationary GPD model, but they still inherit the assumptions of that model.

## Threshold uncertainty

`pot_threshold_uncertainty` reports an empirical return-level envelope over candidate thresholds. It is intentionally **not** presented as a conventional confidence interval: candidate thresholds are not probability-weighted models. For publication, report the selected threshold, the defensible threshold range, sensitivity of the fitted parameters/return levels, and the rationale for the final choice.

The existing `pot_return_level_ci` bootstrap interval quantifies sampling uncertainty conditional on the selected threshold. It does not automatically incorporate threshold-selection uncertainty.

## Non-stationary GPD

`gpd_fit_nonstationary` provides a conservative first non-stationary model in which the GPD shape parameter is constant and the scale parameter follows a log-linear covariate model:

`log(sigma_t) = beta_0 + beta_1 x_t`.

The covariate is standardized internally and the fitted mean/standard deviation are returned so the model can be reproduced. This is a modelling foundation, not an automatic declaration that non-stationarity is present. Compare stationary and non-stationary formulations using scientifically justified diagnostics and out-of-sample or information-criterion evidence before interpreting covariate effects.

## Return-period units

The exceedance rate and return period must use the same observation unit. For daily rainfall, a rate expressed per day should be combined with a return period expressed in days. For annual rates, use years. A return period shorter than the mean threshold-exceedance recurrence interval is rejected because the POT return-level formula is intended to describe levels above the threshold.

## Publication checklist

For a rainfall POT analysis, report at minimum:

1. data source and version;
2. temporal resolution and units;
3. threshold and threshold-selection diagnostics;
4. wet/extreme-event definition;
5. declustering rule and run length, if used;
6. number and rate of retained exceedances;
7. GPD parameter estimates and parameter convention;
8. return-period unit;
9. confidence/uncertainty method and bootstrap size;
10. random seed for stochastic calculations;
11. goodness-of-fit diagnostics;
12. sensitivity to threshold choice; and
13. whether stationarity or covariate dependence was assumed.
