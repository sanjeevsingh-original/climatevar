# Extreme-value analysis

`climatevar` provides block-maxima GEV and peaks-over-threshold (POT) analysis. The POT workflow is diagnostic-first: threshold choice, dependence treatment, uncertainty and stationarity are explicit.

## Reproducible POT workflow

```python
import xarray as xr
from climatevar.extremes import (
    pot_exceedances, gpd_fit, threshold_diagnostics, decluster_exceedances,
    pot_threshold_sensitivity, gpd_goodness_of_fit,
    gpd_goodness_of_fit_bootstrap, pot_threshold_bootstrap,
    pot_return_level_ci, gpd_model_comparison_bootstrap,
    gpd_nonstationary_ci, pot_diagnostic_report,
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
threshold_ci = pot_threshold_bootstrap(rain, [80, 90, 100, 110, 120], 100 * 365.25,
                                       n_resamples=2000, random_state=42)
ci = pot_return_level_ci(rain, u, 100 * 365.25, decluster_run_length=3,
                         n_resamples=2000, random_state=42)
```

## Threshold selection and uncertainty

A POT threshold should be high enough for the GPD approximation to be credible while retaining enough exceedances for stable estimation. Examine exceedance counts/rates, mean residual life, shape/scale stability, return-level stability, dependence and physical event structure.

`threshold_diagnostics` and `pot_threshold_sensitivity` expose these diagnostics without silently selecting a threshold. `pot_threshold_uncertainty` gives an empirical return-level envelope across candidate thresholds. It is **not** a probability confidence interval.

`pot_threshold_bootstrap` goes one step further: each bootstrap sample refits every candidate threshold and applies an explicit AIC-based threshold-selection rule. The resulting percentile interval propagates sampling variability through that stated selection procedure. It should be reported as conditional on the candidate set and selection rule, not as universal threshold-selection uncertainty.

## Dependence and declustering

Daily rainfall extremes can cluster because one storm can generate several consecutive exceedances. `decluster_exceedances` applies a runs rule and retains the cluster maximum. Select `run_length` using the temporal dependence structure and sampling interval; report the rule explicitly.

## Goodness of fit

`gpd_goodness_of_fit` provides PIT, QQ and descriptive KS/Anderson-Darling diagnostics. Ordinary KS/AD p-values are not calibrated for the fact that parameters were estimated from the same observations.

`gpd_goodness_of_fit_bootstrap` repeatedly simulates from the fitted GPD, refits it and calibrates KS/AD statistics. The result is a fitted-model bootstrap diagnostic, not a model-free test.

## Non-stationary GPD: comparison and inference

The baseline non-stationary model keeps shape constant and models scale as

`log(sigma_t) = beta_0 + beta_1 x_t`.

`gpd_model_comparison` reports log-likelihood, AIC, BIC and an asymptotic likelihood-ratio statistic. `gpd_model_comparison_bootstrap` provides a parametric-bootstrap LR p-value and critical value under the fitted stationary null. For publication, prefer the bootstrap result when sample size or regularity assumptions make asymptotic inference questionable.

`gpd_nonstationary_ci` provides percentile bootstrap intervals for shape, intercept and covariate-scale coefficient. A covariate effect should only be interpreted after checking the physical rationale, residual/QQ diagnostics, dependence and model adequacy.

## Return-period units

The exceedance rate and return period must use the same observation unit. For daily rainfall, use a rate per day and a return period in days; for annual exceedances, use years. Return periods at or below the threshold-exceedance recurrence interval are rejected.

## Publication-quality figures

With the optional `plot` dependency:

```python
from climatevar.extremes import plot_threshold_stability, plot_mean_excess, plot_gpd_qq

ax = plot_threshold_stability(diag)
ax.figure.savefig("threshold_stability.png", dpi=300, bbox_inches="tight")
ax = plot_mean_excess(diag)
ax.figure.savefig("mean_excess.png", dpi=300, bbox_inches="tight")
ax = plot_gpd_qq(gof)
ax.figure.savefig("gpd_qq.png", dpi=300, bbox_inches="tight")
```

The plotting functions intentionally return Matplotlib axes so journal-specific fonts, dimensions and annotation can be applied by the researcher without hidden styling.

## Automated diagnostic report

`pot_diagnostic_report` creates a self-contained HTML report from the numerical diagnostics:

```python
html = pot_diagnostic_report(diag, gof=gof, uncertainty=threshold_ci)
with open("pot_report.html", "w", encoding="utf-8") as f:
    f.write(html)
```

The report records threshold diagnostics and a publication checklist. It does not replace scientific interpretation or peer-review-quality figure checking.

## Publication checklist

Report: data source/version; temporal resolution and units; threshold candidates and selection rule; event definition; declustering rule; number/rate of retained exceedances; GPD parameter convention; return-period unit; uncertainty method and bootstrap size; random seed; GOF diagnostics; threshold sensitivity; stationarity assumption; covariates and model comparison where relevant.
