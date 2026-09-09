# Peaks-over-threshold diagnostics

`climatevar` provides a transparent POT workflow for rainfall extremes: threshold exceedances → optional runs declustering → GPD fit → threshold diagnostics → return level → bootstrap uncertainty.

## 1. Do not choose a threshold from one fit

A high threshold is required for the asymptotic GPD approximation, but increasing the threshold reduces the number of exceedances and increases sampling uncertainty. Evaluate a scientifically defensible grid of candidate thresholds.

```python
from climatevar.extremes import threshold_diagnostics, pot_threshold_sensitivity

diag = threshold_diagnostics(rain, [80, 90, 100, 110, 120, 130])
sensitivity = pot_threshold_sensitivity(
    rain, [80, 90, 100, 110, 120, 130], return_period=100 * 365.25
)
```

Inspect:

- `n_exceedances` and `exceedance_rate` — whether enough tail observations remain;
- `mean_excess` — empirical mean residual life behavior;
- `shape` and `scale` — parameter stability as the threshold changes;
- `return_level` — sensitivity of the scientific quantity of interest.

There is no universal numerical threshold. Choose a region where the tail diagnostics are reasonably stable, while retaining enough independent extremes for estimation, and document the rationale.

## 2. Account for temporal clustering

Daily monsoon rainfall can contain serially dependent extreme days. If the intended GPD likelihood assumes approximately independent exceedances, use runs declustering and retain the maximum from each cluster.

```python
from climatevar.extremes import decluster_exceedances, gpd_fit

peaks = decluster_exceedances(rain, threshold=100, run_length=3)
excess = peaks - 100
fit = gpd_fit(excess)
```

The run length is not a universal constant. Test sensitivity to plausible run lengths based on the dependence structure and storm duration relevant to the study region.

## 3. Check GPD fit quality

```python
from climatevar.extremes import gpd_goodness_of_fit

gof = gpd_goodness_of_fit(excess)
print(gof[["ks_statistic", "ks_pvalue", "ad_statistic", "n_exceedances"]])
```

The diagnostic returns probability-integral-transform values and GPD QQ data in addition to KS and Anderson-Darling statistics. Because the parameters are estimated from the same exceedance sample, the reported KS p-value should be treated as descriptive rather than as an exact calibrated hypothesis test. For formal inference, calibrate the statistic with a parametric bootstrap or simulation under the fitted model.

## 4. Return levels and units

`exceedance_rate` and `return_period` must use the same time unit. For daily rainfall, a 100-year return period can be represented as approximately `100 * 365.25` days when using a rate per day.

```python
from climatevar.extremes import pot_return_level

rl100 = pot_return_level(
    threshold=100,
    shape=float(fit.shape),
    scale=float(fit.scale),
    exceedance_rate=float(fit.n_exceedances / rain.count()),
    return_period=100 * 365.25,
)
```

Return periods shorter than the mean recurrence interval of threshold exceedances are rejected because the standard POT return-level expression is intended to describe levels above the threshold.

## 5. Uncertainty

```python
from climatevar.extremes import pot_return_level_ci

ci = pot_return_level_ci(
    rain,
    threshold=100,
    return_period=100 * 365.25,
    decluster_run_length=3,
    n_resamples=2000,
    random_state=42,
)
```

The bootstrap quantifies sampling uncertainty conditional on the chosen threshold, declustering rule, and stationary GPD model. It does **not** automatically quantify uncertainty from threshold selection, dependence-model choice, or non-stationarity. Report a threshold-sensitivity envelope alongside the selected estimate when these sources matter.

## Publication checklist

Before using a POT result in a paper, record:

1. variable and temporal resolution;
2. threshold candidates and final threshold;
3. threshold-selection evidence (mean residual life and parameter stability);
4. minimum exceedance/cluster count used for fitting;
5. declustering rule and run length, if used;
6. GPD estimation method;
7. return-period time unit and exceedance-rate definition;
8. goodness-of-fit diagnostics and how their uncertainty was calibrated;
9. bootstrap size and random seed;
10. sensitivity of return levels to threshold and declustering choices.

This workflow is deliberately diagnostic rather than automatic: the package should help make an extreme-value analysis reproducible without silently choosing a threshold or overstating statistical significance.
