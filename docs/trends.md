# Robust trend analysis

`climatevar` provides classical Mann-Kendall (MK), Sen's slope, an autocorrelation-aware modified MK, trend-free pre-whitening (TFPW), calendar-aware seasonal trends, spatial FDR, regional trends, moving-block confidence intervals for Sen slopes, and a block-bootstrap field-significance diagnostic.

## Seasonal aggregation and completeness

`seasonal_series` now handles monthly, daily, and regular sub-daily data directly. It detects precipitation-like variables and sums them by default; state variables such as temperature are averaged. Use `aggregation` explicitly when the scientific definition is known.

```python
from climatevar.trends import seasonal_series, seasonal_mann_kendall, seasonal_sen_slope

# Daily precipitation: calendar-aware JJA total
jja_rain = seasonal_series(daily_rain, "JJA", aggregation="sum", min_valid_fraction=0.9)
jja_mk = seasonal_mann_kendall(daily_rain, "JJA", aggregation="sum", min_valid_fraction=0.9)
jja_slope = seasonal_sen_slope(daily_rain, "JJA", aggregation="sum", min_valid_fraction=0.9)

# Daily temperature: JJA mean
jja_temp = seasonal_series(daily_temperature, "JJA", aggregation="mean", min_valid_fraction=0.9)
```

Completeness is based on the expected number of samples for the detected frequency and calendar, including leap years, no-leap calendars, all-leap calendars, and 360-day calendars. Incomplete seasons are returned as missing rather than silently treated as complete. Irregular time axes should be pre-aggregated before seasonal trend analysis.

For DJF, December is assigned to the following January/February season-year. This avoids ambiguous December/January labeling.

## Classical MK

```python
from climatevar.trends import mann_kendall, sens_slope
mk = mann_kendall(annual_rain)
slope = sens_slope(annual_rain)
```

Classical MK assumes serial independence. Positive autocorrelation can inflate apparent significance, so persistent climate and hydrological series should report the autocorrelation treatment used.

## Modified Mann-Kendall

```python
from climatevar.trends import modified_mann_kendall
mmk = modified_mann_kendall(annual_rain)
```

The implementation uses a Yue-Wang-style effective-sample-size correction after removing the Sen-slope component before estimating persistence. It is intentionally described as an effective-sample-size correction rather than a Hamed-Rao implementation.

## Trend-free pre-whitening

```python
from climatevar.trends import trend_free_prewhitening, mann_kendall
prewhitened = trend_free_prewhitening(annual_rain)
mk_pw = mann_kendall(prewhitened)
```

TFPW removes the estimated monotonic component, estimates lag-1 persistence, removes the AR(1) component, and restores the trend component. Report the exact pre-whitening method because variants can produce different significance estimates.

## Sen slope confidence intervals

```python
from climatevar.trends import sens_slope_ci
ci = sens_slope_ci(annual_rain, alpha=0.05, n_resamples=2000, block_length=5, random_state=42)
```

`sens_slope_ci` reports the Sen slope together with percentile confidence limits from a moving-block bootstrap. Block resampling is preferred to an IID bootstrap when observations are serially dependent because it retains local dependence within blocks. The default block length is data-length dependent; for publication, explicitly report the selected block length and sensitivity-test it. The slope remains expressed per observation step, so convert it to physical units such as mm/year only when the observation interval is known and regular.

For an area-weighted regional series:

```python
from climatevar.trends import regional_slope_ci
regional_ci = regional_slope_ci(
    annual_rain,
    lat_dim="latitude",
    lon_dim="longitude",
    n_resamples=2000,
    block_length=5,
    random_state=42,
)
```

The regional field is first reduced to one area-weighted time series and the bootstrap is then performed on that series. This avoids treating grid cells as independent bootstrap observations.

Confidence intervals quantify uncertainty under the selected resampling model; they do not replace the separate autocorrelation-aware trend-significance test.

## Spatial trend fields and FDR

```python
from climatevar.trends import spatial_trend
trend = spatial_trend(annual_rain, alpha=0.05)
```

The result includes local modified-MK p-values, Sen slopes, an uncorrected significance mask and a Benjamini-Hochberg FDR mask. FDR controls the expected false-discovery proportion among rejected local hypotheses; it does not by itself answer whether the field contains a significant trend.

## Regional trend

```python
from climatevar.trends import regional_mean, regional_trend
series = regional_mean(annual_rain, lat_dim="latitude", lon_dim="longitude")
summary = regional_trend(annual_rain, lat_dim="latitude", lon_dim="longitude")
```

Regular geographic grids use cosine-latitude weighting by default. Supply exact cell-area weights when available. Regional trend results answer a different question from grid-cell significance and should be reported separately.

## Field significance beyond FDR

```python
from climatevar.trends import field_significance
field = field_significance(annual_rain, block_length=5, n_resamples=1000, random_state=42)
```

This performs a moving-block bootstrap on detrended residual fields. A common set of resampled time indices is applied to every grid cell, preserving the spatial covariance structure of the sampled field while retaining short-range temporal dependence through the blocks. The observed number of locally significant cells is compared with the bootstrap null distribution.

`field_pvalue` answers a field-level question; `significant_fdr` remains available for local multiple-testing control. The bootstrap count uses classical MK on the null residual fields, while observed local p-values use modified MK. For publication, report block length, number of resamples, random seed, completeness rule, local trend method, and weighting scheme.

## Recommended publication workflow

1. Verify time ordering, calendar and seasonal completeness.
2. Convert precipitation to physically appropriate amounts before aggregation.
3. Construct the seasonal series with an explicit `aggregation` and completeness threshold.
4. Report Sen slope and classical MK as a baseline.
5. Diagnose serial dependence and use modified MK or TFPW when justified.
6. Report a bootstrap confidence interval for the Sen slope, with the block-length rationale.
7. For spatial fields, report local p-values together with FDR results.
8. If making a field-wide claim, add a documented block-bootstrap field-significance test.
9. For regional claims, report the area-weighted regional series, trend and confidence interval separately.
10. Report slope magnitude, units, sample period and uncertainty/significance—not p-value alone.
