# Robust trend analysis

`climatevar` provides classical Mann-Kendall (MK), Sen's slope, an autocorrelation-aware modified MK, trend-free pre-whitening (TFPW), seasonal trends, spatial FDR, regional trends, and a block-bootstrap field-significance diagnostic.

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
print(mmk["tau"])
print(mmk["pvalue"])
print(mmk["n_eff"])
```

The implementation uses a Yue-Wang-style effective-sample-size correction after removing the Sen-slope component before estimating persistence. It is intentionally described as an effective-sample-size correction rather than a Hamed-Rao implementation.

## Trend-free pre-whitening

```python
from climatevar.trends import trend_free_prewhitening, mann_kendall

prewhitened = trend_free_prewhitening(annual_rain)
mk_pw = mann_kendall(prewhitened)
```

TFPW removes the estimated monotonic component, estimates lag-1 persistence, removes the AR(1) component, and restores the trend component. Report the exact pre-whitening method because different variants can produce different significance estimates.

## Seasonal trends

```python
from climatevar.trends import seasonal_series, seasonal_mann_kendall, seasonal_sen_slope

jja = seasonal_series(monthly_rain, "JJA")
jja_mk = seasonal_mann_kendall(monthly_rain, "JJA")
jja_slope = seasonal_sen_slope(monthly_rain, "JJA")
```

`seasonal_series` aggregates complete climatological seasons by summation and is therefore intended for monthly precipitation totals. For temperature or other state variables, construct a seasonal mean first and then apply the trend functions.

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

Regular geographic grids use cosine-latitude weighting by default. Supply exact cell-area weights when the grid is curvilinear or when precise cell areas are available.

## Field significance beyond FDR

```python
from climatevar.trends import field_significance

field = field_significance(
    annual_rain,
    block_length=5,
    n_resamples=1000,
    random_state=42,
)
```

This performs a moving-block bootstrap on detrended residual fields. A common set of resampled time indices is applied to every grid cell, preserving the spatial covariance structure of the sampled field while retaining short-range temporal dependence through the blocks. The observed number of locally significant cells is compared with the bootstrap null distribution.

The returned `field_pvalue` answers a field-level question: how often does the null produce at least as many locally significant cells as observed? `significant_fdr` remains available for local multiple-testing control. The two diagnostics are complementary.

The bootstrap count uses classical MK on the null residual fields, while observed local p-values use modified MK. For publication, report the block length, number of resamples, random seed, completeness rule, local trend method, and weighting scheme.

## Recommended publication workflow

For annual or seasonal rainfall/drought indices:

1. Verify temporal completeness and flag invalid years.
2. Report Sen slope and classical MK as a baseline.
3. Diagnose serial dependence.
4. Use modified MK or TFPW when persistence is material.
5. For spatial fields, report local p-values together with FDR results.
6. If making a field-wide claim, add a documented block-bootstrap field-significance test.
7. For regional claims, report the area-weighted regional series and its trend separately from grid-cell results.
8. Report slope magnitude, units, sample period and uncertainty/significance—not p-value alone.

For long-range persistent hydrological/climate series, more advanced long-memory or scaling-aware methods may be appropriate; these should not be silently substituted for the effective-sample-size method.
