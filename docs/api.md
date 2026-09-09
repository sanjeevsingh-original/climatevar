# API cookbook

Practical examples for the public `climatevar` functions. All examples use xarray objects and can be adapted to ERA5, ERA5-Land, IMERG/GPM, IMDAA, CMIP6, WRF and station observations.

## Imports

```python
import numpy as np
import pandas as pd
import xarray as xr
```

## Peaks-over-threshold extremes

POT analysis models excesses above a sufficiently high threshold with a generalized Pareto distribution (GPD). Threshold choice is a model assumption, not a cosmetic setting: inspect parameter stability and exceedance counts across several candidate thresholds. For daily rainfall, dependent exceedances should generally be declustered before fitting an independent-exceedance model. These diagnostics are especially important for Indian monsoon rainfall, where multi-day storm systems can generate clusters of exceedances. Recent Indian precipitation research also demonstrates substantial sensitivity of return-level estimates to threshold choice. citeturn0search0turn0search4

```python
from climatevar.extremes import (
    pot_exceedances,
    gpd_fit,
    decluster_exceedances,
    threshold_diagnostics,
    pot_return_level,
    pot_return_level_ci,
)

rain = xr.open_dataarray("daily_rainfall.nc")
threshold = 100.0  # mm/day; determine scientifically for your study region

excess = pot_exceedances(rain, threshold)
fit = gpd_fit(excess)

# Check several candidate thresholds before selecting one.
diag = threshold_diagnostics(rain, [80, 90, 100, 110, 120, 130])

# For dependent daily extremes, retain cluster maxima.
peaks = decluster_exceedances(rain, threshold, run_length=3)
peak_excess = peaks - threshold

# Example: rate per day and return period in days.
rate = float(fit["n_exceedances"] / rain.count())
rl100 = pot_return_level(
    threshold,
    float(fit["shape"]),
    float(fit["scale"]),
    rate,
    return_period=100 * 365.25,
)

ci = pot_return_level_ci(
    rain,
    threshold,
    return_period=100 * 365.25,
    decluster_run_length=3,
    n_resamples=2000,
    random_state=42,
)
```

The GPD uses the standard excess formulation with shape `xi` and scale `sigma`; negative shape implies a finite upper endpoint. POT return levels combine the fitted GPD tail with the exceedance rate, so the return-period unit must match the rate unit. citeturn0search0turn0search1

`threshold_diagnostics` provides exceedance counts/rates and GPD shape/scale estimates over a candidate threshold sequence. A stable region of the shape parameter, adequate exceedance count, and sensible mean-excess behavior are useful evidence when selecting a threshold; no single automatic threshold should be accepted without diagnostics. citeturn0search2turn0search9

`decluster_exceedances` uses a runs rule and retains the maximum from each cluster. The run length is application-specific and should be justified from the temporal dependence of the variable rather than treated as universal. POT methodology requires attention to independence because clustered extremes violate the simple independent-exceedance assumption. citeturn0search0

`pot_return_level_ci` uses a reproducible parametric-bootstrap procedure for the GPD excesses. Bootstrap uncertainty does **not** include threshold-selection uncertainty; for publication, sensitivity across a defensible threshold range should be reported separately. This is particularly important for non-stationary rainfall applications, where threshold choice can materially affect return-level estimates. citeturn0search4

---

## Formal drought indices

### `spi`

`spi` calculates the formal Standardized Precipitation Index from accumulated precipitation. The default Gamma distribution explicitly accounts for zero precipitation; Pearson Type III is also supported. The fitted distribution is estimated separately for each calendar month.

```python
from climatevar.drought import spi
monthly_rain = xr.open_dataarray("monthly_precipitation.nc")
spi3 = spi(monthly_rain, scale=3, calibration_start="1991-01-01", calibration_end="2020-12-31")
```

### `spei`

SPEI uses climatic water balance `P - PET` and a three-parameter log-logistic distribution.

```python
from climatevar.drought import spei
spei3 = spei(monthly_rain, monthly_pet, scale=3, calibration_start="1991-01-01", calibration_end="2020-12-31")
```

### `drought_category`, `fit_quality`, `spi_like`

```python
from climatevar.drought import drought_category, fit_quality, spi_like
category = drought_category(spi3)
quality = fit_quality(spi3)
legacy = spi_like(monthly_rain, scale=3)  # not formal SPI
```

---

## Climatology and anomalies

```python
from climatevar.climatology import climatology, anomaly
monthly = climatology(rain, group="month")
anom = anomaly(rain, monthly, group="month")
```

---

## Quality control

```python
from climatevar.quality import valid_fraction, require_completeness, annual_valid_fraction
from climatevar.time import year_complete_mask
fraction = valid_fraction(rain, dim="time")
annual = annual_valid_fraction(rain, dim="time")
rain_qc = require_completeness(rain, min_fraction=0.9)
complete = year_complete_mask(rain, min_valid_fraction=0.9)
```

---

## Time and calendars

```python
from climatevar.time import calendar_name, time_step_seconds, infer_frequency
from climatevar.time import expected_days_in_year, expected_days, year_complete_mask
from climatevar.time import season_year, drop_leap_day, validate_time

print(calendar_name(rain))
print(infer_frequency(rain))
print(expected_days_in_year(2024))
expected = expected_days(rain)
complete = year_complete_mask(rain)
djf_year = season_year(rain, season="DJF")
rain_365 = drop_leap_day(rain)
validate_time(rain)
```

CF time bounds are preferred when available.

---

## Seasonal analysis

```python
from climatevar.seasonal import seasonal_mean, monsoon_total
jja = seasonal_mean(rain, "JJA")
ism = monsoon_total(rain, start_month=6, end_month=9)
```

---

## Seasonal trend series

`seasonal_series` creates one value per climatological season and is calendar-aware. It supports monthly, daily, and regular sub-daily data. For precipitation totals use `aggregation="sum"`; for state variables such as temperature use `aggregation="mean"`. `aggregation="auto"` can infer the choice from precipitation-like metadata, but explicit aggregation is recommended for publication analyses.

```python
from climatevar.trends import seasonal_series, seasonal_mann_kendall, seasonal_sen_slope

jja_rain = seasonal_series(daily_rain, "JJA", aggregation="sum", min_valid_fraction=0.9)
jja_mk = seasonal_mann_kendall(daily_rain, "JJA", aggregation="sum", min_valid_fraction=0.9)
jja_slope = seasonal_sen_slope(daily_rain, "JJA", aggregation="sum", min_valid_fraction=0.9)

jja_temperature = seasonal_series(daily_temperature, "JJA", aggregation="mean", min_valid_fraction=0.9)
```

Completeness is evaluated against the expected sample count for the detected regular frequency and calendar. Incomplete seasons are masked rather than silently included. Irregular time axes should be pre-aggregated before seasonal trend analysis. For DJF, December is assigned to the following January/February season-year.

---

## Precipitation normalization

```python
from climatevar.precipitation import interval_amount_mm, normalize_precipitation, daily_amount
amount = interval_amount_mm(precip)
hourly = normalize_precipitation(hourly_rain)       # mm/hr
three_hour = normalize_precipitation(rain_3hour)   # mm/3hr
daily_rate = normalize_precipitation(daily_rain)   # mm/day
daily = daily_amount(hourly_rain)
```

---

## Precipitation extremes

```python
from climatevar.precipitation import Rx1day, Rx5day, PRCPTOT, R10mm, R20mm, R95p, R99p, CDD, CWD
rx1 = Rx1day(daily)
rx5 = Rx5day(daily)
prcptot = PRCPTOT(daily)
r10 = R10mm(daily)
r20 = R20mm(daily)
cdd = CDD(daily)
cwd = CWD(daily)
r95 = R95p(daily, baseline_start="1991-01-01", baseline_end="2020-12-31")
r99 = R99p(daily, baseline_start="1991-01-01", baseline_end="2020-12-31")
```

---

## WRF precipitation

```python
from climatevar.precipitation import wrf_total_precipitation, wrf_precipitation_amount
wrf = xr.open_dataset("wrfout_d01_2024-07-01_00:00:00")
total = wrf_total_precipitation(wrf)
interval = wrf_precipitation_amount(wrf)
```

---

## Thermodynamics and moisture

```python
from climatevar.thermodynamics import potential_temperature, equivalent_potential_temperature
from climatevar.thermodynamics import saturation_vapor_pressure, specific_humidity_from_rh
es = saturation_vapor_pressure(T)
q = specific_humidity_from_rh(T, p, rh)
theta = potential_temperature(T, p)
theta_e = equivalent_potential_temperature(T, p, q)
```

---

## Trends

### Classical and autocorrelation-aware trends

```python
from climatevar.trends import mann_kendall, modified_mann_kendall, sens_slope, trend_free_prewhitening

mk = mann_kendall(annual_rain)
mmk = modified_mann_kendall(annual_rain)
slope = sens_slope(annual_rain)
tfpw_series = trend_free_prewhitening(annual_rain)
```

Classical MK assumes serial independence. The modified test uses a Yue-Wang-style effective-sample-size variance correction after Sen-slope detrending. It should be reported explicitly rather than being labelled as Hamed-Rao unless a full Hamed-Rao formulation is implemented.

### Seasonal MK and Sen slope

```python
from climatevar.trends import seasonal_series, seasonal_mann_kendall, seasonal_sen_slope

jja = seasonal_series(monthly_rain, "JJA", aggregation="sum")
jja_mk = seasonal_mann_kendall(monthly_rain, "JJA", aggregation="sum")
jja_slope = seasonal_sen_slope(monthly_rain, "JJA", aggregation="sum")
```

For temperature or other state variables, use `aggregation="mean"`.

### Spatial trend fields and FDR

```python
from climatevar.trends import spatial_trend

trend = spatial_trend(annual_rain, dim="time", alpha=0.05)
trend["sen_slope"]
trend["pvalue"]
trend["significant"]
trend["significant_fdr"]
```

`spatial_trend` returns grid-cell trend statistics, autocorrelation-aware p-values, Sen slopes, uncorrected significance and Benjamini-Hochberg FDR significance. FDR controls the expected false-discovery proportion among rejected local tests; it does not establish that the field as a whole contains a significant trend.

### Regional trend

```python
from climatevar.trends import regional_mean, regional_trend

region_series = regional_mean(annual_rain, lat_dim="latitude", lon_dim="longitude")
region = regional_trend(annual_rain, lat_dim="latitude", lon_dim="longitude", alpha=0.05)
```

Cosine-latitude weighting is used by default for regular geographic grids. For exact cell areas or curvilinear grids, supply a two-dimensional `weights` DataArray. Regional trends should be reported separately from grid-cell significance because they answer a different scientific question.

### Field significance beyond FDR

```python
from climatevar.trends import field_significance

field = field_significance(
    annual_rain,
    dim="time",
    alpha=0.05,
    block_length=5,
    n_resamples=1000,
    random_state=42,
)

field["observed_count"]
field["n_cells"]
field["critical_count"]
field["field_pvalue"]
field["significant_fdr"]
```

`field_significance` is a field-wide Monte-Carlo diagnostic. It uses moving-block resampling of detrended residual fields and applies a common time-index sequence to all complete grid cells, preserving their spatial covariance while retaining short-range temporal dependence. The observed number of locally significant cells is compared with its null distribution.

Interpretation:

- `significant` is the local uncorrected modified-MK mask.
- `significant_fdr` is the Benjamini-Hochberg FDR mask.
- `observed_count` is the number of significant complete cells at the chosen `alpha`.
- `n_cells` is the number of complete cells included in the field-wide test.
- `critical_count` is the upper-tail bootstrap threshold for the count.
- `field_pvalue` is the Monte-Carlo probability of obtaining at least the observed count under the block-resampled, detrended null.
- `null_counts` contains the bootstrap distribution for reproducibility.

The field-wide test is complementary to FDR, not a replacement for it. The bootstrap count uses classical MK on null residuals, while observed local p-values use the modified MK implementation. For publication, report block length, number of resamples, random seed, completeness rule, local-test method and spatial weighting.

### FDR directly

```python
from climatevar.trends import fdr_mask
significant = fdr_mask(trend["pvalue"], alpha=0.05)
```

---

## GEV

```python
from climatevar.extremes import gev_fit, gev_return_level, gev_return_level_ci
fit = gev_fit(annual_maxima)
rl50 = gev_return_level(fit["shape"], fit["loc"], fit["scale"], 50)
rl50_ci = gev_return_level_ci(annual_maxima, 50, n_resamples=2000, random_state=42)
```

The GEV bootstrap interval is parametric: it treats the fitted GEV as the data-generating model and refits each bootstrap sample. Report the bootstrap size and random seed.

---

## Spatial analysis

```python
from climatevar.spatial import cosine_latitude_weights, area_weighted_mean
weights = cosine_latitude_weights(rain["latitude"])
regional = area_weighted_mean(rain, lat_dim="latitude", spatial_dims=("latitude", "longitude"))
```

---

## Grid utilities

```python
from climatevar.grid import normalize_longitude, regrid
ds180 = normalize_longitude(ds, center=0.0)
out = regrid(ds["precipitation"], target, method="linear")
```

---

## Verification metrics

```python
from climatevar.metrics import bias, mae, rmse, correlation, nse
bias_map = bias(obs, model, dim="time")
mae_map = mae(obs, model, dim="time")
rmse_map = rmse(obs, model, dim="time")
corr_map = correlation(obs, model, dim="time")
nse_map = nse(obs, model, dim="time")
```

---

## Dataset interoperability

```python
from climatevar.io.normalize import normalize_coords, standardize_variables, normalize_dataset, find_variable
coords = normalize_coords(raw, dataset="era5")
standard = standardize_variables(raw, variables={"precipitation": "tp", "temperature": "t2m"})
era5 = normalize_dataset(raw, dataset="era5")
cmip6 = normalize_dataset(raw_cmip6, dataset="cmip6")
name = find_variable(raw, "precipitation")
```

---

## Schema and station data

```python
from climatevar.io.schema import validate_dataset, describe_dataset
validate_dataset(ds, required=("precipitation", "latitude", "longitude", "time"))
print(describe_dataset(ds))
```

```python
from climatevar.io.stations import normalize_station_dataframe, station_dataframe_to_xarray
df = normalize_station_dataframe(pd.read_csv("station_rainfall.csv"))
station_ds = station_dataframe_to_xarray(df)
```

---

## Complete monsoon + trend workflow

```python
import xarray as xr
from climatevar.io.normalize import normalize_dataset
from climatevar.precipitation import daily_amount, PRCPTOT, Rx1day
from climatevar.seasonal import monsoon_total
from climatevar.trends import modified_mann_kendall, sens_slope, spatial_trend

raw = xr.open_dataset("era5.nc")
ds = normalize_dataset(raw, dataset="era5")
daily = daily_amount(ds["precipitation"])

rx1 = Rx1day(daily)
prcptot = PRCPTOT(daily)
ism = monsoon_total(daily)

ism_mk = modified_mann_kendall(ism)
ism_slope = sens_slope(ism)
field = spatial_trend(ism)
```

## Research reproducibility checklist

For publication-quality work, report dataset/version, domain, temporal aggregation, units, calendar, completeness rule, SPI/SPEI distribution and calibration period, PET method for SPEI, percentile baseline, event thresholds, POT threshold-selection diagnostics, declustering rule, GPD parameterization, exceedance-rate units, return-period units, bootstrap method/size/seed, spatial weighting, trend method, autocorrelation treatment, significance level, multiple-testing correction, and field-significance resampling design when used.
