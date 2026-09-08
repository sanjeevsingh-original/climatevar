# API cookbook

Practical examples for the public `climatevar` functions. All examples use xarray objects and can be adapted to ERA5, ERA5-Land, IMERG/GPM, IMDAA, CMIP6, WRF and station observations.

## Imports

```python
import numpy as np
import pandas as pd
import xarray as xr
```

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

# Climatology and anomalies

```python
from climatevar.climatology import climatology, anomaly
monthly = climatology(rain, group="month")
anom = anomaly(rain, monthly, group="month")
```

---

# Quality control

```python
from climatevar.quality import valid_fraction, require_completeness, annual_valid_fraction
from climatevar.time import year_complete_mask
fraction = valid_fraction(rain, dim="time")
annual = annual_valid_fraction(rain, dim="time")
rain_qc = require_completeness(rain, min_fraction=0.9)
complete = year_complete_mask(rain, min_valid_fraction=0.9)
```

---

# Time and calendars

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

# Seasonal analysis

```python
from climatevar.seasonal import seasonal_mean, monsoon_total
jja = seasonal_mean(rain, "JJA")
ism = monsoon_total(rain, start_month=6, end_month=9)
```

---

# Precipitation normalization

```python
from climatevar.precipitation import interval_amount_mm, normalize_precipitation, daily_amount
amount = interval_amount_mm(precip)
hourly = normalize_precipitation(hourly_rain)       # mm/hr
three_hour = normalize_precipitation(rain_3hour)   # mm/3hr
daily_rate = normalize_precipitation(daily_rain)   # mm/day
daily = daily_amount(hourly_rain)
```

---

# Precipitation extremes

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

# WRF precipitation

```python
from climatevar.precipitation import wrf_total_precipitation, wrf_precipitation_amount
wrf = xr.open_dataset("wrfout_d01_2024-07-01_00:00:00")
total = wrf_total_precipitation(wrf)
interval = wrf_precipitation_amount(wrf)
```

---

# Thermodynamics and moisture

```python
from climatevar.thermodynamics import potential_temperature, equivalent_potential_temperature
from climatevar.thermodynamics import saturation_vapor_pressure, specific_humidity_from_rh
es = saturation_vapor_pressure(T)
q = specific_humidity_from_rh(T, p, rh)
theta = potential_temperature(T, p)
theta_e = equivalent_potential_temperature(T, p, q)
```

---

# Trends

## Classical and autocorrelation-aware trends

```python
from climatevar.trends import mann_kendall, modified_mann_kendall, sens_slope, trend_free_prewhitening

mk = mann_kendall(annual_rain)
mmk = modified_mann_kendall(annual_rain)
slope = sens_slope(annual_rain)
tfpw_series = trend_free_prewhitening(annual_rain)
```

Classical MK assumes serial independence. The modified test accounts for serial correlation through an effective-sample-size variance correction; this is important for climate and hydrological series because positive autocorrelation can inflate apparent significance. citeturn0search0

## Seasonal MK and Sen slope

For monsoon trend studies, calculate the seasonal series first and then apply an autocorrelation-aware trend test:

```python
from climatevar.trends import seasonal_series, seasonal_mann_kendall, seasonal_sen_slope

jja = seasonal_series(monthly_rain, "JJA")
ism = seasonal_series(monthly_rain, "JJA")

jja_mk = seasonal_mann_kendall(monthly_rain, "JJA")
jja_slope = seasonal_sen_slope(monthly_rain, "JJA")
```

`seasonal_series` uses complete climatological seasons. DJF is assigned to the year in which January and February occur, avoiding a December/January boundary ambiguity. Seasonal MK is preferred over pooling all monthly observations when seasonality is intrinsic to the variable. citeturn0search11

> **Important:** `seasonal_series` currently aggregates the selected season by summation. For temperature or other state variables, first construct the appropriate seasonal mean series with `seasonal_mean`, then apply `modified_mann_kendall` or `sens_slope` to that series.

## Spatial trend fields

```python
from climatevar.trends import spatial_trend

trend = spatial_trend(
    annual_rain,
    dim="time",
    alpha=0.05,
)

trend["sen_slope"]
trend["pvalue"]
trend["significant"]
trend["significant_fdr"]
```

`spatial_trend` returns the trend statistic, Kendall tau, autocorrelation-aware p-value, effective sample size, Sen slope, uncorrected significance and Benjamini-Hochberg FDR significance. Multiple testing should be considered when interpreting grid-cell significance fields rather than treating every uncorrected `p < 0.05` cell as an independent discovery. citeturn0search13

For a different family of tests, use `fdr_mask` directly:

```python
from climatevar.trends import fdr_mask
significant = fdr_mask(trend["pvalue"], alpha=0.05)
```

---

# GEV

```python
from climatevar.extremes import gev_fit, gev_return_level
fit = gev_fit(annual_maxima)
rl50 = gev_return_level(fit["shape"], fit["loc"], fit["scale"], 50)
```

---

# Spatial analysis

```python
from climatevar.spatial import cosine_latitude_weights, area_weighted_mean
weights = cosine_latitude_weights(rain["latitude"])
regional = area_weighted_mean(rain, lat_dim="latitude", spatial_dims=("latitude", "longitude"))
```

---

# Grid utilities

```python
from climatevar.grid import normalize_longitude, regrid
ds180 = normalize_longitude(ds, center=0.0)
out = regrid(ds["precipitation"], target, method="linear")
```

---

# Verification metrics

```python
from climatevar.metrics import bias, mae, rmse, correlation, nse
bias_map = bias(obs, model, dim="time")
mae_map = mae(obs, model, dim="time")
rmse_map = rmse(obs, model, dim="time")
corr_map = correlation(obs, model, dim="time")
nse_map = nse(obs, model, dim="time")
```

---

# Dataset interoperability

```python
from climatevar.io.normalize import normalize_coords, standardize_variables, normalize_dataset, find_variable
coords = normalize_coords(raw, dataset="era5")
standard = standardize_variables(raw, variables={"precipitation": "tp", "temperature": "t2m"})
era5 = normalize_dataset(raw, dataset="era5")
cmip6 = normalize_dataset(raw_cmip6, dataset="cmip6")
name = find_variable(raw, "precipitation")
```

---

# Schema and station data

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

# Complete monsoon + trend workflow

```python
import xarray as xr
from climatevar.io.normalize import normalize_dataset
from climatevar.precipitation import daily_amount, PRCPTOT, Rx1day
from climatevar.seasonal import monsoon_total
from climatevar.drought import spi
from climatevar.trends import seasonal_mann_kendall, seasonal_sen_slope, spatial_trend

raw = xr.open_dataset("era5.nc")
ds = normalize_dataset(raw, dataset="era5")
daily = daily_amount(ds["precipitation"])

rx1 = Rx1day(daily)
prcptot = PRCPTOT(daily)
ism = monsoon_total(daily)

# Annual monsoon trend
ism_mk = seasonal_mann_kendall(daily, "JJA")
ism_slope = seasonal_sen_slope(daily, "JJA")

# Grid-cell trend + FDR correction
field = spatial_trend(ism)
```

## Research reproducibility checklist

For publication-quality work, report dataset/version, domain, temporal aggregation, units, calendar, completeness rule, SPI/SPEI distribution and calibration period, PET method for SPEI, percentile baseline, event thresholds, spatial weighting, trend method, autocorrelation treatment, significance level, and multiple-testing correction.
