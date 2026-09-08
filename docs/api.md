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

`spi` calculates the formal Standardized Precipitation Index from accumulated precipitation. The default Gamma distribution includes explicit zero-precipitation probability treatment; `pearson3` is also supported.

For conventional monthly SPI-3:

```python
from climatevar.drought import spi

monthly_rain = xr.open_dataarray("monthly_precipitation.nc")
spi3 = spi(monthly_rain, scale=3)
```

SPI-1, SPI-6 and SPI-12:

```python
spi1 = spi(monthly_rain, scale=1)
spi6 = spi(monthly_rain, scale=6)
spi12 = spi(monthly_rain, scale=12)
```

Pearson Type III:

```python
spi3_p3 = spi(monthly_rain, scale=3, distribution="pearson3")
```

Input precipitation must be non-negative and should represent consistent time-step amounts (normally monthly totals for standard SPI). Parameters are fitted separately for each calendar month.

### `spei`

SPEI is calculated from climatic water balance `P - PET` using a log-logistic (Fisk) distribution fitted by calendar month.

```python
from climatevar.drought import spei

monthly_pet = xr.open_dataarray("monthly_pet.nc")
spei3 = spei(monthly_rain, monthly_pet, scale=3)
spei12 = spei(monthly_rain, monthly_pet, scale=12)
```

Document the PET method and calibration/reference period in research publications. PET and precipitation must have matching dimensions and coordinates.

### `spi_like`

The legacy transparent diagnostic remains available:

```python
from climatevar.drought import spi_like
baseline = spi_like(monthly_rain, scale=3)
```

`spi_like` is **not** formal SPI and should not be labelled SPI in a publication.

---

# Climatology and anomalies

```python
from climatevar.climatology import climatology, anomaly

monthly = climatology(rain, group="month")
daily = climatology(rain, group="dayofyear")
reference = climatology(rain, group="month")
anom = anomaly(rain, reference, group="month")
```

---

# Quality control

```python
from climatevar.quality import valid_fraction, require_completeness, annual_valid_fraction

fraction = valid_fraction(rain, dim="time")
annual = annual_valid_fraction(rain, dim="time")
rain_qc = require_completeness(rain, min_fraction=0.9)
```

For calendar-aware annual completeness:

```python
from climatevar.time import year_complete_mask
complete = year_complete_mask(rain, min_valid_fraction=0.9)
```

---

# Time and calendars

```python
from climatevar.time import (
    calendar_name, time_step_seconds, infer_frequency,
    expected_days_in_year, expected_days, year_complete_mask,
    season_year, drop_leap_day, validate_time,
)

print(calendar_name(rain))
dt = time_step_seconds(rain)
print(infer_frequency(rain))
print(expected_days_in_year(2024))
print(expected_days_in_year(2024, "noleap"))
print(expected_days_in_year(2001, "360_day"))
expected = expected_days(rain)
complete = year_complete_mask(rain)
djf_year = season_year(rain, season="DJF")
rain_365 = drop_leap_day(rain)
validate_time(rain)
```

CF time bounds are preferred when available. `infer_frequency` returns practical labels such as `hourly`, `3-hourly`, `6-hourly`, `daily`, `monthly`, or `irregular`.

---

# Seasonal analysis

```python
from climatevar.seasonal import seasonal_mean, monsoon_total

jja = seasonal_mean(rain, "JJA")
djf = seasonal_mean(rain, "DJF")
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

Period totals remain `mm`; analysis-ready subdaily/daily representations retain their temporal resolution.

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
r95_external = R95p(daily, threshold=25.0)
```

---

# WRF precipitation

```python
from climatevar.precipitation import wrf_total_precipitation, wrf_precipitation_amount

wrf = xr.open_dataset("wrfout_d01_2024-07-01_00:00:00")
total = wrf_total_precipitation(wrf)
total_shallow = wrf_total_precipitation(wrf, include_shallow=True)
interval = wrf_precipitation_amount(wrf)
```

Custom component names:

```python
interval = wrf_precipitation_amount(
    wrf,
    convective="RAINC",
    nonconvective="RAINNC",
    shallow="RAINSH",
    include_shallow=False,
)
```

---

# Thermodynamics and moisture

```python
from climatevar.thermodynamics import (
    potential_temperature, equivalent_potential_temperature,
    saturation_vapor_pressure, specific_humidity_from_rh,
)

T = xr.DataArray([298.15, 303.15], dims="sample")
p = xr.DataArray([100000., 99000.], dims="sample")
rh = xr.DataArray([0.70, 0.80], dims="sample")
es = saturation_vapor_pressure(T)
q = specific_humidity_from_rh(T, p, rh)
theta = potential_temperature(T, p)
theta_e = equivalent_potential_temperature(T, p, q)
```

Temperature is K, pressure Pa, RH fractional 0–1.

---

# Trends

```python
from climatevar.trends import mann_kendall, sens_slope

mk = mann_kendall(annual_rain)
print(mk["s"], mk["tau"], mk["pvalue"], mk["n"])
slope = sens_slope(annual_rain)
```

Sen's slope is expressed per observation step.

---

# GEV

```python
from climatevar.extremes import gev_fit, gev_return_level

fit = gev_fit(annual_maxima)
rl50 = gev_return_level(fit["shape"], fit["loc"], fit["scale"], 50)
```

`gev_fit` uses SciPy's `genextreme` shape convention.

---

# Spatial analysis

```python
from climatevar.spatial import cosine_latitude_weights, area_weighted_mean

weights = cosine_latitude_weights(rain["latitude"])
regional = area_weighted_mean(
    rain,
    lat_dim="latitude",
    spatial_dims=("latitude", "longitude"),
)
```

Cosine-latitude weighting is intended for regular geographic grids. Use true cell-area weights for exact-area or curvilinear applications.

---

# Grid utilities

```python
from climatevar.grid import normalize_longitude, regrid

ds180 = normalize_longitude(ds, center=0.0)
ds360 = normalize_longitude(ds, center=180.0)

target = xr.Dataset({
    "latitude": np.arange(5, 30.1, 0.25),
    "longitude": np.arange(65, 100.1, 0.25),
})
out = regrid(ds["precipitation"], target, method="linear")
```

Methods: `linear`, `nearest`. Curvilinear grids require xESMF.

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

Categorical verification:

```python
from climatevar.metrics import contingency_table, pod, far, f1_score, heidke_skill_score

counts = contingency_table(obs, model, threshold=1.0, dim="time")
pod_map = pod(obs, model, threshold=1.0, dim="time")
far_map = far(obs, model, threshold=1.0, dim="time")
f1_map = f1_score(obs, model, threshold=1.0, dim="time")
hss_map = heidke_skill_score(obs, model, threshold=1.0, dim="time")
```

---

# Dataset interoperability

```python
from climatevar.io.normalize import normalize_coords, standardize_variables, normalize_dataset, find_variable

coords = normalize_coords(raw, dataset="era5")
standard = standardize_variables(raw, variables={"precipitation": "tp", "temperature": "t2m"})
era5 = normalize_dataset(raw, dataset="era5")
cmip6 = normalize_dataset(raw_cmip6, dataset="cmip6")
imdaa = normalize_dataset(raw_imdaa, dataset="imdaa")
name = find_variable(raw, "precipitation")
```

Without unit conversion:

```python
ds = normalize_dataset(raw, dataset="era5", si=False)
```

Presets: `era5`, `era5-land`, `imerg`, `gpm`, `wrf`, `imd`, `imdaa`, `cmip6`.

---

# Schema and station data

```python
from climatevar.io.schema import validate_dataset, describe_dataset

validate_dataset(ds, required=("precipitation", "latitude", "longitude", "time"))
print(describe_dataset(ds))
```

```python
from climatevar.io.stations import normalize_station_dataframe, station_dataframe_to_xarray

df = pd.read_csv("station_rainfall.csv")
df = normalize_station_dataframe(df)
station_ds = station_dataframe_to_xarray(df)
```

Explicit mapping:

```python
df = normalize_station_dataframe(df, columns={
    "time": "DATE_OBS",
    "precipitation": "RAIN_MM",
    "latitude": "LATITUDE_DD",
    "longitude": "LONGITUDE_DD",
})
```

---

# Complete monsoon + drought workflow

```python
import xarray as xr
from climatevar.io.normalize import normalize_dataset
from climatevar.precipitation import daily_amount, PRCPTOT, Rx1day
from climatevar.seasonal import monsoon_total
from climatevar.drought import spi
from climatevar.trends import mann_kendall, sens_slope

raw = xr.open_dataset("era5.nc")
ds = normalize_dataset(raw, dataset="era5")
daily = daily_amount(ds["precipitation"])

rx1 = Rx1day(daily)
prcptot = PRCPTOT(daily)
ism = monsoon_total(daily)

# Monthly totals for conventional SPI.
monthly = daily.resample(time="MS").sum()
spi3 = spi(monthly, scale=3)
spi12 = spi(monthly, scale=12)

mk = mann_kendall(ism)
slope = sens_slope(ism)
```

## Research reproducibility checklist

For publication-quality work, report dataset/version, domain, temporal aggregation, units, calendar, completeness rule, SPI/SPEI distribution and calibration period, PET method for SPEI, percentile baseline, event thresholds, spatial weighting, and trend/extreme-value assumptions.
