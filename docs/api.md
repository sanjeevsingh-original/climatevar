# API cookbook

Practical examples for the public `climatevar` functions. All examples use xarray objects and can be adapted to ERA5, ERA5-Land, IMERG/GPM, IMDAA, CMIP6, WRF and station observations.

## Imports

```python
import numpy as np
import pandas as pd
import xarray as xr
from climatevar.climatology import climatology, anomaly
from climatevar.quality import valid_fraction, require_completeness, annual_valid_fraction
from climatevar.seasonal import seasonal_mean, monsoon_total
from climatevar.time import calendar_name, time_step_seconds, infer_frequency, expected_days_in_year, expected_days, year_complete_mask, season_year, drop_leap_day, validate_time
```

## Climatology and anomalies

### `climatology`

```python
monthly = climatology(rain, group="month")
daily = climatology(rain, group="dayofyear")
```

### `anomaly`

```python
reference = climatology(rain, group="month")
anom = anomaly(rain, reference, group="month")
```

## Quality control

### `valid_fraction`

```python
fraction = valid_fraction(rain, dim="time")
```

### `annual_valid_fraction`

```python
annual = annual_valid_fraction(rain, dim="time")
```

### `require_completeness`

```python
rain_qc = require_completeness(rain, min_fraction=0.9)
```

### `year_complete_mask`

```python
complete = year_complete_mask(rain, min_valid_fraction=0.9)
```

This is calendar-aware and is preferable when deciding whether annual indices are sufficiently complete.

## Time and calendars

### `calendar_name`

```python
print(calendar_name(rain))
```

### `time_step_seconds`

```python
dt = time_step_seconds(rain)
```

CF time bounds are preferred when available.

### `infer_frequency`

```python
print(infer_frequency(rain))
# hourly, 3-hourly, 6-hourly, daily, monthly, or irregular
```

### `expected_days_in_year`

```python
expected_days_in_year(2024)              # 366
aexpected = expected_days_in_year(2023)  # 365
expected_days_in_year(2024, "noleap")    # 365
expected_days_in_year(2001, "360_day")   # 360
```

### `expected_days`

```python
expected = expected_days(rain)
```

### `season_year`

```python
djf_year = season_year(rain, season="DJF")
jja_year = season_year(rain, season="JJA")
```

For DJF, December is assigned to the following January/February year.

### `drop_leap_day`

```python
rain_365 = drop_leap_day(rain)
```

### `validate_time`

```python
validate_time(rain)
validate_time(rain, require_sorted=False)
```

## Seasonal analysis

### `seasonal_mean`

```python
jja = seasonal_mean(rain, "JJA")
djf = seasonal_mean(rain, "DJF")
```

Supported seasons: `DJF`, `MAM`, `JJA`, `SON`.

### `monsoon_total`

```python
ism = monsoon_total(rain, start_month=6, end_month=9)
ond = monsoon_total(rain, start_month=10, end_month=12)
```

## Precipitation normalization

```python
from climatevar.precipitation import interval_amount_mm, normalize_precipitation, daily_amount
```

### `interval_amount_mm`

```python
amount = interval_amount_mm(precip)
```

Converts supported amounts/fluxes into interval rainfall depth in `mm`.

### `normalize_precipitation`

```python
hourly = normalize_precipitation(hourly_rain)       # mm/hr
three_hour = normalize_precipitation(rain_3hour)    # mm/3hr
daily_rate = normalize_precipitation(daily_rain)    # mm/day
```

Period accumulations such as monthly totals remain `mm`.

### `daily_amount`

```python
daily = daily_amount(hourly_rain)
daily = daily_amount(rain_3hour)
daily = daily_amount(rain_6hour)
```

Use this before daily precipitation indices when the source is subdaily.

## Precipitation extremes

```python
from climatevar.precipitation import Rx1day, Rx5day, PRCPTOT, R10mm, R20mm, R95p, R99p, CDD, CWD
```

```python
rx1 = Rx1day(daily)
rx5 = Rx5day(daily)
prcptot = PRCPTOT(daily)
r10 = R10mm(daily)
r20 = R20mm(daily)
cdd = CDD(daily)
cwd = CWD(daily)
```

For percentile indices, define a reproducible reference period:

```python
r95 = R95p(daily, baseline_start="1991-01-01", baseline_end="2020-12-31")
r99 = R99p(daily, baseline_start="1991-01-01", baseline_end="2020-12-31")
```

Or use an externally calculated threshold:

```python
r95 = R95p(daily, threshold=25.0)
```

## WRF precipitation

```python
from climatevar.precipitation import wrf_total_precipitation, wrf_precipitation_amount
wrf = xr.open_dataset("wrfout_d01_2024-07-01_00:00:00")
```

```python
total = wrf_total_precipitation(wrf)                         # RAINC + RAINNC
total_shallow = wrf_total_precipitation(wrf, include_shallow=True)
interval = wrf_precipitation_amount(wrf)
```

Custom names are supported:

```python
interval = wrf_precipitation_amount(
    wrf, convective="RAINC", nonconvective="RAINNC", shallow="RAINSH"
)
```

## Drought diagnostic

```python
from climatevar.drought import spi_like
spi3_baseline = spi_like(monthly_rain, scale=3)
spi12_baseline = spi_like(monthly_rain, scale=12)
```

`spi_like` is a standardized rolling precipitation diagnostic, **not formal fitted-distribution SPI**.

## Thermodynamics and moisture

```python
from climatevar.thermodynamics import potential_temperature, equivalent_potential_temperature, saturation_vapor_pressure, specific_humidity_from_rh

T = xr.DataArray([298.15, 303.15], dims="sample")
p = xr.DataArray([100000., 99000.], dims="sample")
rh = xr.DataArray([0.70, 0.80], dims="sample")

es = saturation_vapor_pressure(T)                 # Pa
q = specific_humidity_from_rh(T, p, rh)           # kg/kg
theta = potential_temperature(T, p)               # K
theta_e = equivalent_potential_temperature(T, p, q) # K
```

Temperature is K, pressure is Pa, and RH is fractional 0–1.

## Trends

```python
from climatevar.trends import mann_kendall, sens_slope

mk = mann_kendall(annual_rain)
print(mk["s"], mk["tau"], mk["pvalue"], mk["n"])
slope = sens_slope(annual_rain)
```

Sen's slope is expressed per observation step.

## GEV

```python
from climatevar.extremes import gev_fit, gev_return_level

fit = gev_fit(annual_maxima)
rl50 = gev_return_level(fit["shape"], fit["loc"], fit["scale"], 50)
```

The return period is in the same block units as the fitted sample. `gev_fit` uses SciPy's `genextreme` shape convention.

## Spatial analysis

```python
from climatevar.spatial import cosine_latitude_weights, area_weighted_mean

weights = cosine_latitude_weights(rain["latitude"])
regional = area_weighted_mean(rain, lat_dim="latitude", spatial_dims=("latitude", "longitude"))
```

Cosine-latitude weighting is intended for regular geographic grids. Use true cell-area weights for curvilinear/exact-area applications.

## Grid utilities

```python
from climatevar.grid import normalize_longitude, regrid

ds180 = normalize_longitude(ds, center=0.0)   # [-180, 180)
ds360 = normalize_longitude(ds, center=180.0) # [0, 360)

target = xr.Dataset({
    "latitude": np.arange(5, 30.1, 0.25),
    "longitude": np.arange(65, 100.1, 0.25),
})
out = regrid(ds["precipitation"], target, method="linear")
```

Methods are `linear` and `nearest`; curvilinear grids require xESMF.

## Verification metrics

```python
from climatevar.metrics import bias, mae, rmse, correlation, nse

bias_map = bias(obs, model, dim="time")
mae_map = mae(obs, model, dim="time")
rmse_map = rmse(obs, model, dim="time")
corr_map = correlation(obs, model, dim="time")
nse_map = nse(obs, model, dim="time")
```

Categorical precipitation verification:

```python
from climatevar.metrics import contingency_table, pod, far, f1_score, heidke_skill_score

counts = contingency_table(obs, model, threshold=1.0, dim="time")
pod_map = pod(obs, model, threshold=1.0, dim="time")
far_map = far(obs, model, threshold=1.0, dim="time")
f1_map = f1_score(obs, model, threshold=1.0, dim="time")
hss_map = heidke_skill_score(obs, model, threshold=1.0, dim="time")
```

## Dataset interoperability

```python
from climatevar.io.normalize import normalize_coords, standardize_variables, normalize_dataset, find_variable

coords = normalize_coords(raw, dataset="era5")
standard = standardize_variables(raw, variables={"precipitation": "tp", "temperature": "t2m"})
era5 = normalize_dataset(raw, dataset="era5")
cmip6 = normalize_dataset(raw_cmip6, dataset="cmip6")
name = find_variable(raw, "precipitation")
```

Set `si=False` to normalize names without unit conversion:

```python
ds = normalize_dataset(raw, dataset="era5", si=False)
```

Presets: `era5`, `era5-land`, `imerg`, `gpm`, `wrf`, `imd`, `imdaa`, `cmip6`.

## Schema and station data

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

Explicit station mapping:

```python
df = normalize_station_dataframe(df, columns={
    "time": "DATE_OBS", "precipitation": "RAIN_MM",
    "latitude": "LATITUDE_DD", "longitude": "LONGITUDE_DD",
})
```

## End-to-end monsoon/extreme-rainfall workflow

```python
import xarray as xr
from climatevar.io.normalize import normalize_dataset
from climatevar.precipitation import daily_amount, Rx1day, Rx5day, PRCPTOT
from climatevar.seasonal import monsoon_total
from climatevar.trends import mann_kendall, sens_slope

raw = xr.open_dataset("era5.nc")
ds = normalize_dataset(raw, dataset="era5")
daily = daily_amount(ds["precipitation"])
rx1 = Rx1day(daily)
rx5 = Rx5day(daily)
prcptot = PRCPTOT(daily)
ism = monsoon_total(daily)
mk = mann_kendall(ism)
slope = sens_slope(ism)
```

For publication, report dataset/version, domain, temporal aggregation, units, calendar, completeness rule, percentile baseline, thresholds, spatial weighting, and statistical assumptions.

## Scientific conventions

- Analysis-ready precipitation retains temporal resolution: `mm/hr`, `mm/3hr`, `mm/6hr`, `mm/day`.
- Period totals such as annual rainfall remain `mm`.
- Do not equate `skipna=True` with scientific completeness; apply an explicit missing-data rule.
- Do not silently mix calendars, grids, accumulation periods, or precipitation components.
- GEV parameters follow SciPy's sign convention.
