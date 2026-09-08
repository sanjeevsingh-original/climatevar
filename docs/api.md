# API cookbook

Practical examples for the public `climatevar` functions. All examples use xarray objects and can be adapted to ERA5, ERA5-Land, IMERG/GPM, IMDAA, CMIP6, WRF and station observations.

## Seasonal trend analysis

`seasonal_series` creates one value per climatological season and is calendar-aware. It supports monthly, daily, and regular sub-daily data.

```python
from climatevar.trends import seasonal_series, seasonal_mann_kendall, seasonal_sen_slope

# Daily precipitation: seasonal accumulation
jja_rain = seasonal_series(daily_rain, "JJA", aggregation="sum", min_valid_fraction=0.9)
jja_mk = seasonal_mann_kendall(daily_rain, "JJA", aggregation="sum", min_valid_fraction=0.9)
jja_slope = seasonal_sen_slope(daily_rain, "JJA", aggregation="sum", min_valid_fraction=0.9)

# Daily temperature: seasonal mean
jja_temperature = seasonal_series(daily_temperature, "JJA", aggregation="mean", min_valid_fraction=0.9)
```

`aggregation="auto"` detects precipitation-like metadata and uses a sum; other variables use a mean. For publication work, explicitly setting `aggregation` is recommended. Completeness is evaluated against the expected sample count for the detected frequency and CF calendar. Incomplete seasons are masked rather than silently included. Irregular time axes should be pre-aggregated before seasonal trend analysis.

For DJF, December is assigned to the following season-year.

## Trends

```python
from climatevar.trends import mann_kendall, modified_mann_kendall, sens_slope
mk = mann_kendall(annual_rain)
mmk = modified_mann_kendall(annual_rain)
slope = sens_slope(annual_rain)
```

The modified MK implementation uses a Yue-Wang-style effective-sample-size correction after Sen-slope detrending. Report the autocorrelation treatment explicitly.

## Spatial, regional and field significance

```python
from climatevar.trends import spatial_trend, regional_mean, regional_trend, field_significance
trend = spatial_trend(annual_rain, alpha=0.05)
region_series = regional_mean(annual_rain, lat_dim="latitude", lon_dim="longitude")
region = regional_trend(annual_rain, lat_dim="latitude", lon_dim="longitude")
field = field_significance(annual_rain, block_length=5, n_resamples=1000, random_state=42)
```

`spatial_trend` provides local modified-MK results, Sen slopes and FDR. `regional_trend` tests an area-weighted regional series. `field_significance` tests whether the observed number of locally significant cells is unusually large under a moving-block bootstrap of detrended residual fields with common time indices across the field.

Use regional, local/FDR and field-wide results as complementary evidence rather than substituting one for another.

## Precipitation normalization

```python
from climatevar.precipitation import interval_amount_mm, normalize_precipitation, daily_amount
amount = interval_amount_mm(precip)
hourly = normalize_precipitation(hourly_rain)       # mm/hr
three_hour = normalize_precipitation(rain_3hour)   # mm/3hr
daily_rate = normalize_precipitation(daily_rain)   # mm/day
daily = daily_amount(hourly_rain)
```

## Research reproducibility checklist

For publication-quality work, report dataset/version, domain, temporal aggregation, units, calendar, completeness rule, SPI/SPEI distribution and calibration period, PET method for SPEI, percentile baseline, event thresholds, spatial weighting, trend method, autocorrelation treatment, significance level, multiple-testing correction, and field-significance resampling design when used.
