# Common climate data model

`climatevar` uses a canonical xarray schema so analysis functions do not need dataset-specific variable names.

## Canonical fields

| Physical quantity | climatevar name | Common examples |
|---|---|---|
| latitude | `latitude` | `lat`, `Latitude`, `XLAT` |
| longitude | `longitude` | `lon`, `Longitude`, `XLONG` |
| time | `time` | `time`, `valid_time`, `Times` |
| precipitation | `precipitation` | ERA5 `tp`, IMERG `precipitation`, IMD `rainfall`, WRF `RAINNC` |
| temperature | `temperature` | ERA5 `t2m`, IMDAA `TMP_2m`, WRF `T2` |
| surface pressure | `surface_pressure` | ERA5 `sp`, IMDAA `PRES_sfc`, WRF `PSFC` |
| relative humidity | `relative_humidity` | `rh`, `RH2`, IMDAA `RH_2m` |
| specific humidity | `specific_humidity` | `q`, `Q2`, IMDAA `SPFH_2m` |
| u wind | `u_wind` | `u10`, `U10`, IMDAA `UGRD_10m` |
| v wind | `v_wind` | `v10`, `V10`, IMDAA `VGRD_10m` |

## Example

```python
import xarray as xr
from climatevar.io import normalize_dataset

raw = xr.open_dataset("dataset.nc")
ds = normalize_dataset(raw, dataset="era5")
rain = ds["precipitation"]
```

Normalization changes **names**, not physical meaning. Unit conversion and temporal interpretation are explicit operations. This is important for precipitation because reanalysis/model products may store accumulated depth or rates, while observations may represent point measurements. CF conventions likewise rely on metadata such as `standard_name` and `units` to make quantities comparable. citeturn0search1turn0search0

## Dataset-specific notes

- **ERA5 / ERA5-Land:** total precipitation is commonly `tp` and represents accumulated precipitation depth; ERA5 documentation describes its units as metres. Do not treat it as a rate without converting according to the accumulation period. citeturn0search3turn0search4
- **WRF:** precipitation may be split between convective and non-convective accumulations. Users should explicitly choose or combine fields rather than having the library guess.
- **Station data:** use `climatevar.io.stations.normalize_station_dataframe()` for CSV/tabular observations.
- **CMIP-style data:** CF metadata should be preferred over filename-based guessing. `standard_name`, `units`, coordinate metadata and calendar information are more reliable than variable names alone. citeturn0search1turn0search2

## Scientific safety rule

`climatevar` will normalize aliases, but it will not silently perform scientifically consequential transformations such as regridding, unit conversion, accumulated-to-rate conversion, calendar conversion, or WRF rainfall-component summation.
