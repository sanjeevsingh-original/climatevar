# Dataset interoperability

`climatevar` separates **dataset-specific naming** from **physical meaning**. The normalization layer recognizes common ERA5, ERA5-Land, IMD, IMDAA, IMERG/GPM, WRF and CMIP6 conventions and maps them to canonical names.

## Canonical fields

Common fields include `latitude`, `longitude`, `time`, `precipitation`, `temperature`, `surface_pressure`, `relative_humidity`, `specific_humidity`, `u_wind`, `v_wind` and `geopotential`.

## Units

`normalize_dataset(..., si=True)` is the default. Recognized variables use SI units **except precipitation amount, which is intentionally represented in millimetres (`mm`)** because precipitation depth and rainfall extremes are conventionally analysed and reported in mm.

| Quantity | climatevar unit |
|---|---|
| temperature | K |
| pressure | Pa |
| precipitation amount | **mm** |
| precipitation flux | kg m-2 s-1 |
| relative humidity | 1 |
| specific humidity | kg kg-1 |
| wind components | m s-1 |
| geopotential | m2 s-2 |

The conversion is explicit and provenance is stored in variable attributes. A variable with missing units is not guessed or changed.

### Important precipitation rule

A precipitation **amount** and precipitation **flux/rate** are different physical quantities. Precipitation amounts are always normalized to **mm**. For example, ERA5 accumulated precipitation represented in metres is converted to mm. CMIP-style precipitation is often a flux, so it remains a flux in `kg m-2 s-1` rather than being incorrectly treated as a depth. A rate/flux is never silently converted to mm without an explicit temporal integration step.

For rainfall analysis, the canonical `precipitation` variable therefore has one of two explicit states:

- `units="mm"` with `climatevar:quantity="precipitation_amount"`
- `units="kg m-2 s-1"` with `climatevar:quantity="precipitation_flux"`

This distinction prevents physically incorrect comparisons between accumulated rainfall and precipitation rates.

## Calendars and frequency

Use `calendar_name()`, `infer_frequency()` and `validate_time()` to inspect CF calendars, time-step length, duplicate timestamps and ordering. Xarray supports non-standard climate calendars through `cftime`; `climatevar` does not coerce a no-leap or 360-day calendar into Gregorian time without an explicit user decision.

## Regridding

Rectilinear latitude/longitude grids use xarray interpolation:

```python
from climatevar.io import regrid

result = regrid(source, target, method="linear")
```

Native WRF/other curvilinear grids require the optional `xesmf` dependency:

```bash
pip install "climatevar[regrid]"
```

Regridding is never performed automatically during dataset normalization because it changes the spatial representation of the data and should be a documented methodological choice.

## Scientific basis

The design follows the CF principle of identifying physical quantities with metadata such as `standard_name` and units rather than relying solely on producer-specific variable names. CF also defines canonical units and supports non-standard climate calendars; xarray provides the corresponding calendar-aware data model.
