# Comparative precipitation verification

`climatevar.verification` provides one reproducible workflow for comparing precipitation products such as ERA5, GPM-IMERG, IMDAA/IMD, WRF and CMIP6 against an explicitly selected reference.

## Recommended protocol

1. Normalize every product to the same physical precipitation definition and accumulation period.
2. Restrict all products to the exact common time period.
3. Regrid to one declared analysis grid before grid-cell comparison. Linear/nearest interpolation is a transparent baseline; use conservative remapping when precipitation totals must be area-conserving.
4. Use an independent reference whenever possible. IMD gridded gauge data are commonly used for Indian rainfall evaluation, but published studies show that skill varies substantially by region and rainfall intensity.
5. Report deterministic metrics (bias, absolute bias, MAE, RMSE, correlation) together with event metrics (POD, FAR, F1, HSS, TS, ETS and frequency bias).
6. Repeat the analysis by season, rainfall intensity, region and extreme threshold rather than relying only on one domain-average score.
7. Preserve the complete component scorecard; a composite rank is a summary, not a substitute for the individual metrics.

## Dataset adapters

The new `ProductSpec`/`load_precipitation()` interface provides family presets while still allowing an explicit variable override:

```python
from climatevar.verification import ProductSpec

reference = ProductSpec("IMD", "imd", "imd_daily.nc", variable="rainfall")
products = (
    ProductSpec("ERA5", "era5", "era5.nc", variable="tp"),
    ProductSpec("IMERG", "imerg", "imerg.nc", variable="precipitationCal"),
    ProductSpec("IMDAA", "imdaa", "imdaa.nc", variable="pr"),
    ProductSpec("WRF", "wrf", "wrfout_d01.nc"),
    ProductSpec("CMIP6", "cmip6", "historical_pr.nc", variable="pr"),
)
```

Supported presets include `era5`, `era5-land`, `imerg`, `gpm`, `imdaa`, `imd`, `cmip6` and `wrf`. WRF precipitation is calculated from `RAINC + RAINNC`; the dedicated WRF path avoids treating one accumulated component as total precipitation.

Because provider files differ between versions, the adapter does not pretend that one variable name is universal. Pass `variable=` whenever the preset cannot identify the field automatically.

## Automated experiment

```python
from climatevar.verification import ExperimentConfig, run_experiment, save_experiment

config = ExperimentConfig(
    reference=reference,
    products=products,
    start="2000-06-01",
    end="2020-09-30",
    threshold=1.0,
    seasons={"JJAS": (6, 7, 8, 9), "DJF": (12, 1, 2)},
    regrid_method="linear",
)

result = run_experiment(config)
paths = save_experiment(result, "results/odisha_comparison")
```

The runner produces:

- `overall`: deterministic and event-based scorecard for every product;
- `seasonal`: the same metrics for every declared season;
- `intensity`: conditional error/event scores for dry, light, moderate, heavy, very-heavy and extreme rainfall;
- `ranking`: transparent weighted composite ranking based on absolute bias, MAE, RMSE, correlation and event metrics;
- `spatial`: NetCDF fields of bias, MAE, RMSE, correlation, POD, FAR, frequency bias and valid count;
- `error_features`: ML/error-tagging-ready diagnostic fields containing the observed target and model error information.

## Regridding

`regrid_method="linear"` and `"nearest"` use xarray interpolation and are intended as transparent baselines. `"conservative"`, `"bilinear"` and `"patch"` use optional xESMF. For precipitation accumulation/totals, conservative remapping should be preferred when the source grid provides appropriate cell geometry.

## Intensity conditioning

The default classes are configurable and use rainfall amount in the normalized analysis units:

| Class | Range |
|---|---:|
| dry | 0–<1 mm |
| light | 1–<10 mm |
| moderate | 10–<20 mm |
| heavy | 20–<50 mm |
| very heavy | 50–<100 mm |
| extreme | ≥100 mm |

These are analysis defaults, not universal scientific thresholds. For an India/Odisha publication, define the thresholds or percentile baseline in the methods section and keep them fixed across products.

## Error tagging and ML

`error_features` is deliberately a diagnostic target dataset: observations define whether a product underestimates or overestimates rainfall. For prospective ML error prediction, observation-derived quantities such as observed rainfall, signed error and error class must not be used as predictors. Use only predictors that would be available at prediction time.

## Scientific interpretation

Published Indian-region evaluations demonstrate that no single precipitation product is uniformly best across space and intensity. ERA5, IMDAA, IMERG and gauge-based products can differ in climatology, event detection and extreme-rainfall representation, particularly over complex terrain. This is why `climatevar` retains the full metric vector and supports conditional evaluation rather than enforcing a universal winner.

A composite rank should therefore be reported alongside the underlying metrics, not instead of them. The ranking implementation uses **absolute bias**, not signed bias, because both wet and dry bias represent error. Its min-max normalization is sample-dependent and should be described as a summary score rather than a universal physical skill measure.
