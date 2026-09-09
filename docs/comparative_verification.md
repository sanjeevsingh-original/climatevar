# Comparative precipitation verification

`climatevar.verification` provides a reproducible workflow for comparing ERA5, GPM-IMERG, IMDAA, WRF and CMIP6 against an explicitly selected reference.

## Odisha-ready experiment

For the standard Odisha analysis, use `odisha_experiment()`. It defaults to the common 2000–2018 period, matching the historical availability of the IMDAA archive described by IMD. IMD's 0.25° daily gridded rainfall is available as a NetCDF product in millimetres and is an appropriate declared reference candidate for an India-focused evaluation. citeturn0search0turn0search3

```python
from climatevar.verification import odisha_experiment, run_experiment, save_experiment

config = odisha_experiment(
    imd="data/IMD_rainfall.nc",
    era5="data/ERA5.nc",
    imerg="data/IMERG.nc",
    imdaa="data/IMDAA.nc",
    wrf="data/wrfout_d01.nc",
    cmip6="data/CMIP6_pr.nc",
    regrid_method="conservative",
    uncertainty_resamples=1000,
    block_length=7,
    random_state=0,
)

result = run_experiment(config)
save_experiment(result, "results/odisha_2000_2018")
```

The experiment uses an explicit Odisha bounding envelope before verification. For publication figures, replace this envelope with an authoritative Odisha administrative polygon or watershed mask if exact area boundaries are required.

## Outputs

The runner writes:

- `overall.csv` — domain-average deterministic and event scores;
- `seasonal.csv` — MAM, JJAS and ON scores;
- `intensity.csv` — dry/light/moderate/heavy/very-heavy/extreme conditional scores;
- `ranking.csv` — transparent composite summary using absolute bias rather than signed bias;
- `uncertainty.csv` — 95% moving-block-bootstrap intervals for key metrics;
- `spatial.nc` — grid-cell bias, MAE, RMSE, correlation, POD, FAR, frequency bias and valid counts;
- `error_features.nc` — diagnostic ML/error-tagging fields.

The uncertainty analysis uses a reproducible circular moving-block bootstrap over time. It accounts for short-range temporal dependence better than independently resampling individual daily observations. The resulting interval is an uncertainty estimate for the evaluation statistic, not a statement that the reference itself is error-free.

## Spatial maps

The optional `climatevar.verification.plots` module provides `plot_spatial_metric()` and `plot_scorecard()`. The spatial NetCDF output is deliberately kept independent of plotting, so the same fields can be rendered with Matplotlib, Cartopy or GIS software.

## Dataset adapters

`ProductSpec`/`load_precipitation()` provides presets for `era5`, `era5-land`, `imerg`, `gpm`, `imdaa`, `imd`, `cmip6` and `wrf`. Variable names can always be overridden because provider files differ between versions.

WRF precipitation is calculated from `RAINC + RAINNC` with restart-aware differencing. A WRF curvilinear grid should be regridded with xESMF before grid-cell verification.

## Regridding

`linear` and `nearest` are transparent interpolation baselines. `conservative`, `bilinear` and `patch` use optional xESMF. For precipitation totals, conservative remapping should be preferred when valid source/target cell geometry is available.

## Intensity conditioning

The default analysis classes are:

| Class | Rainfall amount |
|---|---:|
| dry | 0–<1 mm |
| light | 1–<10 mm |
| moderate | 10–<20 mm |
| heavy | 20–<50 mm |
| very heavy | 50–<100 mm |
| extreme | ≥100 mm |

These are configurable analysis defaults, not universal scientific thresholds. Percentile-based extreme definitions can be supplied separately for a publication analysis.

## Error tagging and ML

The error-feature dataset is a diagnostic target dataset: observations define the product error. For prospective ML prediction, observation-derived quantities such as observed rainfall, signed error and error class must not be used as predictors. Only information available at prediction time should enter the predictor matrix.

## Scientific reporting

Do not report the composite ranking as a universal winner. Product skill can vary by region, topography, season and rainfall intensity. The IMD data portal describes the 0.25° product as a daily gauge-based gridded rainfall dataset over India, while IMDAA is a 12-km regional reanalysis covering 1979–2018. citeturn0search0turn0search3

For a manuscript, retain the full component scorecard, report the common period and reference explicitly, document the regridding method, provide missing-data handling, and report uncertainty alongside the point estimates.
