# climatevar

**Research-oriented climate and atmospheric science analysis in Python.**

`climatevar` provides transparent, xarray-first building blocks for climate variability, precipitation extremes, statistical trends, extreme-value analysis, atmospheric thermodynamics, spatial diagnostics, and model verification.

## Scientific workflow

```text
Raw observations / reanalysis / model output
                |
                v
        xarray Dataset/DataArray
                |
      +---------+----------+
      |                    |
 climatology          quality control
      |                    |
 anomalies          derived variables
      |                    |
      +---------+----------+
                |
      indices / trends / extremes
                |
        verification & plots
                |
       reproducible results
```

## API areas

- `climatevar.climatology` — climatologies and anomalies
- `climatevar.precipitation` — precipitation extremes
- `climatevar.trends` — Mann-Kendall and Sen slope
- `climatevar.extremes` — GEV fitting and return levels
- `climatevar.thermodynamics` — potential and equivalent potential temperature
- `climatevar.spatial` — latitude weighting and spatial means
- `climatevar.metrics` — continuous and precipitation-event verification
- `climatevar.io` — explicit xarray dataset loading
- `climatevar.wrf` — WRF nesting utilities
- `climatevar.visualization` — lightweight plotting helpers

## Research-use note

The package is pre-alpha. Scientific definitions are deliberately exposed in function documentation rather than hidden behind automatic preprocessing. Before publication, users should verify index definitions, missing-data rules, calendar handling, baseline periods, units, and statistical assumptions for their specific study.
