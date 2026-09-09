# Comparative precipitation verification

`climatevar.verification` provides one reproducible scorecard for comparing precipitation products such as ERA5, GPM-IMERG, IMDAA/IMD, WRF and CMIP6 against an explicitly selected reference.

## Recommended protocol

1. Normalize every product to the same physical precipitation definition and accumulation period.
2. Restrict all products to the exact common time period.
3. Regrid to one declared analysis grid before grid-cell comparison. The baseline `common_grid()` uses interpolation; use conservative remapping when precipitation totals must be area-conserving.
4. Use an independent reference whenever possible. For Indian rainfall evaluation, IMD gridded gauge data are commonly used as a reference, while the literature shows that product skill varies substantially by region and rainfall intensity. citeturn0search0turn0search6
5. Report deterministic metrics (bias, MAE, RMSE, correlation) together with event metrics (POD, FAR, F1, HSS, TS, ETS and frequency bias).
6. Repeat the analysis by season, rainfall intensity, hydroclimatic region and extreme threshold rather than relying only on one domain-average score.
7. Preserve the complete component scorecard; a composite rank is a summary, not a substitute for the individual metrics.

## Example

```python
from climatevar.verification import compare_products, rank_products

rows = compare_products(
    imd_reference,
    {"ERA5": era5, "IMERG": imerg, "IMDAA": imdaa, "WRF": wrf, "CMIP6": cmip6},
    threshold=1.0,
    common_grid_method="linear",
)
ranking = rank_products(rows)
```

The event threshold is configurable. For publication work, define the threshold and units explicitly (for example, 1 mm/day for wet-day occurrence, or a percentile/extreme threshold).

## Error-tagging connection

After the scorecard, use `climatevar.error_tagging.build_error_features()` separately for each model/product. The resulting diagnostic target can classify underestimation, near-zero error and overestimation by intensity. For prospective ML error prediction, do **not** use observations or observation-derived errors as predictors.

## Scientific interpretation

Published Indian-region evaluations demonstrate that no single precipitation product is uniformly best across space and intensity: ERA5, IMDAA, IMERG and gauge-based products can differ in climatology, extremes and topographically complex regions. citeturn0search0turn0search2turn0search4 This is why `climatevar` retains the full metric vector and supports conditional evaluation rather than enforcing a universal winner.
