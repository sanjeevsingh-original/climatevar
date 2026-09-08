# Robust trend analysis

`climatevar` provides classical Mann-Kendall (MK), Sen's slope, an autocorrelation-aware modified MK, and trend-free pre-whitening (TFPW).

## Classical MK

```python
from climatevar.trends import mann_kendall, sens_slope

mk = mann_kendall(annual_rain)
slope = sens_slope(annual_rain)
```

Classical MK assumes serial independence. Positive autocorrelation can inflate the apparent significance of a trend, so it should not automatically be used for persistent climate and hydrological series. citeturn0search0turn0search5

## Modified Mann-Kendall

```python
from climatevar.trends import modified_mann_kendall

mmk = modified_mann_kendall(annual_rain)
print(mmk["tau"])
print(mmk["pvalue"])
print(mmk["n_eff"])
```

The implementation uses a Yue-Wang-style effective-sample-size correction after removing the Sen-slope component before estimating persistence. The original ESS approach is designed to account for serial correlation, although estimating persistence in a trending series requires care. citeturn0search5turn0search10

## Trend-free pre-whitening

```python
from climatevar.trends import trend_free_prewhitening, mann_kendall

prewhitened = trend_free_prewhitening(annual_rain)
mk_pw = mann_kendall(prewhitened)
```

TFPW removes the estimated monotonic component, estimates lag-1 persistence, removes the AR(1) component, and restores the trend component. Pre-whitening methods can alter trend information, so results should be reported with the exact method used. citeturn0search10turn0search6

## Recommended publication workflow

For annual or seasonal rainfall/drought indices:

1. Verify temporal completeness and remove/flag invalid years.
2. Report classical MK + Sen slope as a baseline.
3. Diagnose serial dependence.
4. Use modified MK or TFPW when persistence is material.
5. Report both slope magnitude and significance rather than p-value alone.
6. For monthly seasonal data, use a seasonal MK framework rather than treating all months as one homogeneous series.

For long-range persistent hydrological/climate series, more advanced scaling-based methods may be appropriate; these should not be silently substituted for the effective-sample-size method. citeturn0search3turn0search7
