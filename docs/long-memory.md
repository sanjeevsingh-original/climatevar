# Persistence and long-memory diagnostics

`climatevar.trends.dfa_hurst` provides detrended fluctuation analysis (DFA) as a diagnostic of temporal scaling and persistence.

```python
from climatevar.trends import dfa_hurst

result = dfa_hurst(annual_rain, min_scale=8, max_scale=80, n_scales=16)
print(result.dfa_exponent)
print(result.r2)
```

The `dfa_exponent` is the fitted slope of log fluctuation versus log scale. Values around 0.5 are broadly compatible with short-memory/no-correlation scaling; values above 0.5 indicate persistent scaling that should be investigated further.

A DFA exponent above 0.5 is **not, by itself, proof of long-range dependence**. Trends, nonstationarity, finite-sample effects, regime changes and short-memory processes can also produce apparent persistence. For publication, compare the exponent across detrending orders and scale ranges and complement it with autocorrelation diagnostics or an explicit long-memory model when the scientific claim requires one.

The result reports `r2` for the log-log scaling fit and `n` for the number of finite observations used.
