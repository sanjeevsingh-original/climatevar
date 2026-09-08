"""Independent reference checks for the public trend diagnostics."""

import numpy as np
import xarray as xr
from scipy.stats import kendalltau, norm, theilslopes

from climatevar.trends import field_significance, mann_kendall, modified_mann_kendall, regional_mean, sens_slope


def _reference_mk(values):
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    tau, p = kendalltau(np.arange(y.size), y)
    s = sum(np.sign(y[j] - y[i]) for i in range(y.size) for j in range(i + 1, y.size))
    return float(s), float(tau), float(p)


def _reference_modified_mk(values, max_lag=None):
    """Independent implementation of climatevar's documented ESS correction."""
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    s, tau, _ = _reference_mk(y)
    slope = float(theilslopes(y, np.arange(n)).slope)
    residual = y - slope * np.arange(n)
    z = residual - residual.mean()
    denom = np.sum(z * z)
    max_lag = max_lag or min(n - 1, int(np.sqrt(n) * 3))
    r = np.array(
        [np.sum(z[k:] * z[:-k]) / denom for k in range(1, min(max_lag, n - 2) + 1)]
    ) if denom else np.zeros(0)
    factor = 1.0 + (2.0 / n) * np.sum((n - np.arange(1, len(r) + 1)) * r)
    factor = max(factor, 1e-6)
    n_eff = float(np.clip(n / factor, 1.0, n))
    _, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0 * n / n_eff
    if s > 0:
        z_mk = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z_mk = (s + 1.0) / np.sqrt(var_s)
    else:
        z_mk = 0.0
    return s, tau, float(2 * norm.sf(abs(z_mk))), n_eff


def test_classical_mk_matches_scipy_reference():
    y = np.array([3.0, 1.0, 2.0, 5.0, 4.0, 7.0, 6.0])
    expected = _reference_mk(y)
    result = mann_kendall(xr.DataArray(y, dims="time"))
    np.testing.assert_allclose(result.s.item(), expected[0])
    np.testing.assert_allclose(result.tau.item(), expected[1])
    np.testing.assert_allclose(result.pvalue.item(), expected[2])


def test_sen_slope_matches_scipy_theil_slope():
    y = np.array([1.2, 2.8, 4.1, 5.0, 7.4, 8.9])
    expected = theilslopes(y, np.arange(y.size)).slope
    actual = sens_slope(xr.DataArray(y, dims="time")).item()
    np.testing.assert_allclose(actual, expected)


def test_modified_mk_matches_independent_effective_sample_size_reference():
    rng = np.random.default_rng(7)
    innovations = rng.normal(size=120)
    y = np.empty(120)
    y[0] = innovations[0]
    for i in range(1, y.size):
        y[i] = 0.65 * y[i - 1] + innovations[i]
    y += 0.03 * np.arange(y.size)

    expected = _reference_modified_mk(y, max_lag=12)
    result = modified_mann_kendall(xr.DataArray(y, dims="time"), max_lag=12)
    np.testing.assert_allclose(result.s.item(), expected[0])
    np.testing.assert_allclose(result.tau.item(), expected[1])
    np.testing.assert_allclose(result.pvalue.item(), expected[2])
    np.testing.assert_allclose(result.n_eff.item(), expected[3])


def test_regional_mean_matches_exact_weighted_average():
    values = xr.DataArray(
        [[1.0, 3.0], [5.0, 7.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [0.0, 60.0], "longitude": [80.0, 81.0]},
    )
    weights = xr.DataArray(
        [[1.0, 1.0], [3.0, 3.0]],
        dims=("latitude", "longitude"),
        coords=values.coords,
    )
    expected = float((values * weights).sum() / weights.sum())
    actual = regional_mean(values, weights=weights).item()
    np.testing.assert_allclose(actual, expected)


def test_field_significance_is_reproducible_and_excludes_incomplete_cells():
    rng = np.random.default_rng(12)
    t = np.arange(30, dtype=float)
    base = rng.normal(size=(30, 2, 2))
    base[:, 1, 1] = np.nan
    base[:, 0, 0] += 0.2 * t
    data = xr.DataArray(
        base,
        dims=("time", "latitude", "longitude"),
        coords={"time": t, "latitude": [10.0, 11.0], "longitude": [80.0, 81.0]},
    )
    a = field_significance(data, block_length=4, n_resamples=80, random_state=42)
    b = field_significance(data, block_length=4, n_resamples=80, random_state=42)
    assert a.n_cells.item() == 3
    assert a.observed_count.item() == b.observed_count.item()
    np.testing.assert_array_equal(a.null_counts.values, b.null_counts.values)
    np.testing.assert_allclose(a.field_pvalue.item(), b.field_pvalue.item())
