"""Regional and field-wide significance diagnostics for climate trends."""
from __future__ import annotations

import numpy as np
import xarray as xr

from ..spatial.weights import area_weighted_mean
from .seasonal import fdr_mask
from .trend import _mk_stats, modified_mann_kendall, sens_slope


def regional_mean(
    data: xr.DataArray,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    weights: xr.DataArray | None = None,
) -> xr.DataArray:
    """Return an area-weighted regional mean time series.

    For regular geographic grids, cosine-latitude weights are used when
    ``weights`` is not supplied. Exact cell-area weights may be supplied for
    irregular or curvilinear grids.
    """
    if lat_dim not in data.dims or lon_dim not in data.dims:
        raise ValueError(f"data must contain {lat_dim!r} and {lon_dim!r} dimensions")
    if weights is None:
        return area_weighted_mean(data, lat_dim=lat_dim, spatial_dims=(lat_dim, lon_dim))
    if not set((lat_dim, lon_dim)).issubset(weights.dims):
        raise ValueError("weights must span the requested latitude and longitude dimensions")
    return data.weighted(weights).mean(dim=(lat_dim, lon_dim), skipna=True)


def regional_trend(
    data: xr.DataArray,
    time_dim: str = "time",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    alpha: float = 0.05,
    weights: xr.DataArray | None = None,
    max_lag: int | None = None,
) -> xr.Dataset:
    """Calculate an autocorrelation-aware trend for an area-weighted region."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    series = regional_mean(data, lat_dim=lat_dim, lon_dim=lon_dim, weights=weights)
    result = modified_mann_kendall(series, dim=time_dim, max_lag=max_lag)
    result["sen_slope"] = sens_slope(series, dim=time_dim)
    result["significant"] = result["pvalue"] < alpha
    result.attrs.update(
        {
            "alpha": alpha,
            "weighting": "user-supplied cell areas" if weights is not None else "cosine latitude",
        }
    )
    return result


def _block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a moving-block bootstrap index sequence of length ``n``."""
    if block_length < 1 or block_length > n:
        raise ValueError("block_length must be between 1 and the number of time steps")
    starts = rng.integers(0, n - block_length + 1, size=int(np.ceil(n / block_length)))
    return np.concatenate([np.arange(s, s + block_length) for s in starts])[:n]


def _mk_count(field: np.ndarray, alpha: float) -> int:
    """Count locally significant classical MK tests in a 2-D time/space array."""
    count = 0
    for j in range(field.shape[1]):
        _, _, p, _ = _mk_stats(field[:, j])
        if np.isfinite(p) and p < alpha:
            count += 1
    return count


def field_significance(
    data: xr.DataArray,
    dim: str = "time",
    alpha: float = 0.05,
    block_length: int = 5,
    n_resamples: int = 500,
    random_state: int | None = 0,
    detrend: bool = True,
) -> xr.Dataset:
    """Assess field-wide trend significance with a block-bootstrap null.

    The null distribution is generated from temporally block-resampled,
    detrended residual fields. A common set of time indices is applied to all
    grid cells, preserving the spatial covariance structure of each sampled
    field. This tests whether the observed number of locally significant MK
    trends is unusually large while retaining dependence that independent-cell
    tests ignore.

    This is a field-significance diagnostic, not a replacement for local
    autocorrelation correction or FDR. Local results are reported separately
    using the package's modified Mann-Kendall test and Benjamini-Hochberg FDR.
    """
    if dim not in data.dims:
        raise ValueError(f"Dimension {dim!r} is not present in the input data")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")

    other_dims = tuple(d for d in data.dims if d != dim)
    if not other_dims:
        raise ValueError("field_significance requires at least one spatial dimension")
    series = data.transpose(dim, *other_dims)
    values = np.asarray(series.values, dtype=float)
    n = values.shape[0]
    flat = values.reshape(n, -1)
    valid = np.isfinite(flat).all(axis=0)
    if not valid.any():
        raise ValueError("no complete grid cells are available for field significance")

    observed = np.full(flat.shape[1], np.nan)
    local = modified_mann_kendall(series, dim=dim)
    pvalue = local["pvalue"]
    observed_mask = np.isfinite(pvalue.values) & (pvalue.values < alpha)
    observed_count = int(observed_mask.sum())

    if detrend:
        slopes = np.asarray(sens_slope(series, dim=dim).values).reshape(-1)
        t = np.arange(n, dtype=float)[:, None]
        residual = flat - t * slopes[None, :]
    else:
        residual = flat.copy()

    rng = np.random.default_rng(random_state)
    null_counts = np.empty(n_resamples, dtype=int)
    work = np.empty((n, flat.shape[1]), dtype=float)
    for i in range(n_resamples):
        idx = _block_indices(n, block_length, rng)
        work[:] = residual[idx, :]
        # Classical MK is used for the bootstrap count because the null is
        # constructed by removing the observed trend; dependence is preserved
        # by the common block indices across the entire field.
        count = 0
        for j in np.flatnonzero(valid):
            _, _, p, _ = _mk_stats(work[:, j])
            if np.isfinite(p) and p < alpha:
                count += 1
        null_counts[i] = count

    field_p = (1.0 + np.count_nonzero(null_counts >= observed_count)) / (n_resamples + 1.0)
    critical_count = int(np.quantile(null_counts, 1.0 - alpha, method="higher"))

    result = xr.Dataset(
        {
            "pvalue": pvalue,
            "significant": pvalue < alpha,
            "significant_fdr": fdr_mask(pvalue, alpha=alpha),
            "observed_count": xr.DataArray(observed_count),
            "critical_count": xr.DataArray(critical_count),
            "field_pvalue": xr.DataArray(field_p),
            "null_counts": xr.DataArray(null_counts, dims=("resample",)),
        }
    )
    result.attrs.update(
        {
            "method": "moving-block bootstrap of detrended residual fields",
            "block_length": block_length,
            "n_resamples": n_resamples,
            "alpha": alpha,
            "random_state": random_state,
            "spatial_dependence": "preserved through common resampled time indices",
            "local_test": "modified Mann-Kendall",
            "bootstrap_count_test": "classical Mann-Kendall",
        }
    )
    return result


__all__ = ["regional_mean", "regional_trend", "field_significance"]
