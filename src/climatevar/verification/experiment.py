"""Automated, reproducible precipitation product comparison experiments."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Sequence
import numpy as np
import pandas as pd
import xarray as xr
from climatevar.error_tagging.features import build_error_features
from .compare import evaluate_product, rank_products
from .datasets import ProductSpec

DEFAULT_SEASONS = {"MAM": (3, 4, 5), "JJA": (6, 7, 8), "JJAS": (6, 7, 8, 9), "SON": (9, 10, 11), "DJF": (12, 1, 2)}
DEFAULT_INTENSITIES = (("dry", 0.0, 1.0), ("light", 1.0, 10.0), ("moderate", 10.0, 20.0), ("heavy", 20.0, 50.0), ("very_heavy", 50.0, 100.0), ("extreme", 100.0, np.inf))
REGION_BOUNDS = {"odisha": (17.7, 22.6, 81.3, 87.5), "india": (6.0, 37.5, 68.0, 97.5)}

@dataclass(frozen=True)
class ExperimentConfig:
    reference: ProductSpec
    products: tuple[ProductSpec, ...]
    start: str | None = None
    end: str | None = None
    threshold: float = 1.0
    seasons: Mapping[str, Sequence[int]] = field(default_factory=lambda: DEFAULT_SEASONS)
    intensity_bins: Sequence[tuple[str, float, float]] = DEFAULT_INTENSITIES
    common_grid_method: str | None = None
    regrid_method: str | None = None
    region: str | tuple[float, float, float, float] | None = None
    uncertainty_resamples: int = 0
    block_length: int = 7
    random_state: int = 0
    output_name: str = "precipitation_comparison"

@dataclass
class ExperimentResult:
    overall: pd.DataFrame
    seasonal: pd.DataFrame
    intensity: pd.DataFrame
    ranking: pd.DataFrame
    spatial: xr.Dataset
    error_features: xr.Dataset
    uncertainty: pd.DataFrame = field(default_factory=pd.DataFrame)
    def tables(self) -> dict[str, pd.DataFrame]:
        return {"overall": self.overall, "seasonal": self.seasonal, "intensity": self.intensity, "ranking": self.ranking, "uncertainty": self.uncertainty}

def _period(data, start, end):
    return data if start is None and end is None else data.sel(time=slice(start, end))

def _subset_region(data, region):
    if region is None: return data
    bounds = REGION_BOUNDS.get(region.lower()) if isinstance(region, str) else region
    if bounds is None or len(bounds) != 4: raise ValueError("region must be 'odisha', 'india', or (lat_min, lat_max, lon_min, lon_max)")
    if data.lat.ndim != 1 or data.lon.ndim != 1: raise ValueError("Regional bbox selection requires 1-D lat/lon; regrid a curvilinear product first.")
    lat_min, lat_max, lon_min, lon_max = map(float, bounds)
    lat_slice = slice(lat_min, lat_max) if float(data.lat[0]) < float(data.lat[-1]) else slice(lat_max, lat_min)
    lon_slice = slice(lon_min, lon_max) if float(data.lon[0]) < float(data.lon[-1]) else slice(lon_max, lon_min)
    return data.sel(lat=lat_slice, lon=lon_slice)

def _regrid(reference, candidate, method):
    if method is None:
        if candidate.lat.ndim != 1 or candidate.lon.ndim != 1: raise ValueError("Curvilinear candidate grids require regrid_method='bilinear' or 'conservative'.")
        return candidate
    if method in {"linear", "nearest", "nearest_s2d"}:
        if candidate.lat.ndim != 1 or candidate.lon.ndim != 1: raise ValueError("xarray interpolation requires 1-D lat/lon; use xESMF bilinear/conservative for WRF.")
        return candidate.interp(lat=reference.lat, lon=reference.lon, method="nearest" if method != "linear" else "linear")
    if method in {"conservative", "bilinear", "patch"}:
        try: import xesmf as xe
        except ImportError as exc: raise ImportError("xesmf is required for conservative/bilinear/patch regridding; install climatevar[regrid].") from exc
        return xe.Regridder(candidate, reference, method, periodic=False, reuse_weights=False)(candidate)
    raise ValueError(f"Unsupported regrid method {method!r}.")

def _metric_row(reference, candidate, name, threshold, period, subset="all"):
    row = evaluate_product(reference, candidate, name=name, threshold=threshold); row.update({"period": period, "subset": subset}); return row

def _spatial_metrics(reference, candidate, name, threshold):
    valid = np.isfinite(reference) & np.isfinite(candidate); r, p = reference.where(valid), candidate.where(valid); err = p - r
    obs_event, pred_event = r >= threshold, p >= threshold; hit, miss, false_alarm = obs_event & pred_event, obs_event & ~pred_event, ~obs_event & pred_event
    hit_n, miss_n, fa_n = hit.sum("time"), miss.sum("time"), false_alarm.sum("time")
    ds = xr.Dataset({"bias": err.mean("time"), "mae": abs(err).mean("time"), "rmse": np.sqrt((err ** 2).mean("time")), "correlation": xr.corr(r, p, dim="time"), "pod": hit_n / (hit_n + miss_n), "far": fa_n / (hit_n + fa_n), "frequency_bias": (hit_n + fa_n) / (hit_n + miss_n), "valid_count": valid.sum("time")})
    return ds.expand_dims(dataset=[name])

def _safe_concat(datasets):
    return xr.concat(datasets, dim="dataset", join="outer", combine_attrs="override") if datasets else xr.Dataset()

def _conditional_rows(reference, candidate, name, threshold, bins, *, period):
    rows = []
    for label, low, high in bins:
        mask = (reference >= low) & (reference < high); r, p = reference.where(mask), candidate.where(mask)
        try:
            row = _metric_row(r, p, name, threshold, period, label); row["n_events"] = int(mask.sum().values)
        except (ValueError, ZeroDivisionError): row = {"dataset": name, "period": period, "subset": label, "n_events": int(mask.sum().values)}
        rows.append(row)
    return rows

def _bootstrap_indices(n, block, rng):
    block = max(1, min(int(block), n)); starts = rng.integers(0, n, size=int(np.ceil(n / block))); idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]; return idx

def _bootstrap_uncertainty(reference, candidate, name, threshold, resamples, block_length, rng):
    n = reference.sizes.get("time", 0)
    if n < 3 or resamples <= 0: return []
    rows = []
    metrics = {m: [] for m in ("bias", "mae", "rmse", "correlation", "pod", "far", "f1", "threat_score")}
    for _ in range(int(resamples)):
        idx = _bootstrap_indices(n, block_length, rng)
        r = reference.isel(time=idx); p = candidate.isel(time=idx)
        row = evaluate_product(r, p, name=name, threshold=threshold)
        for metric in metrics: metrics[metric].append(row[metric])
    for metric, values in metrics.items():
        q = np.nanpercentile(values, [2.5, 50, 97.5])
        rows.append({"dataset": name, "metric": metric, "lower_95": q[0], "median": q[1], "upper_95": q[2], "n_resamples": int(resamples), "block_length": int(block_length)})
    return rows

def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run the full comparative protocol from files/specifications."""
    reference = _subset_region(_period(config.reference.load(), config.start, config.end), config.region)
    products = {}
    for spec in config.products:
        data = _period(spec.load(), config.start, config.end)
        data = _regrid(reference, data, config.regrid_method or config.common_grid_method)
        products[spec.name] = _subset_region(data, config.region)
    overall_rows, seasonal_rows, intensity_rows, spatial_sets, error_sets, uncertainty_rows = [], [], [], [], [], []
    rng = np.random.default_rng(config.random_state)
    for name, product in products.items():
        overall_rows.append(_metric_row(reference, product, name, config.threshold, "all")); spatial_sets.append(_spatial_metrics(reference, product, name, config.threshold))
        for season, months in config.seasons.items():
            r = reference.where(reference.time.dt.month.isin(list(months)), drop=True); p = product.where(product.time.dt.month.isin(list(months)), drop=True)
            if r.sizes.get("time", 0): seasonal_rows.append(_metric_row(r, p, name, config.threshold, season))
        intensity_rows.extend(_conditional_rows(reference, product, name, config.threshold, config.intensity_bins, period="all"))
        uncertainty_rows.extend(_bootstrap_uncertainty(reference, product, name, config.threshold, config.uncertainty_resamples, config.block_length, rng))
        ef = build_error_features(reference, product)
        if "dataset" not in ef.dims: ef = ef.expand_dims(dataset=[name])
        error_sets.append(ef)
    overall, seasonal, intensity = pd.DataFrame(overall_rows), pd.DataFrame(seasonal_rows), pd.DataFrame(intensity_rows)
    ranking = rank_products(overall.to_dict("records")) if len(overall) else pd.DataFrame()
    uncertainty = pd.DataFrame(uncertainty_rows)
    return ExperimentResult(overall, seasonal, intensity, ranking, _safe_concat(spatial_sets), _safe_concat(error_sets), uncertainty)

def save_experiment(result: ExperimentResult, output_dir, *, prefix="precipitation_comparison"):
    """Save tables plus spatial and ML/error-tagging NetCDF outputs."""
    from pathlib import Path
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True); paths = {}
    for key, table in result.tables().items():
        path = out / f"{prefix}_{key}.csv"; table.to_csv(path, index=False); paths[key] = str(path)
    spatial_path = out / f"{prefix}_spatial.nc"; result.spatial.to_netcdf(spatial_path); paths["spatial"] = str(spatial_path)
    error_path = out / f"{prefix}_error_features.nc"; result.error_features.to_netcdf(error_path); paths["error_features"] = str(error_path)
    return paths
