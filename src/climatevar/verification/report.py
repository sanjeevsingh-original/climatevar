"""Publication-oriented report and figure generation for precipitation verification."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


def _mpl():
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    return str(path)


def spatial_metric_panels(spatial, metric: str, *, output=None, datasets=None, ncols=2, levels=15, cmap="RdBu_r", title=None):
    """Create a manuscript-ready panel of one spatial metric for all products.

    Cartopy is optional. If a Cartopy GeoAxes is supplied by the caller, the
    underlying xarray plotting call can be used; otherwise this remains a
    projection-free latitude/longitude figure.
    """
    plt = _mpl()
    if metric not in spatial:
        raise KeyError(f"Unknown spatial metric {metric!r}")
    available = [str(x) for x in spatial.dataset.values]
    names = available if datasets is None else [str(x) for x in datasets]
    missing = [x for x in names if x not in available]
    if missing:
        raise KeyError(f"Unknown datasets: {missing}")
    ncols = max(1, min(int(ncols), len(names)))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), squeeze=False)
    finite = np.asarray(spatial[metric].sel(dataset=names).values, dtype=float)
    if metric not in {"pod", "far"}:
        vmax = np.nanmax(np.abs(finite)) if np.isfinite(finite).any() else 1.0
        vmin = -vmax
    else:
        vmin, vmax = 0.0, 1.0
    mappable = None
    for i, name in enumerate(names):
        ax = axes.flat[i]
        field = spatial[metric].sel(dataset=name)
        mappable = field.plot(ax=ax, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax, add_colorbar=False)
        ax.set_title(str(name))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    fig.colorbar(mappable, ax=[a for a in axes.flat[:len(names)]], shrink=0.86, label=metric)
    fig.suptitle(title or f"Odisha precipitation verification: {metric.upper()}", y=1.02)
    fig.tight_layout()
    return fig, _save(fig, output) if output else None


def seasonal_heatmap(seasonal: pd.DataFrame, metric="rmse", *, output=None, title=None, annotate=True):
    """Create a product × season heatmap from the seasonal scorecard."""
    plt = _mpl()
    if metric not in seasonal.columns:
        raise KeyError(f"Seasonal scorecard is missing {metric!r}")
    table = seasonal.pivot(index="dataset", columns="period", values=metric)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * table.shape[1]), max(3.5, 0.65 * table.shape[0] + 1.5)))
    image = ax.imshow(table.to_numpy(float), aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("Season")
    ax.set_ylabel("Dataset")
    ax.set_title(title or f"Seasonal {metric.upper()}")
    if annotate:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                value = table.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label=metric)
    fig.tight_layout()
    return fig, _save(fig, output) if output else None


def intensity_distribution(intensity: pd.DataFrame, metric="bias", *, output=None, title=None):
    """Plot error distributions by observed-rainfall intensity class."""
    plt = _mpl()
    if metric not in intensity.columns:
        raise KeyError(f"Intensity scorecard is missing {metric!r}")
    groups = list(intensity.groupby("subset", sort=False))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions = []
    values = []
    labels = []
    for i, (label, group) in enumerate(groups, start=1):
        values.append(group[metric].dropna().to_numpy(float))
        positions.append(i)
        labels.append(label.replace("_", " "))
    if values:
        ax.boxplot(values, positions=positions, widths=0.65, showfliers=False)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_xlabel("Observed precipitation intensity")
    ax.set_title(title or f"Error by rainfall intensity: {metric}")
    fig.tight_layout()
    return fig, _save(fig, output) if output else None


def uncertainty_plot(uncertainty: pd.DataFrame, *, output=None, metrics: Sequence[str] | None = None, title=None):
    """Plot bootstrap median and 95% interval for each product and metric."""
    plt = _mpl()
    if uncertainty.empty:
        raise ValueError("Uncertainty table is empty; run the experiment with uncertainty_resamples > 0.")
    metrics = list(metrics or uncertainty["metric"].drop_duplicates())
    data = uncertainty[uncertainty.metric.isin(metrics)].copy()
    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, max(3.0, 2.8 * len(metrics))), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        d = data[data.metric == metric]
        x = np.arange(len(d))
        ax.errorbar(x, d["median"], yerr=[d["median"] - d["lower_95"], d["upper_95"] - d["median"]], fmt="o", capsize=4)
        ax.set_xticks(x, d["dataset"].astype(str), rotation=25, ha="right")
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title or "Bootstrap uncertainty (95% intervals)", y=1.0)
    fig.tight_layout()
    return fig, _save(fig, output) if output else None


def ranking_heatmap(ranking: pd.DataFrame, *, output=None, title="Comparative precipitation-product ranking"):
    """Plot the normalized ranking scorecard when available."""
    plt = _mpl()
    if ranking.empty:
        raise ValueError("Ranking table is empty")
    numeric = ranking.select_dtypes(include=[np.number]).copy()
    for col in ["rank", "composite_score"]:
        if col in numeric:
            numeric = numeric.drop(columns=col)
    if numeric.empty:
        raise ValueError("Ranking table contains no numeric metrics")
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * numeric.shape[1]), max(3.5, 0.65 * len(ranking) + 1.5)))
    image = ax.imshow(numeric.to_numpy(float), aspect="auto")
    ax.set_xticks(range(numeric.shape[1]), numeric.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(ranking)), ranking.get("dataset", ranking.index).astype(str))
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Metric value")
    fig.tight_layout()
    return fig, _save(fig, output) if output else None


def manuscript_summary(result, *, output=None, reference_name=None):
    """Build a compact manuscript-ready CSV table from an ExperimentResult."""
    overall = result.overall.copy()
    cols = [c for c in ["dataset", "bias", "mae", "rmse", "correlation", "pod", "far", "f1", "threat_score", "ets", "frequency_bias"] if c in overall.columns]
    table = overall[cols].copy()
    if reference_name is not None:
        table.insert(0, "reference", reference_name)
    if "ranking" in result.__dict__ and not result.ranking.empty and "rank" in result.ranking.columns:
        ranks = result.ranking[[c for c in ["dataset", "rank", "composite_score"] if c in result.ranking.columns]]
        table = table.merge(ranks, on="dataset", how="left")
    if output:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True); table.to_csv(path, index=False)
    return table


def generate_publication_report(result, output_dir, *, prefix="odisha_verification", spatial_metrics=("bias", "mae", "rmse", "pod", "far"), uncertainty_metrics=("bias", "mae", "rmse", "correlation", "pod", "far")):
    """Generate the complete figure/table bundle from an ExperimentResult."""
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for metric in spatial_metrics:
        if metric in result.spatial:
            _, path = spatial_metric_panels(result.spatial, metric, output=out / f"{prefix}_spatial_{metric}.png")
            outputs[f"spatial_{metric}"] = path
    if not result.seasonal.empty:
        _, path = seasonal_heatmap(result.seasonal, "rmse", output=out / f"{prefix}_seasonal_rmse.png")
        outputs["seasonal_rmse"] = path
        if "bias" in result.seasonal:
            _, path = seasonal_heatmap(result.seasonal, "bias", output=out / f"{prefix}_seasonal_bias.png")
            outputs["seasonal_bias"] = path
    if not result.intensity.empty:
        _, path = intensity_distribution(result.intensity, "bias", output=out / f"{prefix}_intensity_bias.png")
        outputs["intensity_bias"] = path
    if not result.uncertainty.empty:
        _, path = uncertainty_plot(result.uncertainty, output=out / f"{prefix}_uncertainty.png", metrics=uncertainty_metrics)
        outputs["uncertainty"] = path
    if not result.ranking.empty:
        _, path = ranking_heatmap(result.ranking, output=out / f"{prefix}_ranking.png")
        outputs["ranking"] = path
    table = manuscript_summary(result, output=out / f"{prefix}_summary.csv")
    outputs["summary"] = str(out / f"{prefix}_summary.csv")
    return outputs, table
