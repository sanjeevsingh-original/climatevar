"""Optional publication-oriented plots for comparative precipitation verification."""
from __future__ import annotations


def plot_spatial_metric(spatial, metric, dataset, *, ax=None, levels=15, cmap="RdBu_r", title=None):
    """Plot one spatial verification field with matplotlib.

    This helper deliberately does not impose a map projection. Users can pass
    a Cartopy axis when Cartopy is installed, while the default remains usable
    with the package's lightweight ``plot`` dependency.
    """
    import matplotlib.pyplot as plt
    if metric not in spatial:
        raise KeyError(f"Unknown spatial metric {metric!r}")
    if "dataset" not in spatial.dims or dataset not in spatial.dataset.values:
        raise KeyError(f"Unknown dataset {dataset!r}")
    field = spatial[metric].sel(dataset=dataset)
    ax = ax or plt.gca()
    mesh = field.plot(ax=ax, levels=levels, cmap=cmap, add_colorbar=True)
    ax.set_title(title or f"{dataset}: {metric}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return ax, mesh


def plot_scorecard(scorecard, *, metrics=("bias", "mae", "rmse", "correlation", "pod", "far", "f1", "threat_score"), ax=None):
    """Plot a compact component-metric scorecard for manuscripts/reports."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    df = scorecard if isinstance(scorecard, pd.DataFrame) else pd.DataFrame(scorecard)
    missing = [m for m in metrics if m not in df]
    if missing: raise KeyError(f"Scorecard is missing metrics: {missing}")
    ax = ax or plt.gca()
    x = np.arange(len(df)); width = 0.8 / len(metrics)
    for i, metric in enumerate(metrics):
        values = df[metric].to_numpy(float)
        ax.bar(x + (i - (len(metrics)-1)/2) * width, values, width, label=metric)
    ax.set_xticks(x, df["dataset"].astype(str))
    ax.legend(ncol=2, fontsize="small")
    ax.set_ylabel("Metric value")
    ax.set_title("Precipitation verification scorecard")
    return ax
