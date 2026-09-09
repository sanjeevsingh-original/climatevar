"""Publication-oriented plotting and automated POT diagnostic reports."""
from __future__ import annotations


def plot_threshold_stability(diagnostics, ax=None):
    """Plot GPD shape and scale against candidate thresholds."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(diagnostics.threshold, diagnostics.shape, marker="o", label="Shape (xi)")
    ax2.plot(diagnostics.threshold, diagnostics.scale, marker="s", label="Scale (sigma)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("GPD shape")
    ax2.set_ylabel("GPD scale")
    ax.set_title("POT threshold stability")
    ax.grid(True, alpha=0.25)
    return ax


def plot_mean_excess(diagnostics, ax=None):
    """Plot the empirical mean-excess function versus threshold."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(diagnostics.threshold, diagnostics.mean_excess, marker="o")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean excess")
    ax.set_title("Mean-excess plot")
    ax.grid(True, alpha=0.25)
    return ax


def plot_gpd_qq(gof, ax=None):
    """Plot a GPD quantile-quantile diagnostic."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(gof.qq_theoretical, gof.qq_observed, s=18)
    lo = min(float(gof.qq_theoretical.min()), float(gof.qq_observed.min()))
    hi = max(float(gof.qq_theoretical.max()), float(gof.qq_observed.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Theoretical GPD quantile")
    ax.set_ylabel("Observed excess")
    ax.set_title("GPD Q-Q diagnostic")
    ax.grid(True, alpha=0.25)
    return ax


def pot_diagnostic_report(diagnostics, gof=None, uncertainty=None, model=None):
    """Create a self-contained HTML summary suitable for supplementary material."""
    def rows(ds):
        if ds is None:
            return ""
        out = []
        for key in ds.data_vars:
            value = ds[key].values
            if getattr(value, "size", 1) == 1:
                out.append(f"<tr><td>{key}</td><td>{float(value):.6g}</td></tr>")
        return "".join(out)
    html = ["<!doctype html><html><head><meta charset='utf-8'><title>climatevar POT diagnostic report</title>",
            "<style>body{font-family:system-ui;max-width:1000px;margin:40px auto;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:6px 10px}</style></head><body>",
            "<h1>POT diagnostic report</h1>",
            "<p>This report records threshold diagnostics, GPD goodness-of-fit diagnostics, uncertainty, and model-comparison results supplied by the analysis.</p>",
            "<h2>Threshold diagnostics</h2>",
            "<table><tr><th>Threshold</th><th>Exceedances</th><th>Rate</th><th>Shape</th><th>Scale</th><th>Mean excess</th></tr>"]
    for i in range(diagnostics.threshold.size):
        html.append("<tr>" + "".join(f"<td>{float(diagnostics[k].values[i]):.6g}</td>" for k in ["threshold", "n_exceedances", "exceedance_rate", "shape", "scale", "mean_excess"]) + "</tr>")
    html.append("</table>")
    for title, ds in [("Goodness of fit", gof), ("Uncertainty", uncertainty), ("Model comparison", model)]:
        if ds is not None:
            html.extend([f"<h2>{title}</h2><table><tr><th>Statistic</th><th>Value</th></tr>", rows(ds), "</table>"])
    html.extend(["<h2>Publication checklist</h2><ul>",
                 "<li>State the threshold-selection rule and candidate thresholds.</li>",
                 "<li>Report the exceedance rate and dependence/declustering rule.</li>",
                 "<li>Report parameter uncertainty and return-level uncertainty.</li>",
                 "<li>Report the reference period and physical units of return periods.</li>",
                 "<li>For non-stationarity, report covariates, scaling, model comparison, and uncertainty.</li>",
                 "</ul></body></html>"])
    return "".join(html)


__all__ = ["plot_threshold_stability", "plot_mean_excess", "plot_gpd_qq", "pot_diagnostic_report"]
