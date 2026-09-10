"""Publication-grade cartographic helpers for precipitation verification."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise ImportError(
            "Cartopy is required for publication_map(); install climatevar[cartography]."
        ) from exc
    return ccrs, cfeature


def _load_boundary(boundary):
    import geopandas as gpd
    if hasattr(boundary, "geometry"):
        return boundary
    return gpd.read_file(boundary)


def _save(fig, output):
    if output is None:
        return None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, bbox_inches="tight", facecolor="white")
    return str(output)


def publication_map(
    field,
    *,
    output=None,
    boundary=None,
    topography=None,
    title=None,
    cmap="RdBu_r",
    levels=15,
    vmin=None,
    vmax=None,
    significance=None,
    significance_level=0.05,
    extend="both",
    figsize=(7.2, 6.0),
    panel_label=None,
):
    """Create a journal-oriented map of a lat/lon verification field.

    Parameters
    ----------
    field : xarray.DataArray
        Two-dimensional field with 1-D ``lat`` and ``lon`` coordinates.
    boundary : path or GeoDataFrame, optional
        Administrative boundary, preferably the authoritative Odisha boundary.
    topography : xarray.DataArray or path, optional
        DEM used as a subdued terrain contour background.
    significance : xarray.DataArray, optional
        Boolean mask or p-value field. Boolean masks are used directly;
        numeric fields are interpreted as p-values and thresholded at
        ``significance_level``.

    Notes
    -----
    This function does not calculate statistical significance. The mask must
    come from a declared statistical procedure so the plotting layer remains
    scientifically transparent.
    """
    import matplotlib.pyplot as plt
    import xarray as xr

    ccrs, cfeature = _cartopy()
    if "lat" not in field.coords or "lon" not in field.coords:
        raise ValueError("field must have lat/lon coordinates")
    if field.lat.ndim != 1 or field.lon.ndim != 1:
        raise ValueError("publication_map currently requires 1-D lat/lon coordinates")

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [float(field.lon.min()), float(field.lon.max()), float(field.lat.min()), float(field.lat.max())],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, linewidth=0.35, facecolor="none")

    if topography is not None:
        if isinstance(topography, (str, Path)):
            topo = xr.open_dataarray(topography)
        else:
            topo = topography
        if "lat" in topo.coords and "lon" in topo.coords:
            topo = topo.interp(lat=field.lat, lon=field.lon)
            ax.contour(
                topo.lon, topo.lat, topo,
                levels=8, colors="0.35", linewidths=0.35,
                transform=ccrs.PlateCarree(), alpha=0.45,
            )

    finite = np.asarray(field.values, dtype=float)
    if vmin is None or vmax is None:
        scale = np.nanmax(np.abs(finite)) if np.isfinite(finite).any() else 1.0
        vmin = -scale if vmin is None else vmin
        vmax = scale if vmax is None else vmax

    mesh = field.plot.pcolormesh(
        ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
        vmin=vmin, vmax=vmax, add_colorbar=False, shading="auto",
    )
    cb = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.82, pad=0.035, extend=extend)
    cb.set_label(str(field.name or "verification metric"))

    if significance is not None:
        if significance.dtype.kind in "b":
            sig = significance
        else:
            sig = significance < significance_level
        sig = sig.broadcast_like(field)
        lat2, lon2 = np.meshgrid(field.lat.values, field.lon.values, indexing="ij")
        ax.scatter(
            lon2[sig.values], lat2[sig.values],
            s=3, marker=".", color="black", alpha=0.55,
            transform=ccrs.PlateCarree(), linewidths=0,
        )

    if boundary is not None:
        gdf = _load_boundary(boundary)
        gdf.boundary.plot(ax=ax, transform=ccrs.PlateCarree(), linewidth=0.75, color="black")

    gl = ax.gridlines(draw_labels=True, linewidth=0.35, alpha=0.45, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    ax.set_title(title or str(field.name or "Verification"), pad=8)
    if panel_label:
        ax.text(
            0.01, 0.99, str(panel_label), transform=ax.transAxes,
            ha="left", va="top", fontsize=12, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        )
    fig.tight_layout()
    return fig, _save(fig, output)


def export_publication_figure(fig, output_stem):
    """Export one figure as 600-dpi PNG, TIFF and vector PDF."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix, kwargs in (
        (".png", {"dpi": 600}),
        (".tif", {"dpi": 600}),
        (".pdf", {}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs[suffix[1:]] = str(path)
    return outputs
