import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from climatevar.verification.cartography import export_publication_figure


def test_export_publication_figure(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    outputs = export_publication_figure(fig, tmp_path / "figure1")
    assert set(outputs) == {"png", "tif", "pdf"}
    assert all((tmp_path / f"figure1.{ext}").exists() for ext in outputs)
    plt.close(fig)


def test_field_has_expected_coordinates():
    field = xr.DataArray(
        np.ones((2, 3)),
        dims=("lat", "lon"),
        coords={"lat": [18.0, 22.0], "lon": [82.0, 85.0, 87.0]},
        name="bias",
    )
    assert field.lat.ndim == 1
    assert field.lon.ndim == 1
