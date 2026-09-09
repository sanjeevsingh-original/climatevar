import numpy as np
import xarray as xr

from climatevar.verification import compare_products, evaluate_product, rank_products


def test_perfect_product_has_perfect_event_skill():
    t = xr.date_range("2000-01-01", periods=5, freq="D")
    obs = xr.DataArray([0., 2., 0., 5., 1.], dims="time", coords={"time": t})
    row = evaluate_product(obs, obs, name="perfect")
    assert row["bias"] == 0.0
    assert row["rmse"] == 0.0
    assert row["pod"] == 1.0
    assert row["threat_score"] == 1.0
    assert row["frequency_bias"] == 1.0


def test_compare_and_rank_multiple_products():
    t = xr.date_range("2000-01-01", periods=5, freq="D")
    obs = xr.DataArray(np.array([0., 2., 0., 5., 1.]), dims="time", coords={"time": t})
    products = {"ERA5": obs, "IMERG": obs * 0.8, "WRF": obs * 1.2}
    rows = compare_products(obs, products)
    assert {r["dataset"] for r in rows} == set(products)
    ranked = rank_products(rows)
    assert ranked.iloc[0]["dataset"] == "ERA5"
