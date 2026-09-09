"""Compare climatevar precipitation indices against an independent RClimDex run."""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from climatevar.precipitation import cdd, cwd, prcptot, r10mm, r20mm, r95p, r99p, rx1day, rx5day, sdii

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "tests" / "external" / "rclimdex_reference.csv"


def make_data():
    time = pd.date_range("1961-01-01", "1991-12-31", freq="D")
    idx = np.arange(1, time.size + 1, dtype=np.int64)
    values = ((idx * 37) % 200) / 10.0
    values[(idx % 97) == 0] += 25.0
    return xr.DataArray(values, coords={"time": time}, dims="time", attrs={"units": "mm"})


def scalar(result):
    return float(result.sel(year=1991).item())


def main():
    ref = pd.read_csv(REFERENCE).iloc[0]
    data = make_data()
    kw = {"min_valid_fraction": 0}
    values = {
        "rx1day": scalar(rx1day(data, **kw)),
        "rx5day": scalar(rx5day(data, **kw)),
        "r10mm": scalar(r10mm(data, **kw)),
        "r20mm": scalar(r20mm(data, **kw)),
        "cdd": scalar(cdd(data, **kw)),
        "cwd": scalar(cwd(data, **kw)),
        "sdii": scalar(sdii(data, **kw)),
        "r95p": scalar(r95p(data, min_valid_fraction=0)),
        "r99p": scalar(r99p(data, min_valid_fraction=0)),
        "prcptot": scalar(prcptot(data, **kw)),
    }
    for name, observed in values.items():
        expected = float(ref[name])
        if not np.isclose(observed, expected, rtol=1e-10, atol=1e-10):
            raise AssertionError(f"{name}: climatevar={observed!r}, RClimDex={expected!r}")
        print(f"PASS {name}: {observed:.12g}")


if __name__ == "__main__":
    main()
