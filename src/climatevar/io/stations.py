"""Station-observation normalization."""
from __future__ import annotations
from typing import Mapping
import pandas as pd
import xarray as xr

STATION_ALIASES: dict[str, tuple[str, ...]] = {
    "station_id": ("station_id", "station", "station_code", "stn", "id"),
    "station_name": ("station_name", "name", "station_name_en"),
    "latitude": ("latitude", "lat", "LAT", "Latitude"),
    "longitude": ("longitude", "lon", "LON", "Longitude"),
    "elevation": ("elevation", "elev", "altitude", "height"),
    "time": ("time", "datetime", "date", "Date", "DATE"),
    "precipitation": ("precipitation", "precip", "rainfall", "rain", "rf", "pr"),
    "temperature": ("temperature", "temp", "tmean", "tavg", "T2M"),
}

def normalize_station_dataframe(df: pd.DataFrame, *, columns: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Normalize common station CSV/table columns to climatevar names."""
    out = df.copy(); explicit = dict(columns or {}); lookup = {str(c).casefold(): c for c in out.columns}; renames = {}
    for canonical, aliases in STATION_ALIASES.items():
        source = explicit.get(canonical)
        if source is None:
            for alias in aliases:
                if alias in out.columns: source = alias; break
                if alias.casefold() in lookup: source = lookup[alias.casefold()]; break
        if source and source != canonical and canonical not in out.columns: renames[source] = canonical
    out = out.rename(columns=renames)
    if "time" in out: out["time"] = pd.to_datetime(out["time"], errors="coerce")
    return out

def station_dataframe_to_xarray(df: pd.DataFrame) -> xr.Dataset:
    """Convert a normalized station table into an xarray Dataset."""
    out = normalize_station_dataframe(df)
    if "time" not in out: raise ValueError("Station data requires a time column.")
    return out.set_index("time").to_xarray()
