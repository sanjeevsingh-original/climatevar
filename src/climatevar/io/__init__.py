"""Input/output and interoperability helpers for common climate datasets."""
from .dataset import open_dataset
from .normalize import find_variable, normalize_coords, normalize_dataset, standardize_variables
from .schema import SCHEMA, ClimateSchema, describe_dataset, validate_dataset
from .temporal import accumulated_to_increment, resample_precipitation, sort_and_validate_time
from ..grid import normalize_longitude, regrid
from ..time import calendar_name, infer_frequency, time_step_seconds, validate_time
from ..units import SI_UNITS, convert_units, normalize_units, to_si


def precipitation_to_mm(data):
    """Backward-compatible explicit conversion of precipitation amounts to mm."""
    return convert_units(data, "mm")


def temperature_to_celsius(data):
    """Backward-compatible explicit conversion of temperature to degrees Celsius."""
    kelvin = convert_units(data, "K")
    out = kelvin - 273.15
    out.attrs = dict(kelvin.attrs)
    out.attrs["units"] = "degC"
    out.attrs["climatevar:target_units"] = "degC"
    return out


__all__ = [
    "SCHEMA", "ClimateSchema", "SI_UNITS", "accumulated_to_increment", "calendar_name",
    "convert_units", "describe_dataset", "find_variable", "infer_frequency", "normalize_coords",
    "normalize_dataset", "normalize_longitude", "normalize_units", "open_dataset", "regrid",
    "resample_precipitation", "sort_and_validate_time", "standardize_variables", "time_step_seconds",
    "to_si", "validate_dataset", "validate_time", "precipitation_to_mm", "temperature_to_celsius",
]
