"""Input/output helpers for common climate datasets."""
from .dataset import open_dataset
from .grid import normalize_latitude, normalize_longitude
from .normalize import find_variable, normalize_coords, normalize_dataset, standardize_variables
from .schema import SCHEMA, ClimateSchema, describe_dataset, validate_dataset
from .temporal import accumulated_to_increment, resample_precipitation, sort_and_validate_time
from .units import precipitation_to_mm, temperature_to_celsius

__all__ = [
    "SCHEMA", "ClimateSchema", "accumulated_to_increment", "describe_dataset",
    "find_variable", "normalize_coords", "normalize_dataset", "normalize_latitude",
    "normalize_longitude", "open_dataset", "precipitation_to_mm", "resample_precipitation",
    "sort_and_validate_time", "standardize_variables", "temperature_to_celsius", "validate_dataset",
]
