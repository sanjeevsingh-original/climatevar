"""Precipitation verification and comparative evaluation utilities."""
from .compare import align_period, common_grid, compare_products, evaluate_product, rank_products
from .datasets import PRESETS, ProductSpec, load_precipitation
from .experiment import DEFAULT_INTENSITIES, DEFAULT_SEASONS, ExperimentConfig, ExperimentResult, REGION_BOUNDS, run_experiment, save_experiment
from .odisha import ODISHA_BBOX, odisha_experiment, subset_odisha, validate_input_paths

__all__ = [
    "align_period", "common_grid", "compare_products", "evaluate_product", "rank_products",
    "PRESETS", "ProductSpec", "load_precipitation",
    "DEFAULT_INTENSITIES", "DEFAULT_SEASONS", "REGION_BOUNDS", "ExperimentConfig", "ExperimentResult",
    "run_experiment", "save_experiment", "ODISHA_BBOX", "odisha_experiment", "subset_odisha", "validate_input_paths",
]
