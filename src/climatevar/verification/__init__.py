"""Precipitation verification and comparative evaluation utilities."""
from .compare import align_period, common_grid, compare_products, evaluate_product, rank_products
from .datasets import PRESETS, ProductSpec, load_precipitation
from .experiment import DEFAULT_INTENSITIES, DEFAULT_SEASONS, ExperimentConfig, ExperimentResult, run_experiment, save_experiment

__all__ = [
    "align_period", "common_grid", "compare_products", "evaluate_product", "rank_products",
    "PRESETS", "ProductSpec", "load_precipitation",
    "DEFAULT_INTENSITIES", "DEFAULT_SEASONS", "ExperimentConfig", "ExperimentResult",
    "run_experiment", "save_experiment",
]
