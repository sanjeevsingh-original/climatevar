"""Precipitation verification and comparative evaluation utilities."""
from .compare import align_period, common_grid, compare_products, evaluate_product, rank_products
from .datasets import PRESETS, ProductSpec, load_precipitation
from .experiment import DEFAULT_INTENSITIES, DEFAULT_SEASONS, ExperimentConfig, ExperimentResult, REGION_BOUNDS, run_experiment, save_experiment
from .odisha import ODISHA_BBOX, odisha_experiment, subset_odisha, validate_input_paths
from .plots import plot_scorecard, plot_spatial_metric
from .report import error_feature_distribution, generate_publication_report, intensity_distribution, manuscript_summary, ranking_heatmap, seasonal_heatmap, spatial_metric_panels, uncertainty_plot
from .cartography import export_publication_figure, publication_map

__all__ = [
    "align_period", "common_grid", "compare_products", "evaluate_product", "rank_products",
    "PRESETS", "ProductSpec", "load_precipitation",
    "DEFAULT_INTENSITIES", "DEFAULT_SEASONS", "REGION_BOUNDS", "ExperimentConfig", "ExperimentResult",
    "run_experiment", "save_experiment", "ODISHA_BBOX", "odisha_experiment", "subset_odisha", "validate_input_paths",
    "plot_spatial_metric", "plot_scorecard",
    "spatial_metric_panels", "seasonal_heatmap", "intensity_distribution", "error_feature_distribution",
    "uncertainty_plot", "ranking_heatmap", "manuscript_summary", "generate_publication_report",
    "publication_map", "export_publication_figure",
]
