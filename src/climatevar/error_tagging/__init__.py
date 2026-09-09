"""Rainfall model-error tagging and leakage-safe ML dataset utilities."""
from .labels import error_class, intensity_class
from .features import build_error_features, to_ml_table
from .split import assign_split, event_group_split, spatial_block_split, temporal_split, validate_no_overlap
from .models import classification_metrics, fit_classifier, make_classifier
from .evaluate import cross_validate_classifier
from .explain import permutation_importance, shap_values

__all__ = [
    "error_class", "intensity_class", "build_error_features", "to_ml_table",
    "temporal_split", "event_group_split", "spatial_block_split", "validate_no_overlap",
    "assign_split", "make_classifier", "fit_classifier", "classification_metrics",
    "cross_validate_classifier", "permutation_importance", "shap_values",
]
