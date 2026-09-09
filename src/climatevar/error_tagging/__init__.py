"""Rainfall model-error tagging and leakage-safe ML dataset utilities."""
from .labels import error_class, intensity_class
from .features import build_error_features, to_ml_table
from .split import event_group_split, spatial_block_split, temporal_split, validate_no_overlap

__all__ = [
    "error_class", "intensity_class", "build_error_features", "to_ml_table",
    "temporal_split", "event_group_split", "spatial_block_split",
    "validate_no_overlap",
]
