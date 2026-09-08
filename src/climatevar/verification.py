"""Categorical and probabilistic precipitation verification utilities."""

from __future__ import annotations

import xarray as xr


def contingency(observed: xr.DataArray, predicted: xr.DataArray, threshold: float = 1.0):
    """Return hits, misses, false alarms and correct negatives."""
    if observed.shape != predicted.shape:
        raise ValueError("observed and predicted must have the same shape.")
    o = observed >= threshold
    p = predicted >= threshold
    return {
        "hits": (o & p).sum(),
        "misses": (o & ~p).sum(),
        "false_alarms": (~o & p).sum(),
        "correct_negatives": (~o & ~p).sum(),
    }


def pod(observed: xr.DataArray, predicted: xr.DataArray, threshold: float = 1.0):
    """Probability of detection."""
    c = contingency(observed, predicted, threshold)
    return c["hits"] / (c["hits"] + c["misses"])


def far(observed: xr.DataArray, predicted: xr.DataArray, threshold: float = 1.0):
    """False alarm ratio."""
    c = contingency(observed, predicted, threshold)
    return c["false_alarms"] / (c["hits"] + c["false_alarms"])


def f1_score(observed: xr.DataArray, predicted: xr.DataArray, threshold: float = 1.0):
    """F1 score for binary precipitation occurrence."""
    c = contingency(observed, predicted, threshold)
    precision = c["hits"] / (c["hits"] + c["false_alarms"])
    recall = c["hits"] / (c["hits"] + c["misses"])
    return 2 * precision * recall / (precision + recall)
