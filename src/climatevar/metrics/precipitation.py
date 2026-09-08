"""Binary precipitation-event verification metrics."""

from __future__ import annotations

import xarray as xr


def contingency_table(observation, prediction, threshold: float = 1.0, dim: str | None = None):
    """Return hits, misses, false alarms and correct negatives."""
    obs = observation >= threshold
    pred = prediction >= threshold
    return {
        "hits": (obs & pred).sum(dim=dim),
        "misses": (obs & ~pred).sum(dim=dim),
        "false_alarms": (~obs & pred).sum(dim=dim),
        "correct_negatives": (~obs & ~pred).sum(dim=dim),
    }


def _counts(observation, prediction, threshold, dim):
    return contingency_table(observation, prediction, threshold, dim)


def pod(observation, prediction, threshold: float = 1.0, dim: str | None = None):
    """Probability of detection (hit rate)."""
    c = _counts(observation, prediction, threshold, dim)
    return c["hits"] / (c["hits"] + c["misses"])


def far(observation, prediction, threshold: float = 1.0, dim: str | None = None):
    """False alarm ratio."""
    c = _counts(observation, prediction, threshold, dim)
    return c["false_alarms"] / (c["hits"] + c["false_alarms"])


def f1_score(observation, prediction, threshold: float = 1.0, dim: str | None = None):
    """F1 score for a binary precipitation event."""
    c = _counts(observation, prediction, threshold, dim)
    precision = c["hits"] / (c["hits"] + c["false_alarms"])
    recall = c["hits"] / (c["hits"] + c["misses"])
    return 2.0 * precision * recall / (precision + recall)


def heidke_skill_score(observation, prediction, threshold: float = 1.0, dim: str | None = None):
    """Heidke Skill Score for a binary event."""
    c = _counts(observation, prediction, threshold, dim)
    h, m, f, cn = c["hits"], c["misses"], c["false_alarms"], c["correct_negatives"]
    total = h + m + f + cn
    expected = ((h + m) * (h + f) + (f + cn) * (m + cn)) / total
    return (h + cn - expected) / (total - expected)
