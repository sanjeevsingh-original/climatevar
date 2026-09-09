"""Leakage-aware model evaluation for rainfall error tagging."""
from __future__ import annotations

import numpy as np


def cross_validate_classifier(X, y, groups=None, strategy="group", n_splits=5,
                              model="hist_gradient_boosting", random_state=0, **model_kwargs):
    """Evaluate a classifier with time/group-aware folds.

    strategy='time' uses TimeSeriesSplit; strategy='group' uses GroupKFold;
    strategy='stratified_group' preserves class proportions where feasible.
    """
    try:
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, TimeSeriesSplit
    except ImportError as exc:
        raise ImportError("Cross-validation requires `climatevar[ml]`.") from exc
    from .models import fit_classifier, classification_metrics

    n = len(y)
    if strategy == "time":
        splitter = TimeSeriesSplit(n_splits=n_splits)
        splits = splitter.split(X)
    elif strategy == "group":
        if groups is None:
            raise ValueError("groups is required for strategy='group'.")
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups)
    elif strategy == "stratified_group":
        if groups is None:
            raise ValueError("groups is required for strategy='stratified_group'.")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=random_state)
        splits = splitter.split(X, y, groups)
    else:
        raise ValueError("strategy must be 'time', 'group', or 'stratified_group'.")

    rows = []
    for fold, (train, test) in enumerate(splits):
        est = fit_classifier(X[train], np.asarray(y)[train], model=model,
                             random_state=random_state + fold, **model_kwargs)
        pred = est.predict(X[test])
        prob = est.predict_proba(X[test]) if hasattr(est, "predict_proba") else None
        metrics = classification_metrics(np.asarray(y)[test], pred, prob)
        metrics["fold"] = fold
        metrics["n_train"] = len(train)
        metrics["n_test"] = len(test)
        rows.append(metrics)
    return rows
