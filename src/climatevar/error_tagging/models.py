"""Optional scikit-learn baselines for rainfall error classification."""
from __future__ import annotations

import numpy as np


def _sklearn():
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import (
            accuracy_score, balanced_accuracy_score, f1_score, log_loss,
            matthews_corrcoef, precision_score, recall_score, roc_auc_score,
        )
    except ImportError as exc:
        raise ImportError(
            "ML error tagging requires scikit-learn. Install with "
            "`pip install climatevar[ml]`."
        ) from exc
    return (HistGradientBoostingClassifier, RandomForestClassifier,
            CalibratedClassifierCV, accuracy_score, balanced_accuracy_score,
            f1_score, log_loss, matthews_corrcoef, precision_score,
            recall_score, roc_auc_score)


def make_classifier(model="hist_gradient_boosting", random_state=0, class_weight="balanced", **kwargs):
    """Create a reproducible baseline classifier without making sklearn a core dependency."""
    HGB, RF, *_ = _sklearn()
    if model in {"hist_gradient_boosting", "hgb"}:
        params = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                      l2_regularization=1.0, random_state=random_state,
                      class_weight=class_weight)
        params.update(kwargs)
        return HGB(**params)
    if model in {"random_forest", "rf"}:
        params = dict(n_estimators=400, min_samples_leaf=2, n_jobs=-1,
                      random_state=random_state, class_weight=class_weight)
        params.update(kwargs)
        return RF(**params)
    if model in {"xgboost", "xgb"}:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("XGBoost requires `pip install climatevar[xgboost]`.") from exc
        params = dict(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.85, colsample_bytree=0.85,
                      objective="multi:softprob", eval_metric="mlogloss",
                      random_state=random_state, n_jobs=-1)
        params.update(kwargs)
        return XGBClassifier(**params)
    raise ValueError("model must be 'hist_gradient_boosting', 'random_forest', or 'xgboost'")


def fit_classifier(X, y, model="hist_gradient_boosting", calibrate=False,
                   random_state=0, calibration_cv=3, **kwargs):
    """Fit a baseline classifier with optional sigmoid probability calibration."""
    estimator = make_classifier(model=model, random_state=random_state, **kwargs)
    if calibrate:
        _, _, CalibratedClassifierCV, *_ = _sklearn()
        estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=calibration_cv)
    estimator.fit(X, y)
    return estimator


def classification_metrics(y_true, y_pred, probability=None, labels=None):
    """Return class-imbalance-aware diagnostic metrics."""
    _, _, _, accuracy, balanced_accuracy, f1, log_loss, mcc, precision, recall, auc = _sklearn()
    out = {
        "accuracy": float(accuracy(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy(y_true, y_pred)),
        "f1_macro": float(f1(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall(y_true, y_pred, average="macro", zero_division=0)),
        "matthews_corrcoef": float(mcc(y_true, y_pred)),
    }
    if probability is not None:
        p = np.asarray(probability)
        classes = np.asarray(labels if labels is not None else np.unique(y_true))
        out["log_loss"] = float(log_loss(y_true, p, labels=classes))
        try:
            out["roc_auc_ovr_macro"] = float(auc(y_true, p, labels=classes, multi_class="ovr", average="macro"))
        except ValueError:
            out["roc_auc_ovr_macro"] = float("nan")
    return out
