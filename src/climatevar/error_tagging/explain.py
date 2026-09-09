"""Model explanation utilities for rainfall error classifiers."""
from __future__ import annotations


def permutation_importance(estimator, X, y, scoring="balanced_accuracy", n_repeats=10, random_state=0):
    """Compute validation-set permutation importance; avoids tree impurity bias."""
    try:
        from sklearn.inspection import permutation_importance as _pi
    except ImportError as exc:
        raise ImportError("Explainability requires scikit-learn; install `climatevar[ml]`.") from exc
    return _pi(estimator, X, y, scoring=scoring, n_repeats=n_repeats,
               random_state=random_state)


def shap_values(estimator, X, max_samples=2000, random_state=0):
    """Return SHAP explanations when optional SHAP is installed.

    Tree models are handled with TreeExplainer. For large climate datasets only a
    reproducible sample is explained to control memory and runtime.
    """
    try:
        import numpy as np
        import shap
    except ImportError as exc:
        raise ImportError("SHAP requires `pip install climatevar[shap]`.") from exc
    X_arr = X
    if hasattr(X, "sample") and len(X) > max_samples:
        X_arr = X.sample(max_samples, random_state=random_state)
    elif len(X) > max_samples:
        rng = np.random.default_rng(random_state)
        X_arr = X[rng.choice(len(X), size=max_samples, replace=False)]
    explainer = shap.TreeExplainer(estimator)
    return explainer(X_arr)
