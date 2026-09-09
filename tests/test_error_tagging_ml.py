import numpy as np
import pytest

pytest.importorskip("sklearn")
from climatevar.error_tagging import classification_metrics, cross_validate_classifier, make_classifier


def test_classifier_factory_and_metrics():
    X = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [2., 1.], [1., 2.]])
    y = np.array(["under", "under", "over", "over", "severe", "severe"])
    model = make_classifier("random_forest", n_estimators=20)
    model.fit(X, y)
    pred = model.predict(X)
    assert set(pred).issubset(set(y))
    scores = classification_metrics(y, pred, model.predict_proba(X))
    assert 0 <= scores["balanced_accuracy"] <= 1
    assert "f1_macro" in scores


def test_group_cross_validation():
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.array(["a", "b"] * 10)
    groups = np.repeat(np.arange(10), 2)
    rows = cross_validate_classifier(X, y, groups=groups, strategy="group", n_splits=5,
                                     model="random_forest", n_estimators=20)
    assert len(rows) == 5
    assert all(r["n_test"] > 0 for r in rows)
