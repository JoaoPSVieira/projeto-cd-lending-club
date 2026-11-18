"""Testes das métricas (src/lending/evaluation.py) e das baselines (src/lending/baselines.py)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error, r2_score,
                             roc_auc_score, root_mean_squared_error)

from lending.baselines import GradeRateBaseline, TermMedianBaseline
from lending.evaluation import (CLASSIFICATION_METRICS, best_threshold, bootstrap_ci,
                                classification_metrics, confusion_table, f1_macro_by_threshold,
                                format_mean_std, paired_bootstrap_diff, positive_proba,
                                regression_metrics)


@pytest.fixture
def binary():
    rng = np.random.default_rng(0)
    y = np.where(rng.random(1_000) < 0.15, "Default", "Good standing")
    p = np.clip(0.15 + 0.35 * (y == "Default") + rng.normal(0, 0.15, len(y)), 0, 1)
    return y, p


# ---------------------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------------------

def test_regression_metrics_match_sklearn():
    y, p = np.array([10.0, 12.0, 15.0, 20.0]), np.array([11.0, 12.5, 14.0, 18.0])
    out = regression_metrics(y, p)
    assert out == pytest.approx({"mae": mean_absolute_error(y, p),
                                 "rmse": root_mean_squared_error(y, p), "r2": r2_score(y, p)})


def test_classification_metrics_match_sklearn(binary):
    y, p = binary
    pred = np.where(p >= 0.3, "Default", "Good standing")
    out = classification_metrics(y, p, threshold=0.3)
    assert list(out) == CLASSIFICATION_METRICS
    assert out["f1_macro"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert out["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert out["recall_default"] == pytest.approx(((pred == "Default") & (y == "Default")).sum()
                                                  / (y == "Default").sum())
    assert out["auc"] == pytest.approx(roc_auc_score(y == "Default", p))


def test_micro_metrics_equal_accuracy_in_binary_case(binary):
    y, p = binary
    out = classification_metrics(y, p, threshold=0.4)
    assert out["f1_micro"] == pytest.approx(out["accuracy"])
    assert out["precision_micro"] == pytest.approx(out["accuracy"])
    assert out["recall_micro"] == pytest.approx(out["accuracy"])


def test_auc_is_nan_with_one_class():
    out = classification_metrics(np.array(["Good standing"] * 5), np.linspace(0, 1, 5))
    assert np.isnan(out["auc"]) and np.isnan(out["recall_default"])


def test_positive_proba_uses_class_name_not_position():
    X = np.zeros((4, 1))
    model = DummyClassifier(strategy="prior").fit(X, ["Default", "Good standing", "Good standing",
                                                      "Good standing"])
    assert list(model.classes_) == ["Default", "Good standing"]   # Default é a 1.ª coluna
    assert positive_proba(model, X) == pytest.approx([0.25] * 4)


def test_vectorized_f1_macro_matches_sklearn(binary):
    y, p = binary
    thresholds = np.array([0.1, 0.25, 0.5, 0.9])
    expected = [f1_score(y, np.where(p >= t, "Default", "Good standing"), average="macro")
                for t in thresholds]
    assert f1_macro_by_threshold(y, p, thresholds) == pytest.approx(expected)


def test_best_threshold_is_the_maximum_of_the_grid(binary):
    y, p = binary
    t, score = best_threshold(y, p)
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = f1_macro_by_threshold(y, p, grid)
    assert score == pytest.approx(scores.max())
    assert t == grid[np.argmax(scores)]
    assert classification_metrics(y, p, t)["f1_macro"] == pytest.approx(score)


def test_confusion_table_with_three_real_classes():
    y_true = ["Good standing", "At risk", "Default", "At risk", "Default"]
    y_pred = ["Good standing", "Default", "Default", "Good standing", "Good standing"]
    table = confusion_table(y_true, y_pred, row_labels=["Good standing", "At risk", "Default"])
    assert table.shape == (3, 2)
    assert table.loc["At risk"].tolist() == [1, 1]     # colunas: Good standing, Default
    assert table.to_numpy().sum() == 5


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    y = rng.normal(12, 3, 500)
    p = y + rng.normal(0, 1, 500)
    a = bootstrap_ci(mean_absolute_error, y, p, n_boot=300)
    b = bootstrap_ci(mean_absolute_error, y, p, n_boot=300)
    assert a == b
    assert a["low"] < a["estimate"] < a["high"]
    assert a["estimate"] == pytest.approx(mean_absolute_error(y, p))


def test_paired_bootstrap_of_identical_models_is_zero():
    y = np.arange(50.0)
    out = paired_bootstrap_diff(mean_absolute_error, y, y + 1, y + 1, n_boot=100)
    assert out == {"estimate": 0.0, "low": 0.0, "high": 0.0}


def test_format_mean_std():
    assert format_mean_std(1.23456, 0.0456, 2) == "1.23 ± 0.05"
    assert format_mean_std(1.0, np.nan, 1) == "1.0"


# ---------------------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------------------

def test_term_median_baseline():
    X = pd.DataFrame({"term": [36, 36, 36, 60, 60]})
    model = TermMedianBaseline().fit(X, [10, 11, 30, 15, 17])
    pred = model.predict(pd.DataFrame({"term": [36, 60, 48]}))
    assert pred.tolist() == [11.0, 16.0, 15.0]       # 48 meses não existe: mediana global


def test_term_median_baseline_needs_the_column():
    with pytest.raises(ValueError, match="term"):
        TermMedianBaseline().fit(np.zeros((3, 1)), [1, 2, 3])


def test_grade_rate_baseline():
    X = pd.DataFrame({"grade": ["A", "A", "A", "A", "G", "G"]})
    y = ["Good standing", "Good standing", "Good standing", "Default", "Default", "Default"]
    model = GradeRateBaseline().fit(X, y)
    assert list(model.classes_) == ["Default", "Good standing"]
    proba = model.predict_proba(pd.DataFrame({"grade": ["A", "G", "C"]}))
    assert proba[:, 0] == pytest.approx([0.25, 1.0, 0.5])      # C não existe: taxa global
    assert np.allclose(proba.sum(axis=1), 1)
    assert model.predict(pd.DataFrame({"grade": ["A", "G"]})).tolist() == ["Good standing", "Default"]


def test_grade_rate_baseline_needs_the_positive_class():
    with pytest.raises(ValueError, match="Default"):
        GradeRateBaseline().fit(pd.DataFrame({"grade": ["A", "B"]}), ["Good standing", "At risk"])
