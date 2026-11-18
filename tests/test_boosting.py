"""Testes do LightGBM com paragem temporal (boosting.py) e do catálogo (model_catalog.py)."""

import importlib.util
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from lending.boosting import LGBMEarlyStoppingClassifier, LGBMEarlyStoppingRegressor, last_rows_cut
from lending.model_catalog import classification_specs, regression_specs


def make_data(n=2_000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"x{i}" for i in range(5)])
    y_reg = 12 + 3 * X["x0"] - 2 * X["x1"] ** 2 + rng.normal(0, 0.5, n)
    score = X["x0"] - X["x1"] + rng.normal(0, 0.7, n)
    y_cls = np.where(score > np.quantile(score, 0.85), "Default", "Good standing")
    return X, y_reg.to_numpy(), y_cls


def test_last_rows_cut():
    assert last_rows_cut(100, 0.15) == 85
    assert last_rows_cut(100, 0.0) == 100
    with pytest.raises(ValueError):
        last_rows_cut(100, 1.0)


def test_regressor_learns_and_stops_early():
    X, y, _ = make_data()
    model = LGBMEarlyStoppingRegressor(learning_rate=0.1, num_leaves=63).fit(X, y)
    assert model.n_train_ == 1_700 and model.n_valid_ == 300
    assert model.best_iteration_ < 500
    mae = np.abs(model.predict(X) - y).mean()
    assert mae < 0.5 * np.abs(y - np.median(y)).mean()
    assert list(model.feature_importances().index[:2]) in (["x0", "x1"], ["x1", "x0"])


def test_early_stopping_is_measured_on_the_last_rows():
    # As últimas linhas têm um alvo diferente: a paragem tem de reagir a elas, e não a uma
    # parte aleatória do treino
    X, y, _ = make_data()
    shifted = y.copy()
    shifted[-300:] = 0.0
    a = LGBMEarlyStoppingRegressor().fit(X, y)
    b = LGBMEarlyStoppingRegressor().fit(X, shifted)
    assert a.best_iteration_ != b.best_iteration_


def test_without_validation_uses_all_trees():
    X, y, _ = make_data(n=500)
    model = LGBMEarlyStoppingRegressor(n_estimators=30, validation_fraction=0.0).fit(X, y)
    assert model.best_iteration_ == 30 and model.n_valid_ == 0


def test_classifier_labels_probabilities_and_weights():
    X, _, y = make_data()
    model = LGBMEarlyStoppingClassifier(class_weight="balanced").fit(X, y)
    assert list(model.classes_) == ["Default", "Good standing"]
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2) and np.allclose(proba.sum(axis=1), 1)
    assert roc_auc_score(y == "Default", proba[:, 0]) > 0.85
    assert set(model.predict(X)) <= set(model.classes_)
    weighted = LGBMEarlyStoppingClassifier(class_weight={"Default": 5.0, "Good standing": 1.0})
    assert weighted.fit(X, y).predict_proba(X)[:, 0].mean() > proba[:, 0].mean() * 0.5


def test_classifier_rejects_three_classes():
    X, _, _ = make_data(n=300)
    with pytest.raises(ValueError, match="binária"):
        LGBMEarlyStoppingClassifier().fit(X, np.resize(["Good standing", "At risk", "Default"], 300))


def test_clone_pickle_and_columns():
    X, y, _ = make_data(n=600)
    model = LGBMEarlyStoppingRegressor(num_leaves=15).fit(X, y)
    assert clone(model).get_params()["num_leaves"] == 15
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(restored.predict(X), model.predict(X))
    with pytest.raises(ValueError):
        model.predict(X[X.columns[::-1]])


# ---------------------------------------------------------------------------------------
# Catálogo (grelhas da secção 3)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("specs_for", [regression_specs, classification_specs])
def test_catalog_grids_match_section_3(specs_for):
    specs = specs_for(include_mlp=False)
    names = [s.name for s in specs]
    assert len(set(names)) == len(names)
    sizes = {s.name: len(s.configs()) for s in specs}
    assert sizes["kNN"] == 10
    tuned = sum(n for name, n in sizes.items() if name in
                {"kNN", "Ridge", "Regressão logística", "Árvore", "Random Forest", "LightGBM"})
    assert tuned == 10 + 6 + 8 + 4 + 6     # + MLP 6 = 40, como na secção 3
    assert [s.name for s in specs if not s.preprocess] == names[:2]   # baselines
    for spec in specs:                     # todas as configurações são válidas
        for _, params in spec.configs():
            spec.make(params)


def test_collinear_feature_dropped_only_in_logistic():
    specs = {s.name: s for s in classification_specs(include_mlp=False)}
    assert specs["Regressão logística"].drop_features == ("dti_with_loan",)
    assert specs["kNN"].drop_features == ("dti_with_loan",)
    assert specs["Árvore"].drop_features == ()


@pytest.mark.skipif(importlib.util.find_spec("keras") is None
                    or importlib.util.find_spec("lending.deep") is None,
                    reason="sem TensorFlow ou sem o deep.py (ramo mlp-keras ainda por juntar)")
def test_catalog_mlp_grid():
    mlp = [s for s in regression_specs() if s.name == "MLP"][0]
    assert len(mlp.configs()) == 6
    model = mlp.make(mlp.configs()[-1][1])
    assert model.hidden_layers == (256, 128, 64) and model.learning_rate == 3e-4
