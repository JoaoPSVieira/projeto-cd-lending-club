"""
Testes do runner da validação cruzada (src/lending/experiment.py)

Usam dados sintéticos pequenos e um pré-processamento simples (StandardScaler), para não
dependerem dos pickles nem da tabela de elegibilidade do projeto.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lending.baselines import GradeRateBaseline
from lending.evaluation import best_threshold, classification_metrics
from lending.experiment import (ColumnDropper, CrossValidationRunner, ModelSpec, config_id)

REG_COLUMNS = ["term", "x1", "x2"]


def scaler():
    return StandardScaler().set_output(transform="pandas")


@pytest.fixture
def reg_data():
    """8 trimestres (2013-2014) por ordem BARALHADA, como nos pickles do projeto."""
    rng = np.random.default_rng(0)
    n = 1_600
    dates = pd.to_datetime(rng.choice(pd.date_range("2013-01-01", "2014-12-01", freq="MS"), n))
    df = pd.DataFrame({"issue_d": dates, "term": rng.choice([36.0, 60.0], n),
                       "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["int_rate"] = 12 + 2 * df["x1"] + 4 * (df["term"] == 60) + rng.normal(0, 1, n)
    assert not df["issue_d"].is_monotonic_increasing
    return df


@pytest.fixture
def cls_data():
    rng = np.random.default_rng(1)
    n = 2_000
    dates = pd.to_datetime(rng.choice(pd.date_range("2007-06-01", "2011-12-01", freq="MS"), n))
    df = pd.DataFrame({"issue_d": dates, "grade": rng.choice(list("ABCDEFG"), n),
                       "x1": rng.normal(size=n), "dti_with_loan": rng.normal(size=n)})
    score = df["x1"] + df["grade"].map(dict(zip("ABCDEFG", range(7)))) / 3 + rng.normal(0, 1, n)
    df["target"] = np.where(score > np.quantile(score, 0.85), "Default", "Good standing")
    return df


def reg_runner(tmp_path, **kwargs):
    return CrossValidationRunner("regression", results_dir=tmp_path, oof_dir=tmp_path,
                                 columns=REG_COLUMNS, make_preprocessor=scaler, verbose=False,
                                 **kwargs)


def cls_runner(tmp_path):
    from sklearn.compose import make_column_transformer
    prep = lambda: make_column_transformer(          # noqa: E731
        (StandardScaler(), ["x1", "dti_with_loan"]),
        verbose_feature_names_out=False).set_output(transform="pandas")
    return CrossValidationRunner("classification", results_dir=tmp_path, oof_dir=tmp_path,
                                 columns=["grade", "x1", "dti_with_loan"], make_preprocessor=prep,
                                 verbose=False)


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Guarda o que recebeu no fit, para verificar o que o runner entrega aos modelos."""

    calls = []

    def fit(self, X, y):
        RecordingRegressor.calls.append(X.copy())
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


# ---------------------------------------------------------------------------------------
# Contrato: dados ordenados por issue_d
# ---------------------------------------------------------------------------------------

def test_models_receive_training_rows_sorted_by_date(tmp_path, reg_data):
    RecordingRegressor.calls = []
    runner = CrossValidationRunner("regression", results_dir=tmp_path, oof_dir=tmp_path,
                                   columns=["issue_d", "x1"], verbose=False)
    runner.run(reg_data, [ModelSpec("Gravador", RecordingRegressor, preprocess=False)])
    assert len(RecordingRegressor.calls) == 6
    for X in RecordingRegressor.calls:
        assert X["issue_d"].is_monotonic_increasing


def test_sample_keeps_temporal_order_and_size(tmp_path, reg_data):
    RecordingRegressor.calls = []
    runner = CrossValidationRunner("regression", results_dir=tmp_path, oof_dir=tmp_path,
                                   columns=["issue_d", "x1"], verbose=False)
    results = runner.run(reg_data, [ModelSpec("Gravador", RecordingRegressor, preprocess=False,
                                              max_train_rows=300)])
    for X in RecordingRegressor.calls:
        assert len(X) <= 300
        assert X["issue_d"].is_monotonic_increasing
    assert (results["n_train"] <= 300).all()
    assert results["n_train"].iloc[-1] == 300


def test_prepare_sorts_and_validates(tmp_path, reg_data, cls_data):
    runner = reg_runner(tmp_path)
    assert runner.prepare(reg_data)["issue_d"].is_monotonic_increasing
    with pytest.raises(ValueError, match="Faltam colunas"):
        runner.prepare(reg_data.drop(columns="x2"))
    with pytest.raises(ValueError, match="valores em falta"):
        runner.prepare(reg_data.assign(int_rate=np.nan))
    with pytest.raises(ValueError, match="At risk"):
        cls_runner(tmp_path).prepare(cls_data.assign(target="At risk"))


# ---------------------------------------------------------------------------------------
# Corrida, ficheiros e retoma
# ---------------------------------------------------------------------------------------

def test_run_writes_one_row_per_config_and_fold(tmp_path, reg_data):
    runner = reg_runner(tmp_path)
    specs = [ModelSpec("Mediana", lambda: DummyRegressor(strategy="median"), preprocess=False),
             ModelSpec("Ridge", Ridge, {"alpha": [0.1, 10.0]})]
    results = runner.run(reg_data, specs)
    assert len(results) == (1 + 2) * 6
    assert runner.folds(reg_data).shape[0] == 6
    assert set(results["valid_period"]) == {"2013Q3", "2013Q4", "2014Q1", "2014Q2", "2014Q3", "2014Q4"}
    assert (results.query("model == 'Mediana'")["prep_seconds"] == 0).all()
    assert results["config_id"].map(type).eq(str).all()
    assert json.loads(results.query("model == 'Ridge'")["params"].iloc[0]) in [{"alpha": 0.1},
                                                                              {"alpha": 10.0}]


def test_run_resumes_without_repeating(tmp_path, reg_data):
    runner = reg_runner(tmp_path)
    ridge = ModelSpec("Ridge", Ridge, {"alpha": [1.0]})
    first = runner.run(reg_data, [ridge])
    again = runner.run(reg_data, [ridge, ModelSpec("Mediana", DummyRegressor, preprocess=False)])
    assert len(again) == len(first) + 6
    assert again.query("model == 'Ridge'")["timestamp"].tolist() == first["timestamp"].tolist()


def test_oof_covers_every_validation_row_once(tmp_path, reg_data):
    runner = reg_runner(tmp_path)
    spec = ModelSpec("Ridge", Ridge, {"alpha": [1.0]})
    runner.run(reg_data, [spec])
    cid = spec.configs()[0][0]
    oof = runner.oof("Ridge", cid)
    sorted_data = runner.prepare(reg_data)
    first_valid = sorted_data["issue_d"].dt.to_period("Q").eq(pd.Period("2013Q3")).idxmax()
    assert oof["row"].is_unique
    assert oof["row"].min() == first_valid and oof["row"].max() == len(reg_data) - 1
    assert oof["y_true"].to_numpy() == pytest.approx(sorted_data.loc[oof["row"], "int_rate"].to_numpy())


def test_config_id_is_read_back_as_text(tmp_path, reg_data):
    runner = reg_runner(tmp_path)
    runner.run(reg_data, [ModelSpec("Ridge", Ridge, {"alpha": [1.0]})])
    csv = runner.results_path.read_text().replace(runner.results()["config_id"].iloc[0], "0012e45678")
    runner.results_path.write_text(csv)
    assert runner.results()["config_id"].iloc[0] == "0012e45678"


def test_duplicate_model_names_are_rejected(tmp_path, reg_data):
    with pytest.raises(ValueError, match="repetidos"):
        reg_runner(tmp_path).run(reg_data, [ModelSpec("A", Ridge), ModelSpec("A", DummyRegressor)])


def test_config_id_is_stable():
    assert config_id("Ridge", {"alpha": 1.0}) == config_id("Ridge", {"alpha": 1.0})
    assert config_id("Ridge", {"alpha": 1.0}) != config_id("Ridge", {"alpha": 10.0})
    assert config_id("MLP", {"hidden_layers": (64,)}) == config_id("MLP", {"hidden_layers": [64]})


# ---------------------------------------------------------------------------------------
# Resumos e modelo final
# ---------------------------------------------------------------------------------------

def test_regression_summary_and_best(tmp_path, reg_data):
    runner = reg_runner(tmp_path)
    runner.run(reg_data, [ModelSpec("Mediana", DummyRegressor, preprocess=False),
                          ModelSpec("Ridge", Ridge, {"alpha": [0.01, 1000.0]})])
    summary = runner.summary()
    assert summary["mae_mean"].is_monotonic_increasing
    ridge = runner.results().query("model == 'Ridge'")
    first = summary.iloc[0]
    rows = ridge[ridge["config_id"] == first["config_id"]]
    assert first["mae_mean"] == pytest.approx(rows["mae"].mean())
    assert first["mae_std"] == pytest.approx(rows["mae"].std())
    best = runner.best()
    assert best["model"].tolist() == ["Ridge", "Mediana"]
    table = runner.comparison_table()
    assert "±" in table.loc["Ridge", "mae"]


def test_classification_summary_uses_chosen_threshold(tmp_path, cls_data):
    runner = cls_runner(tmp_path)
    spec = ModelSpec("Regressão logística", lambda: LogisticRegression(class_weight="balanced"),
                     {"C": [1.0]}, drop_features=("dti_with_loan",))
    runner.run(cls_data, [spec, ModelSpec("Taxa do grau", GradeRateBaseline, preprocess=False)])
    summary = runner.summary().set_index("model")
    cid = spec.configs()[0][0]
    oof = runner.oof("Regressão logística", cid)
    threshold, _ = best_threshold(oof["y_true"], oof["p_default"])
    assert summary.loc["Regressão logística", "threshold"] == threshold == runner.threshold(
        "Regressão logística", cid)
    per_fold = [classification_metrics(f["y_true"], f["p_default"], threshold)["f1_macro"]
                for _, f in oof.groupby("fold")]
    assert summary.loc["Regressão logística", "f1_macro_mean"] == pytest.approx(np.mean(per_fold))
    assert (runner.results()["threshold"] == 0.5).all()       # as linhas do CSV usam 0,5


def test_column_dropper_removes_feature_and_indicator():
    X = pd.DataFrame({"dti": [1.0], "dti_with_loan": [2.0], "missingindicator_dti_with_loan": [0.0]})
    dropper = ColumnDropper(("dti_with_loan",)).fit(X)
    assert list(dropper.transform(X).columns) == ["dti"]


def test_refit_returns_ready_pipeline(tmp_path, reg_data, cls_data):
    runner = reg_runner(tmp_path)
    spec = ModelSpec("Ridge", Ridge, {"alpha": [0.01, 1000.0]})
    runner.run(reg_data, [spec])
    pipe = runner.refit(reg_data, spec)
    assert isinstance(pipe, Pipeline)
    assert pipe["model"].alpha == json.loads(runner.best()["params"].iloc[0])["alpha"]
    assert pipe.predict(reg_data[REG_COLUMNS]).shape == (len(reg_data),)

    runner_c = cls_runner(tmp_path)
    logistic = ModelSpec("Regressão logística", LogisticRegression, {"C": [1.0]},
                         drop_features=("dti_with_loan",))
    runner_c.run(cls_data, [logistic])
    pipe_c = runner_c.refit(cls_data, logistic)
    assert [name for name, _ in pipe_c.steps] == ["prep", "drop", "model"]
    assert "dti_with_loan" not in pipe_c["model"].feature_names_in_
    assert pipe_c.predict_proba(cls_data[["grade", "x1", "dti_with_loan"]]).shape == (len(cls_data), 2)


def test_refit_without_results_fails_clearly(tmp_path, reg_data):
    with pytest.raises(ValueError, match="configuração completa"):
        reg_runner(tmp_path).refit(reg_data, ModelSpec("Ridge", Ridge, {"alpha": [1.0]}))
