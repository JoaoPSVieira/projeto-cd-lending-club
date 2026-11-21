"""
Testes da análise dos resultados (model_analysis.py), do treino num fold
(CrossValidationRunner.fold_pipeline) e da seleção de modelos do scripts/run_cv.py
"""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from lending.baselines import GradeRateBaseline, TermMedianBaseline
from lending.experiment import CrossValidationRunner, ModelSpec
from lending.model_analysis import CVAnalysis

matplotlib.use("Agg")

REG_COLUMNS = ["term", "x1", "x2"]
REG_SPECS = [ModelSpec("Mediana", lambda: DummyRegressor(strategy="median"), preprocess=False),
             ModelSpec("Mediana por prazo", TermMedianBaseline, preprocess=False),
             ModelSpec("Ridge", Ridge, {"alpha": [0.01, 1000.0]})]


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    n = 1_600
    dates = pd.to_datetime(rng.choice(pd.date_range("2013-01-01", "2014-12-01", freq="MS"), n))
    df = pd.DataFrame({"issue_d": dates, "term": rng.choice([36.0, 60.0], n),
                       "x1": rng.normal(size=n), "x2": rng.normal(size=n),
                       "sub_grade": rng.choice(["A1", "B2", "C3"], n)})
    df["int_rate"] = 12 + 2 * df["x1"] + 4 * (df["term"] == 60) + rng.normal(0, 1, n)
    return df


@pytest.fixture
def reg_analysis(tmp_path, reg_data):
    runner = CrossValidationRunner("regression", results_dir=tmp_path, oof_dir=tmp_path,
                                   columns=REG_COLUMNS, verbose=False,
                                   make_preprocessor=lambda: StandardScaler().set_output(transform="pandas"))
    runner.run(reg_data, REG_SPECS)
    return CVAnalysis(runner, reg_data)


@pytest.fixture
def cls_analysis(tmp_path):
    rng = np.random.default_rng(1)
    n = 2_000
    dates = pd.to_datetime(rng.choice(pd.date_range("2007-06-01", "2011-12-01", freq="MS"), n))
    df = pd.DataFrame({"issue_d": dates, "grade": rng.choice(list("ABCDEFG"), n), "x1": rng.normal(size=n)})
    score = df["x1"] + df["grade"].map(dict(zip("ABCDEFG", range(7)))) / 3 + rng.normal(0, 1, n)
    df["target"] = np.where(score > np.quantile(score, 0.85), "Default", "Good standing")
    prep = lambda: make_column_transformer((StandardScaler(), ["x1"]),   # noqa: E731
                                           verbose_feature_names_out=False).set_output(transform="pandas")
    runner = CrossValidationRunner("classification", results_dir=tmp_path, oof_dir=tmp_path,
                                   columns=["grade", "x1"], make_preprocessor=prep, verbose=False)
    runner.run(df, [ModelSpec("Taxa do grau", GradeRateBaseline, preprocess=False),
                    ModelSpec("Regressão logística", LogisticRegression, {"C": [1.0]})])
    return CVAnalysis(runner, df)


# ---------------------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------------------

def test_fold_table_matches_the_csv(reg_analysis):
    table = reg_analysis.fold_table("mae")
    assert table.shape == (3, 6)
    assert list(table.columns) == ["2013Q3", "2013Q4", "2014Q1", "2014Q2", "2014Q3", "2014Q4"]
    results = reg_analysis.runner.results()
    cid = reg_analysis.best_config("Ridge")
    expected = results[(results["model"] == "Ridge") & (results["config_id"] == cid)]["mae"]
    assert table.loc["Ridge"].to_numpy() == pytest.approx(expected.to_numpy())
    assert reg_analysis.models[0] == "Ridge"


def test_fold_table_in_classification_uses_the_chosen_threshold(cls_analysis):
    table = cls_analysis.fold_table("f1_macro")
    summary = cls_analysis.runner.summary().set_index("model")
    for model in table.index:
        assert table.loc[model].mean() == pytest.approx(summary.loc[model, "f1_macro_mean"])


def test_params_cost_and_improvement(reg_analysis):
    params = reg_analysis.params_table("Ridge", metrics=["mae", "r2"])
    assert list(params.columns[:3]) == ["alpha", "mae_mean", "mae_std"]
    assert len(params) == 2 and params["mae_mean"].is_monotonic_increasing
    cost = reg_analysis.cost_table()
    assert cost.loc["Ridge", "configuracoes"] == 2 and cost.loc["Ridge", "ajustes"] == 12
    improvement = reg_analysis.improvement("Mediana por prazo")
    assert improvement["Mediana por prazo"] == 0 and improvement["Ridge"] > 0


def test_oof_and_residuals(reg_analysis, reg_data):
    oof = reg_analysis.oof("Ridge", columns=["sub_grade"])
    assert (oof["residual"] == oof["y_pred"] - oof["y_true"]).all()
    assert oof["sub_grade"].tolist() == reg_analysis.data.loc[oof["row"], "sub_grade"].tolist()
    by_grade = reg_analysis.residuals_by("sub_grade")
    assert by_grade["n"].sum() == len(oof) and set(by_grade.index) == {"A1", "B2", "C3"}
    by_month = reg_analysis.residuals_by("issue_d", freq="M")
    assert by_month.index[0] == "2013-07" and by_month["n"].sum() == len(oof)


# ---------------------------------------------------------------------------------------
# Treino num fold e importância
# ---------------------------------------------------------------------------------------

def test_fold_pipeline_reproduces_the_cross_validation(reg_analysis, reg_data):
    runner = reg_analysis.runner
    ridge = REG_SPECS[2]
    cid = reg_analysis.best_config("Ridge")
    pipe, X_valid, y_valid = runner.fold_pipeline(reg_data, ridge, fold=6)
    oof = runner.oof("Ridge", cid)
    fold6 = oof[oof["fold"] == 6]
    assert len(X_valid) == len(fold6)
    assert pipe.predict(X_valid) == pytest.approx(fold6["y_pred"].to_numpy())
    last, _, _ = runner.fold_pipeline(reg_data, ridge, fold=-1)
    assert last.predict(X_valid) == pytest.approx(pipe.predict(X_valid))


def test_permutation_importance_finds_the_informative_column(reg_analysis):
    importance = reg_analysis.permutation_importance(REG_SPECS[2], n_repeats=2, max_rows=200)
    assert set(importance.index) == set(REG_COLUMNS)
    assert importance.index[0] in ("x1", "term")
    assert importance.loc["x2", "importancia"] < importance.loc["x1", "importancia"]


def test_plots_run(reg_analysis):
    assert len(reg_analysis.plot_folds("mae").get_lines()) == 3
    ax = reg_analysis.plot_tuning("Ridge", REG_SPECS[2].param_grid, "mae")
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0.01", "1000.0"]
    reg_analysis.plot_pred_vs_real("Ridge")
    importance = pd.DataFrame({"importancia": [0.3, 0.1], "desvio": [0.01, 0.01]}, index=["a", "b"])
    CVAnalysis.plot_importance(importance)


# ---------------------------------------------------------------------------------------
# scripts/run_cv.py
# ---------------------------------------------------------------------------------------

def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_cv.py"
    spec = importlib.util.spec_from_file_location("run_cv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_selects_models():
    run_cv = load_script()
    chosen = run_cv.select_specs(REG_SPECS, models="Ridge, Mediana")
    assert [s.name for s in chosen] == ["Mediana", "Ridge"]
    assert [s.name for s in run_cv.select_specs(REG_SPECS, exclude="Ridge")] == ["Mediana", "Mediana por prazo"]
    with pytest.raises(SystemExit, match="desconhecido"):
        run_cv.select_specs(REG_SPECS, models="Foo")
