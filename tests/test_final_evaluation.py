"""
Testes do congelamento e da avaliação final (final_evaluation.py e scripts/freeze_models.py)
"""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import make_column_transformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lending import config
from lending import final_evaluation as fe
from lending.baselines import GradeRateBaseline, TermMedianBaseline
from lending.evaluation import bootstrap_ci, f1_macro_by_threshold
from lending.experiment import CrossValidationRunner, ModelSpec

matplotlib.use("Agg")

STATES = ["CA", "NY", "TX", "IL", "WA", "PR"]


def loans(rng, n, start, end):
    dates = pd.to_datetime(rng.choice(pd.date_range(start, end, freq="MS"), n))
    return pd.DataFrame({"issue_d": dates, "term": rng.choice([36.0, 60.0], n),
                         "x1": rng.normal(size=n), "x2": rng.normal(size=n),
                         "grade": rng.choice(list("ABCDEFG"), n),
                         "addr_state": rng.choice(STATES, n, p=[.3, .2, .2, .15, .14, .01]),
                         "home_ownership": rng.choice(["RENT", "MORTGAGE", "ANY"], n, p=[.5, .49, .01]),
                         "annual_inc": rng.lognormal(11, 0.5, n)})


# ---------------------------------------------------------------------------------------
# Regressão
# ---------------------------------------------------------------------------------------

REG_CATALOG = {s.name: s for s in [
    ModelSpec("Mediana", lambda: DummyRegressor(strategy="median"), preprocess=False),
    ModelSpec("Mediana por prazo", TermMedianBaseline, preprocess=False),
    ModelSpec("Ridge", Ridge, {"alpha": [0.01, 1000.0]})]}


def with_rate(df, rng):
    return df.assign(int_rate=12 + 2 * df["x1"] + 4 * (df["term"] == 60) + rng.normal(0, 1, len(df)))


@pytest.fixture
def reg(tmp_path):
    rng = np.random.default_rng(0)
    train = with_rate(loans(rng, 1_600, "2013-01-01", "2014-12-01"), rng)
    test = with_rate(loans(rng, 3_000, "2015-01-01", "2015-12-01"), rng)
    runner = CrossValidationRunner("regression", results_dir=tmp_path / "results",
                                   oof_dir=tmp_path / "oof", columns=["term", "x1", "x2"],
                                   verbose=False,
                                   make_preprocessor=lambda: StandardScaler().set_output(transform="pandas"))
    runner.run(train, list(REG_CATALOG.values()))
    record = fe.frozen_models("regression", fe.cv_table(runner), REG_CATALOG)
    record = fe.fit_frozen(record, runner, REG_CATALOG, train, models_dir=tmp_path / "models",
                           log=lambda *_: None)
    record = fe.write_frozen(record, path=tmp_path / "results" / "regression_frozen.json")
    evaluation = fe.FinalEvaluation("regression", train, test, frozen=record, runner=runner,
                                    models_dir=tmp_path / "models", cache_dir=tmp_path / "cache")
    return {"runner": runner, "record": record, "train": train, "test": test,
            "evaluation": evaluation, "tmp": tmp_path}


def test_frozen_models_follow_the_cross_validation(reg):
    record, runner = reg["record"], reg["runner"]
    best = runner.best().set_index("model")
    assert [e["model"] for e in record["models"]] == list(best.index)
    assert record["final_model"] == best.index[0] == "Ridge"
    for entry in record["models"]:
        assert entry["config_id"] == best.loc[entry["model"], "config_id"]
        assert entry["cv"]["mae"] == runner.comparison_table().loc[entry["model"], "mae"]
        assert entry["threshold"] is None
    with pytest.raises(ValueError, match="não está na tabela"):
        fe.frozen_models("regression", fe.cv_table(runner), REG_CATALOG, final_model="LightGBM")
    table = fe.cv_table(runner)
    table.loc[table["model"] == "Ridge", "params"] = json.dumps({"alpha": 5.0})
    with pytest.raises(ValueError, match="não está na grelha"):
        fe.frozen_models("regression", table, REG_CATALOG)


def test_fit_frozen_saves_hashes_and_reuses(reg):
    record, tmp = reg["record"], reg["tmp"]
    for entry in record["models"]:
        path = tmp / "models" / entry["file"]
        assert path.exists() and fe.file_sha256(path) == entry["sha256"]
        assert entry["n_train"] == record["train"]["rows"] == 1_600
    messages = []
    again = fe.frozen_models("regression", fe.cv_table(reg["runner"]), REG_CATALOG)
    again = fe.fit_frozen(again, reg["runner"], REG_CATALOG, reg["train"],
                          models_dir=tmp / "models", log=messages.append)
    assert all("já treinado" in m for m in messages)
    assert [e["sha256"] for e in again["models"]] == [e["sha256"] for e in record["models"]]
    saved = json.loads((tmp / "results" / "regression_frozen.json").read_text(encoding="utf-8"))
    assert saved["models"] == record["models"] and "versions" in saved


def test_refit_matches_the_runner(reg):
    entry = next(e for e in reg["record"]["models"] if e["model"] == "Ridge")
    pipe = fe.load_pipeline(entry, reg["tmp"] / "models")
    direct = reg["runner"].refit(reg["train"], REG_CATALOG["Ridge"], cid=entry["config_id"])
    X = reg["test"][reg["runner"].columns]
    assert pipe.predict(X) == pytest.approx(direct.predict(X))


def test_load_pipeline_refuses_a_changed_file(reg):
    entry = reg["record"]["models"][0]
    path = reg["tmp"] / "models" / entry["file"]
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"):
        fe.load_pipeline(entry, reg["tmp"] / "models")


def test_predict_is_cached_and_refuses_changed_choices(reg):
    evaluation, tmp = reg["evaluation"], reg["tmp"]
    first = evaluation.predict()
    assert list(first.columns) == evaluation.models and len(first) == 3_000
    assert evaluation.cache_path.exists()
    shutil.rmtree(tmp / "models")                     # o segundo uso não lê os modelos
    again = fe.FinalEvaluation("regression", reg["train"], reg["test"], frozen=reg["record"],
                               runner=reg["runner"], models_dir=tmp / "models", cache_dir=tmp / "cache")
    pd.testing.assert_frame_equal(again.predict(), first)
    changed = json.loads(json.dumps(reg["record"]))
    changed["models"][0]["params"] = {"alpha": 1000.0}
    with pytest.raises(ValueError, match="mudaram depois"):
        fe.FinalEvaluation("regression", reg["train"], reg["test"], frozen=changed,
                           runner=reg["runner"], cache_dir=tmp / "cache").predict()


def test_metrics_and_bootstrap_match_the_evaluation_module(reg):
    evaluation = reg["evaluation"]
    table = evaluation.metrics_table(n_boot=40)
    assert list(table.index) == evaluation.models
    assert table.loc["Ridge", "final"] == "sim"
    y, p = evaluation.y.to_numpy(), evaluation.predict()["Ridge"].to_numpy()
    assert table.loc["Ridge", "MAE (teste)"] == pytest.approx(mean_absolute_error(y, p), abs=5e-4)
    ci = bootstrap_ci(mean_absolute_error, y, p, n_boot=40)
    assert table.loc["Ridge", "IC 95% MAE"] == f"[{ci['low']:.3f}; {ci['high']:.3f}]"
    paired = evaluation.paired_table(n_boot=40)
    assert paired.loc["Mediana", "MAE: diferença"] < 0 and paired.loc["Mediana", "MAE: clara"] == "sim"


def test_period_subgroups_and_drift(reg):
    evaluation = reg["evaluation"]
    months = evaluation.by_period()
    assert len(months) == 12 and months["n"].sum() == 3_000
    regions = evaluation.subgroup_table("região", min_n=100)
    assert {"Oeste", "Sul", "Nordeste", "Centro-Oeste"} <= set(regions.index)
    assert regions.loc["total", "n"] == 3_000 and regions.drop(index="total")["n"].sum() == 3_000
    housing = evaluation.subgroup_table("home_ownership", min_n=100)
    assert "ANY" not in housing.index and "outros" in housing.index
    income = evaluation.groups("rendimento", min_n=0)
    assert income.nunique() == 5 and income.str.startswith("Q1").mean() == pytest.approx(0.2, abs=0.04)
    drift = evaluation.drift_table(["x1", "int_rate", "grade", "nao_existe"])
    assert list(drift.columns) == ["treino", "teste", "em falta treino (%)", "em falta teste (%)", "PSI"]
    assert set(drift.index) == {"x1", "int_rate", "grade"}


def test_plots_run(reg):
    evaluation = reg["evaluation"]
    evaluation.bootstrap(n_boot=20)
    ax = evaluation.plot_cv_vs_test("mae")
    assert [t.get_text() for t in ax.get_yticklabels()][-1] == "Ridge (final)"
    evaluation.plot_by_period()
    evaluation.plot_subgroups(["região", "rendimento"], ["MAE", "viés"], min_n=50)


# ---------------------------------------------------------------------------------------
# Classificação
# ---------------------------------------------------------------------------------------

FINAL = ModelSpec("Regressão logística final",
                  lambda: LogisticRegression(class_weight="balanced"), {"C": [0.1]})
CLS_CATALOG = {s.name: s for s in [
    ModelSpec("Taxa do grau", GradeRateBaseline, preprocess=False),
    ModelSpec("Regressão logística", lambda: LogisticRegression(class_weight="balanced"),
              {"C": [1.0, 0.001]}), FINAL]}


def with_outcome(df, rng, test=False):
    score = df["x1"] + df["grade"].map(dict(zip("ABCDEFG", range(7)))) / 3 + rng.normal(0, 1, len(df))
    bad = score > np.quantile(score, 0.85)
    status = np.where(bad, "Charged Off", "Fully Paid").astype(object)
    if test:
        status[bad & (rng.random(len(df)) < 0.2)] = "Late (31-120 days)"
        status[~bad & (rng.random(len(df)) < 0.02)] = "Current"
    return df.assign(loan_status=status)


@pytest.fixture
def cls(tmp_path):
    rng = np.random.default_rng(1)
    train = with_outcome(loans(rng, 2_000, "2007-06-01", "2011-12-01"), rng)
    train = train.assign(target=np.where(train["loan_status"] == "Charged Off", "Default", "Good standing"))
    raw_test = with_outcome(loans(rng, 4_000, "2015-01-01", "2015-12-01"), rng, test=True)
    prep = lambda: make_column_transformer((StandardScaler(), ["x1"]),   # noqa: E731
                                           verbose_feature_names_out=False).set_output(transform="pandas")
    results = tmp_path / "results"
    runner = CrossValidationRunner("classification", results_dir=results, oof_dir=tmp_path / "oof",
                                   columns=["grade", "x1"], make_preprocessor=prep, verbose=False)
    runner.run(train, [CLS_CATALOG["Taxa do grau"], CLS_CATALOG["Regressão logística"]])
    runner.comparison_table().to_csv(results / "classification_comparison.csv")   # como no 04c
    pd.DataFrame([{"modelo": FINAL.name, "parâmetros": json.dumps({"C": 0.1}), "limiar": 0.57,
                   "F1 macro (validação cruzada)": "0.570 ± 0.008", "AUC (validação cruzada)": "0.668 ± 0.020",
                   "recall Default (validação cruzada)": "0.257 ± 0.015"}]).to_csv(
        results / "classification_final_model.csv", index=False)
    return {"runner": runner, "train": train, "raw_test": raw_test, "tmp": tmp_path}


def freeze_classification(cls):
    runner, tmp = cls["runner"], cls["tmp"]
    record = fe.frozen_models("classification", fe.cv_table(runner, tmp / "results"), CLS_CATALOG)
    return fe.fit_frozen(record, runner, CLS_CATALOG, cls["train"], models_dir=tmp / "models",
                         log=lambda *_: None)


def test_classification_freeze_uses_the_04c_tables(cls):
    record = freeze_classification(cls)
    assert record["final_model"] == FINAL.name
    entries = {e["model"]: e for e in record["models"]}
    assert entries[FINAL.name]["threshold"] == 0.57
    assert entries[FINAL.name]["cv"] == {"f1_macro": "0.570 ± 0.008", "auc": "0.668 ± 0.020",
                                         "recall_default": "0.257 ± 0.015"}
    runner = cls["runner"]
    best = runner.best().set_index("model")
    for name in ["Taxa do grau", "Regressão logística"]:
        assert entries[name]["threshold"] == pytest.approx(best.loc[name, "threshold"])
        assert entries[name]["config_id"] == best.loc[name, "config_id"]


def test_classification_cv_table_detects_an_outdated_csv(cls):
    path = cls["tmp"] / "results" / "classification_comparison.csv"
    table = pd.read_csv(path)
    table.loc[0, "auc"] = "0.999 ± 0.001"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="não coincide"):
        fe.cv_table(cls["runner"], cls["tmp"] / "results")


def test_test_cohort_keeps_three_classes():
    rng = np.random.default_rng(2)
    df = with_outcome(loans(rng, 600, "2014-07-01", "2015-12-01"), rng, test=True)
    df["int_rate"] = 10.0
    cohort, log = fe.test_cohort("classification", df)
    assert (cohort["term"] == 36).all() and (cohort["issue_d"].dt.year == 2015).all()
    assert not cohort["loan_status"].eq("Current").any()
    at_risk = cohort[fe.THREE_CLASS_COL] == "At risk"
    assert at_risk.any() and (cohort.loc[at_risk, config.CLASS_COL] == "Default").all()
    assert len(log.to_frame()) == 3
    regression, _ = fe.test_cohort("regression", df)
    assert (regression["issue_d"].dt.year == 2015).all() and set(regression["term"]) == {36.0, 60.0}


def test_classification_evaluation(cls):
    tmp = cls["tmp"]
    record = freeze_classification(cls)
    test, _ = fe.test_cohort("classification", cls["raw_test"])
    evaluation = fe.FinalEvaluation("classification", cls["train"], test, frozen=record,
                                    runner=cls["runner"], models_dir=tmp / "models", cache_dir=tmp / "cache")
    table = evaluation.metrics_table(n_boot=30)
    assert table.loc[FINAL.name, "limiar"] == 0.57 and table.loc[FINAL.name, "F1 macro (CV)"] == "0.570 ± 0.008"
    assert table.loc["Taxa do grau", "accuracy (CV)"] != "—"
    y, p = evaluation.y.to_numpy(), evaluation.predict()[FINAL.name].to_numpy()
    f1 = lambda a, b: float(f1_macro_by_threshold(a, b, [0.57])[0])   # noqa: E731
    ci = bootstrap_ci(f1, y, p, n_boot=30)
    assert table.loc[FINAL.name, "IC 95% F1 macro"] == f"[{ci['low']:.3f}; {ci['high']:.3f}]"

    binary, three = evaluation.confusion(), evaluation.confusion(three_classes=True)
    assert binary.to_numpy().sum() == three.to_numpy().sum() == len(test)
    assert list(three.index) == ["Good standing", "At risk", "Default"]
    assert binary.loc["Default"].sum() == three.loc[["At risk", "Default"]].to_numpy().sum()
    assert evaluation.confusion(percent=True).sum(axis=1).to_numpy() == pytest.approx([100, 100], abs=0.2)

    regions = evaluation.subgroup_table("região")
    assert {"recall Default", "falsos alarmes (%)", "Default real (%)"} <= set(regions.columns)
    paired = evaluation.paired_table(n_boot=30)
    assert set(paired.index) == {"Taxa do grau", "Regressão logística"}
    drift = evaluation.drift_table(["target", "x1"])
    assert drift.loc["target", "treino"].endswith("% Default")
    evaluation.plot_by_period()


# ---------------------------------------------------------------------------------------
# Peças pequenas
# ---------------------------------------------------------------------------------------

def test_weighted_auc_matches_sklearn():
    rng = np.random.default_rng(3)
    is_pos = rng.random(500) < 0.3
    p = np.round(rng.random(500) + 0.3 * is_pos, 1)              # com empates
    w = rng.integers(0, 4, 500).astype(float)
    auc = fe._weighted_auc(w, fe._auc_setup(is_pos, p))
    assert auc == pytest.approx(roc_auc_score(is_pos, p, sample_weight=w))


def test_population_stability():
    rng = np.random.default_rng(4)
    a, b = rng.normal(size=20_000), rng.normal(size=20_000)
    assert fe.population_stability(a, b) < 0.01
    assert fe.population_stability(a, b + 1) > 0.25
    cats = pd.Series(rng.choice(["x", "y"], 5_000))
    assert fe.population_stability(cats, cats) == pytest.approx(0)
    assert fe.population_stability(cats, pd.Series(["z"] * 5_000)) > 1


def test_parse_mean_std():
    assert fe.parse_mean_std("0.570 ± 0.008") == (0.57, 0.008)
    mean, std = fe.parse_mean_std("-0.1")
    assert mean == -0.1 and np.isnan(std)
    assert np.isnan(fe.parse_mean_std("—")[0])


@pytest.mark.skipif(shutil.which("git") is None, reason="git não instalado")
def test_tag_status(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    git = lambda *a: subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],   # noqa: E731
                                    cwd=tmp_path, check=True, capture_output=True)
    results = tmp_path / "reports" / "results"
    record = fe.write_frozen({"task": "regression", "models": []}, path=fe.frozen_path("regression", results))
    assert fe.tag_status("regression", results) == "sem git"
    git("init")
    git("add", ".")
    git("commit", "-m", "congelar")
    assert fe.tag_status("regression", results) == "sem tag"
    git("tag", fe.TAG)
    assert fe.tag_status("regression", results) == "ok"
    fe.write_frozen({**record, "final_model": "outro"}, path=fe.frozen_path("regression", results))
    assert fe.tag_status("regression", results) == "diferente"


def test_freeze_script_refuses_after_the_test(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts" / "freeze_models.py"
    spec = importlib.util.spec_from_file_location("freeze_models", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    (tmp_path / "classification.pkl").write_bytes(b"")
    monkeypatch.setattr(script, "TEST_PREDICTIONS_DIR", tmp_path)
    with pytest.raises(SystemExit, match="já foi previsto"):
        script.main(["classification"])
