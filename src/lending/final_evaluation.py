"""
Congelar os modelos e avaliá-los no teste de 2015 (fase 5, bloco 8)

Requisitos do enunciado:
    - "Preserve a final test set for final evaluation": o teste só é aberto depois de as
      escolhas estarem congeladas. O scripts/freeze_models.py grava, para cada tarefa, a
      configuração escolhida de cada modelo, o limiar da classificação e o pipeline treinado
      em todo o treino; o JSON entra no Git com a tag modelos-congelados. O notebook 05 só
      prevê: lê os pipelines (confirmando o hash de cada ficheiro), nunca treina nem escolhe.
    - "Present the mean and the variation of the metric in the cross-validation procedures
      and the mean in the test dataset": metrics_table() junta a média ± desvio da validação
      cruzada, o valor no teste e o intervalo de confiança a 95% por bootstrap.
    - "Discuss selection bias, unequal follow-up, temporal change and fairness":
      by_period() e drift_table() dão a evidência da mudança temporal; subgroup_table() mede
      o erro por região, tipo de habitação e escalão de rendimento (equidade).

Passos:
    1. Congelar (freeze_models.py): frozen_models() lê a tabela de comparação da validação
       cruzada, fit_frozen() treina cada modelo em todo o treino e grava-o em models/<tarefa>/,
       write_frozen() grava reports/results/<tarefa>_frozen.json. Commit e tag.
    2. Avaliar (notebook 05): test_cohort() constrói a coorte de teste com o mesmo
       CohortBuilder; FinalEvaluation prevê o teste uma vez e guarda as previsões; as tabelas
       e os gráficos saem daí.
"""

import hashlib
import json
import re
import subprocess
import time
from importlib import metadata

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lending import config
from lending.cohorts import TARGET_CLASSES, CohortBuilder
from lending.evaluation import (CLASSIFICATION_METRICS, POSITIVE_LABEL, REGRESSION_METRICS,
                                classification_metrics, confusion_table, labels_from_proba,
                                positive_proba, regression_metrics)
from lending.exclusions import ExclusionLog
from lending.experiment import RESULTS_DIR, CrossValidationRunner
from lending.knn import NEIGHBOR_CACHE
from lending.plot_style import CATEGORICAL, MUTED
from lending.storage import PickleStore

TAG = "modelos-congelados"
MODELS_DIR = config.MODELS_DIR                                   # fora do Git
TEST_PREDICTIONS_DIR = config.PROCESSED_DIR / "test_predictions"  # fora do Git
THREE_CLASS_COL = "target_3"                                     # as 3 classes do enunciado
LIBRARIES = ["numpy", "pandas", "scikit-learn", "lightgbm", "tensorflow", "keras"]
TRAIN_PKL = {"regression": config.REG_DEV_PKL, "classification": config.CLS_TRAIN_PKL}

# Métricas do teste com intervalo de confiança (bootstrap)
BOOTSTRAP_METRICS = {"regression": ["mae", "rmse", "r2"],
                     "classification": ["f1_macro", "auc", "recall_default", "accuracy"]}
METRIC_NAMES = {"mae": "MAE", "rmse": "RMSE", "r2": "R²", "f1_macro": "F1 macro", "auc": "AUC",
                "recall_default": "recall Default", "accuracy": "accuracy"}

# Regiões dos EUA (Census Bureau): medem a equidade sem pôr a geografia nos modelos
CENSUS_REGIONS = {
    **dict.fromkeys(["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"], "Nordeste"),
    **dict.fromkeys(["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
                    "Centro-Oeste"),
    **dict.fromkeys(["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS",
                     "TN", "AR", "LA", "OK", "TX"], "Sul"),
    **dict.fromkeys(["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR",
                     "WA"], "Oeste"),
}


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _params_key(params):
    """Parâmetros em texto canónico (os tuplos e as listas ficam iguais)."""
    return json.dumps(params, sort_keys=True, default=str)


def parse_mean_std(text):
    """'0.570 ± 0.008' -> (0.570, 0.008); (nan, nan) se o texto não tiver esse formato."""
    match = re.match(r"\s*(-?[\d.]+)\s*(?:±\s*(-?[\d.]+))?\s*$", str(text))
    if not match:
        return np.nan, np.nan
    return float(match.group(1)), float(match.group(2)) if match.group(2) else np.nan


def frozen_path(task, results_dir=RESULTS_DIR):
    """reports/results/<tarefa>_frozen.json (entra no Git)."""
    return results_dir / f"{task}_frozen.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    """Saída de um comando git na pasta do projeto, ou None se falhar."""
    try:
        out = subprocess.run(["git", *args], capture_output=True, cwd=config.PROJECT_ROOT)
    except OSError:
        return None
    return out.stdout if out.returncode == 0 else None


def _versions():
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def task_catalog(task, include_mlp=True):
    """Os modelos de uma tarefa pelo nome (o catálogo e, na classificação, o modelo final)."""
    from lending.model_catalog import (classification_specs, final_classification_spec,
                                       regression_specs)
    if task == "regression":
        specs = regression_specs(include_mlp)
    else:
        specs = classification_specs(include_mlp) + [final_classification_spec()]
    return {spec.name: spec for spec in specs}


# ---------------------------------------------------------------------------------------
# 1. Congelar
# ---------------------------------------------------------------------------------------

def classification_cv_table(runner, results_dir=RESULTS_DIR):
    """
    Tabela da validação cruzada da classificação, a partir dos CSV do notebook 04c

    O limiar de cada modelo foi escolhido nas previsões fora do fold, que só existem na máquina
    onde a grelha correu. Os CSV do 04c (no Git) já têm esse limiar, por isso o congelamento
    pode ser feito em qualquer máquina. Confirma que a tabela corresponde ao
    classification_cv.csv: cada configuração tem de estar completa e a AUC média (que não
    depende do limiar) tem de coincidir.

    Returns:
        pd.DataFrame: Uma linha por modelo (o modelo final no fim), com model, params,
            threshold e as métricas em texto "média ± desvio"
    """
    table = pd.read_csv(results_dir / "classification_comparison.csv", dtype={"params": str})
    results = runner.results()
    n_folds = results["fold"].nunique()
    keys = results["params"].map(lambda p: _params_key(json.loads(p)))
    for row in table.itertuples():
        rows = results[(results["model"] == row.model) & (keys == _params_key(json.loads(row.params)))]
        if len(rows) != n_folds:
            raise ValueError(f"{row.model} {row.params}: {len(rows)} de {n_folds} folds no "
                             "classification_cv.csv. O classification_comparison.csv está "
                             "desatualizado: correr de novo o 04c.")
        auc_mean, _ = parse_mean_std(row.auc)
        if abs(rows["auc"].astype(float).mean() - auc_mean) > 6e-4:
            raise ValueError(f"{row.model}: a AUC do classification_comparison.csv não coincide "
                             "com o classification_cv.csv. Correr de novo o 04c.")

    final = pd.read_csv(results_dir / "classification_final_model.csv").iloc[0]
    final_row = {"model": final["modelo"], "params": final["parâmetros"],
                 "threshold": float(final["limiar"])}
    for metric, label in [("f1_macro", "F1 macro"), ("auc", "AUC"),
                          ("recall_default", "recall Default")]:
        final_row[metric] = final[f"{label} (validação cruzada)"]
    return pd.concat([table, pd.DataFrame([final_row])], ignore_index=True)


def cv_table(runner, results_dir=RESULTS_DIR):
    """A tabela de comparação da validação cruzada de uma tarefa (a fonte das escolhas)."""
    if runner.task.is_classification:
        return classification_cv_table(runner, results_dir)
    return runner.comparison_table().reset_index()


def frozen_models(task, table, catalog, final_model=None):
    """
    O registo das escolhas da validação cruzada (ainda sem os pipelines treinados)

    Parameters:
        task (str): "regression" ou "classification"
        table (pd.DataFrame): cv_table(): uma linha por modelo, a melhor primeiro
        catalog (dict[str, ModelSpec]): task_catalog(task)
        final_model (str | None): O modelo que segue para o deployment. Por omissão, na
            regressão o melhor da validação cruzada e na classificação o último da tabela
            (o modelo final do 04c)

    Returns:
        dict: O registo, pronto para fit_frozen()
    """
    metrics = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    entries = []
    for row in table.to_dict("records"):
        spec = catalog.get(row["model"])
        if spec is None:
            raise KeyError(f"{row['model']!r} não está no catálogo da tarefa.")
        wanted = _params_key(json.loads(row["params"]))
        matches = [(cid, p) for cid, p in spec.configs() if _params_key(p) == wanted]
        if len(matches) != 1:
            raise ValueError(f"{row['model']}: a configuração {row['params']} não está na grelha "
                             "do catálogo (o catálogo mudou depois da validação cruzada?).")
        cid, params = matches[0]
        entries.append({
            "model": spec.name, "config_id": cid, "params": json.loads(_params_key(params)),
            "preprocess": spec.preprocess, "max_train_rows": spec.max_train_rows,
            "drop_features": list(spec.drop_features),
            "threshold": float(row["threshold"]) if task == "classification" else None,
            "cv": {m: row[m] for m in metrics if isinstance(row.get(m), str)},
        })
    names = [e["model"] for e in entries]
    if final_model is None:
        final_model = names[-1] if task == "classification" else names[0]
    if final_model not in names:
        raise ValueError(f"O modelo final {final_model!r} não está na tabela: {names}")
    return {"task": task, "tag": TAG, "final_model": final_model, "models": entries}


def _saved_payload(path, task, entry):
    """O conteúdo de um modelo já gravado, se for desta configuração (senão, None)."""
    try:
        payload = joblib.load(path)
    except Exception:  # ficheiro incompleto ou de outra versão: treina-se de novo
        return None
    ok = (isinstance(payload, dict) and payload.get("task") == task
          and payload.get("config_id") == entry["config_id"])
    return payload if ok else None


def fit_frozen(record, runner, catalog, train, models_dir=MODELS_DIR, refit=False, log=print):
    """
    Treina cada modelo congelado em todo o treino e grava o pipeline (nunca usa o teste)

    Cada ficheiro models/<tarefa>/<modelo>.joblib guarda o pipeline e o config_id. Se o script
    for interrompido, os modelos já gravados com a mesma configuração não se treinam de novo
    (a não ser com refit=True).

    Returns:
        dict: O registo, com o ficheiro, o hash, as linhas de treino e o tempo de cada modelo
    """
    task = record["task"]
    folder = models_dir / task
    folder.mkdir(parents=True, exist_ok=True)
    data = runner.prepare(train)
    for entry in record["models"]:
        spec = catalog[entry["model"]]
        path = folder / f"{_slug(entry['model'])}.joblib"
        payload = None if refit or not path.exists() else _saved_payload(path, task, entry)
        if payload is None:
            log(f"{entry['model']}: a treinar em {len(data):,} empréstimos...")
            start = time.perf_counter()
            pipe = runner.refit(data, spec, cid=entry["config_id"])
            payload = {"task": task, "model": entry["model"], "config_id": entry["config_id"],
                       "fit_seconds": round(time.perf_counter() - start, 1), "pipeline": pipe}
            joblib.dump(payload, path, compress=3)
        else:
            log(f"{entry['model']}: já treinado ({path.name})")
        entry.update({"file": f"{task}/{path.name}", "sha256": file_sha256(path),
                      "size_mb": round(path.stat().st_size / 2**20, 1),
                      "fit_seconds": payload["fit_seconds"],
                      "n_train": int(min(len(data), spec.max_train_rows or len(data)))})
    dates = data[runner.task.date_col]
    record["train"] = {"rows": int(len(data)), "first": str(dates.min().date()),
                       "last": str(dates.max().date())}
    return record


def write_frozen(record, path=None):
    """Grava o registo com a data, o commit, as versões das bibliotecas e mudanças por gravar."""
    commit = _git("rev-parse", "--short", "HEAD")
    dirty = _git("status", "--porcelain", "--", "src", "scripts")
    record = {**record, "frozen_at": pd.Timestamp.now().isoformat(timespec="seconds"),
              "git_commit": commit.decode().strip() if commit else None,
              "uncommitted_code_changes": bool(dirty.strip()) if dirty is not None else None,
              "versions": _versions()}
    path = path or frozen_path(record["task"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return record


def load_frozen(task, results_dir=RESULTS_DIR):
    path = frozen_path(task, results_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe: primeiro, python scripts/freeze_models.py {task}")
    return json.loads(path.read_text(encoding="utf-8"))


def tag_status(task, results_dir=RESULTS_DIR, tag=TAG):
    """
    Confirma que o JSON congelado no disco é o mesmo que está na tag

    Compara o conteúdo (e não os bytes), por isso as mudanças de fim de linha do Git no
    Windows não contam.

    Returns:
        str: "ok", "sem git", "sem tag" (a tag não existe ou não tem o ficheiro) ou
            "diferente" (o JSON mudou depois da tag)
    """
    path = frozen_path(task, results_dir)
    try:
        relative = path.resolve().relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return "sem git"
    if _git("rev-parse", "--git-dir") is None:
        return "sem git"
    tagged = _git("show", f"{tag}:{relative}")
    if tagged is None:
        return "sem tag"
    return "ok" if json.loads(tagged.decode("utf-8")) == load_frozen(task, results_dir) else "diferente"


def load_pipeline(entry, models_dir=MODELS_DIR):
    """O pipeline congelado de um modelo, depois de confirmar o hash do ficheiro."""
    path = models_dir / entry["file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} não existe (os modelos não estão no Git: copiar a pasta "
                                "models/ de quem congelou).")
    if file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{path} não é o ficheiro congelado (o hash mudou).")
    payload = joblib.load(path)
    if payload["config_id"] != entry["config_id"]:
        raise ValueError(f"{path}: a configuração não coincide com a congelada.")
    return payload["pipeline"]


# ---------------------------------------------------------------------------------------
# 2. Coorte de teste e deriva
# ---------------------------------------------------------------------------------------

def test_cohort(task, test_df=None):
    """
    A coorte de teste (2015), com as mesmas regras da de treino

    Na classificação acrescenta THREE_CLASS_COL, com as 3 classes do enunciado (Good
    standing, At risk, Default), para a tabela 3 x 2. O alvo binário continua em
    config.CLASS_COL, com os At risk contados como Default (cohorts.BINARY_CLASSES).

    Parameters:
        task (str): "regression" ou "classification"
        test_df (pd.DataFrame | None): O teste limpo; por omissão, config.TEST_PKL

    Returns:
        tuple[pd.DataFrame, ExclusionLog]: coorte e registo das exclusões
    """
    df = (test_df if test_df is not None
          else PickleStore.load(config.TEST_PKL, expected_type=pd.DataFrame))
    log = ExclusionLog(f"Teste final (2015): {task}")
    builder = CohortBuilder()
    if task == "regression":
        return builder.regression(df, log, partition="test"), log
    cohort = builder.classification(df, log, partition="test")
    cohort[THREE_CLASS_COL] = cohort["loan_status"].map(TARGET_CLASSES)
    return cohort, log


def population_stability(train, test, bins=10):
    """
    PSI (population stability index) de uma variável entre o treino e o teste

    Numérica: os intervalos são os decis do treino. Categórica: as categorias do treino (as
    novas juntam-se numa só). Os valores em falta são uma categoria própria.
    PSI = soma((p_teste - p_treino) x ln(p_teste / p_treino)). Convenção habitual em risco de
    crédito: < 0,1 estável, 0,1 a 0,25 mudança moderada, > 0,25 mudança grande.
    """
    train, test = pd.Series(train).reset_index(drop=True), pd.Series(test).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(train) and not pd.api.types.is_bool_dtype(train):
        edges = np.unique(np.nanquantile(train.astype(float), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf

        def shares(values):
            values = values.astype(float)
            counts = pd.cut(values, edges, include_lowest=True).value_counts(sort=False).to_numpy()
            return np.append(counts, values.isna().sum()) / len(values)
    else:
        categories = list(train.dropna().astype(str).unique())

        def shares(values):
            text = values.astype(str).where(values.notna(), "<em falta>")
            text = text.where(text.isin(categories + ["<em falta>"]), "<nova>")
            return text.value_counts().reindex(categories + ["<em falta>", "<nova>"],
                                               fill_value=0).to_numpy() / len(values)

    p_train, p_test = (np.clip(shares(s), 1e-4, None) for s in (train, test))
    return float(np.sum((p_test - p_train) * np.log(p_test / p_train)))


# ---------------------------------------------------------------------------------------
# Bootstrap vetorizado: cada reamostragem é um vetor de pesos (quantas vezes cada linha sai)
# ---------------------------------------------------------------------------------------

def _auc_setup(is_pos, p):
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    starts = np.flatnonzero(np.r_[True, sorted_p[1:] != sorted_p[:-1]])
    return order, starts, is_pos[order].astype(float)


def _weighted_auc(w, setup):
    """AUC de Mann-Whitney com pesos (empates contam metade): igual à AUC das linhas repetidas."""
    order, starts, pos = setup
    if len(order) == 0:
        return np.nan
    ws = w[order]
    pos_g = np.add.reduceat(ws * pos, starts)
    neg_g = np.add.reduceat(ws * (1 - pos), starts)
    total_pos, total_neg = pos_g.sum(), neg_g.sum()
    if total_pos == 0 or total_neg == 0:
        return np.nan
    below = np.cumsum(neg_g) - neg_g
    return float((pos_g * (below + 0.5 * neg_g)).sum() / (total_pos * total_neg))


class _MetricsFromWeights:
    """As métricas do teste de um modelo para um vetor de pesos (as linhas da reamostragem)."""

    def __init__(self, task, y, p, threshold=None):
        self.task, self.n = task, float(len(y))
        if task == "regression":
            y, p = np.asarray(y, float), np.asarray(p, float)
            self.y, self.y2 = y, y ** 2
            self.abs_err, self.sq_err = np.abs(p - y), (p - y) ** 2
        else:
            is_pos = np.asarray(y) == POSITIVE_LABEL
            pred = np.asarray(p) >= threshold
            self.masks = [m.astype(float) for m in
                          (pred & is_pos, pred & ~is_pos, ~pred & is_pos, ~pred & ~is_pos)]
            self.auc = _auc_setup(is_pos, np.asarray(p, float))

    def __call__(self, w):
        if self.task == "regression":
            sse = w @ self.sq_err
            mean_y = (w @ self.y) / self.n
            sst = w @ self.y2 - self.n * mean_y ** 2
            return {"mae": (w @ self.abs_err) / self.n, "rmse": np.sqrt(sse / self.n),
                    "r2": 1 - sse / sst if sst > 0 else np.nan}
        tp, fp, fn, tn = (w @ m for m in self.masks)
        f1_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn > 0 else 0.0
        f1_neg = 2 * tn / (2 * tn + fn + fp) if 2 * tn + fn + fp > 0 else 0.0
        return {"f1_macro": (f1_pos + f1_neg) / 2, "auc": _weighted_auc(w, self.auc),
                "recall_default": tp / (tp + fn) if tp + fn > 0 else np.nan,
                "accuracy": (tp + tn) / self.n}


# ---------------------------------------------------------------------------------------
# 3. Avaliação no teste
# ---------------------------------------------------------------------------------------

class FinalEvaluation:
    """
    Os modelos congelados de uma tarefa no teste de 2015

    Parameters:
        task (str): "regression" ou "classification"
        train (pd.DataFrame): Coorte de treino (para os quintis de rendimento e a deriva)
        test (pd.DataFrame): Coorte de teste (test_cohort)
        frozen (dict | None): Registo congelado; por omissão, load_frozen(task)
        runner (CrossValidationRunner | None): Dá as colunas de entrada e a ordenação
        models_dir, cache_dir (Path): Pipelines congelados e previsões guardadas

    Methods:
        predict(), test_metrics(), bootstrap(), metrics_table(), paired_table(), confusion(),
        by_period(), subgroup_table(), drift_table(), plot_cv_vs_test(), plot_by_period(),
        plot_subgroups()
    """

    def __init__(self, task, train, test, frozen=None, runner=None, models_dir=MODELS_DIR,
                 cache_dir=TEST_PREDICTIONS_DIR):
        self.runner = runner or CrossValidationRunner(task, verbose=False)
        self.task = self.runner.task
        self.frozen = frozen or load_frozen(self.task.name)
        if self.frozen["task"] != self.task.name:
            raise ValueError("O registo congelado é de outra tarefa.")
        self.entries = {e["model"]: e for e in self.frozen["models"]}
        self.models = list(self.entries)              # pela ordem da validação cruzada
        self.final_model = self.frozen["final_model"]
        self.train = self.runner.prepare(train)
        self.test = self.runner.prepare(test)
        self.models_dir = models_dir
        self.cache_path = cache_dir / f"{self.task.name}.pkl"
        self.fingerprint = hashlib.sha1(json.dumps(self.frozen["models"], sort_keys=True)
                                        .encode("utf-8")).hexdigest()[:12]
        self._predictions, self.timing, self._boot = None, None, None

    @property
    def y(self):
        return self.test[self.task.target]

    def threshold(self, model):
        return self.entries[model]["threshold"]

    # ------------------------------------------------------------------ previsões

    def predict(self):
        """
        Previsões de todos os modelos no teste: calculadas uma vez e depois lidas do ficheiro

        Returns:
            pd.DataFrame: Uma coluna por modelo (taxa prevista ou P(Default)), pela ordem do teste

        Raises:
            ValueError: Se as escolhas congeladas ou a coorte de teste mudaram depois de o
                teste ter sido previsto
        """
        if self._predictions is not None:
            return self._predictions
        y = self.y.to_numpy()
        if self.cache_path.exists():
            cached = PickleStore.load(self.cache_path)
            if cached["fingerprint"] != self.fingerprint:
                raise ValueError("As escolhas congeladas mudaram depois de o teste ter sido "
                                 "previsto. Mudar um modelo depois de ver o teste invalida a "
                                 "avaliação final. Para repetir de propósito, apagar "
                                 f"{self.cache_path} e explicar a mudança no relatório.")
            if not np.array_equal(cached["y_true"], y):
                raise ValueError("A coorte de teste não é a mesma das previsões guardadas.")
            self._predictions, self.timing = cached["predictions"], cached["timing"]
            return self._predictions

        X_test = self.test[self.runner.columns]
        predictions, timing = {}, []
        for name in self.models:
            pipe = load_pipeline(self.entries[name], self.models_dir)
            start = time.perf_counter()
            if self.task.is_classification:
                predictions[name] = positive_proba(pipe, X_test)
            else:
                predictions[name] = np.asarray(pipe.predict(X_test), dtype=float)
            NEIGHBOR_CACHE.clear()          # a pesquisa do kNN no teste ocupa centenas de MB
            timing.append({"modelo": name, "treino (s)": self.entries[name]["fit_seconds"],
                           "previsão do teste (s)": round(time.perf_counter() - start, 1)})
        self._predictions = pd.DataFrame(predictions)
        self.timing = pd.DataFrame(timing).set_index("modelo")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        PickleStore.save({"fingerprint": self.fingerprint, "y_true": y,
                          "predictions": self._predictions, "timing": self.timing,
                          "created_at": pd.Timestamp.now().isoformat(timespec="seconds")},
                         self.cache_path)
        return self._predictions

    # ------------------------------------------------------------------ métricas

    def test_metrics(self):
        """Todas as métricas no teste, por modelo (na classificação, com o limiar congelado)."""
        P, rows = self.predict(), {}
        for name in self.models:
            if self.task.is_classification:
                rows[name] = classification_metrics(self.y, P[name], self.threshold(name))
            else:
                rows[name] = regression_metrics(self.y, P[name])
        return pd.DataFrame(rows).T

    def bootstrap(self, n_boot=1000, seed=config.RANDOM_SEED):
        """
        Métricas do teste em n_boot reamostragens das linhas, as mesmas para todos os modelos

        As reamostragens são as de evaluation.bootstrap_ci com a mesma semente (cada uma
        escolhe n linhas com reposição); aqui são pesos (quantas vezes sai cada linha), o que
        dá os mesmos valores muito mais depressa. Como todos os modelos usam as mesmas
        reamostragens, a diferença entre dois modelos em cada uma é o bootstrap emparelhado.

        Returns:
            dict[str, pd.DataFrame]: Por modelo, uma linha por reamostragem
        """
        if self._boot is not None and self._boot[0] == (n_boot, seed):
            return self._boot[1]
        P = self.predict()
        calculators = {name: _MetricsFromWeights(self.task.name, self.y, P[name].to_numpy(),
                                                 self.threshold(name)) for name in self.models}
        rng, n = np.random.default_rng(seed), len(self.y)
        values = {name: [] for name in self.models}
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
            for name, calculate in calculators.items():
                values[name].append(calculate(w))
        boot = {name: pd.DataFrame(v) for name, v in values.items()}
        self._boot = ((n_boot, seed), boot)
        return boot

    def metrics_table(self, n_boot=1000, decimals=3):
        """
        Tabela de comparação no teste: a validação cruzada (média ± desvio), o valor no teste e
        o intervalo de confiança a 95%, pela ordem da validação cruzada

        Returns:
            pd.DataFrame: Uma linha por modelo, com o modelo final assinalado
        """
        test, boot, rows = self.test_metrics(), self.bootstrap(n_boot), []
        for name in self.models:
            entry = self.entries[name]
            row = {"modelo": name, "final": "sim" if name == self.final_model else "",
                   "configuração": (json.dumps(entry["params"], ensure_ascii=False)
                                    if entry["params"] else "—"),
                   "n treino": entry["n_train"]}
            if self.task.is_classification:
                row["limiar"] = entry["threshold"]
            for metric in BOOTSTRAP_METRICS[self.task.name]:
                low, high = np.nanpercentile(boot[name][metric], [2.5, 97.5])
                label = METRIC_NAMES[metric]
                row[f"{label} (CV)"] = entry["cv"].get(metric, "—")
                row[f"{label} (teste)"] = round(float(test.loc[name, metric]), decimals)
                row[f"IC 95% {label}"] = f"[{low:.{decimals}f}; {high:.{decimals}f}]"
            row["treino (s)"] = entry["fit_seconds"]
            rows.append(row)
        return pd.DataFrame(rows).set_index("modelo")

    def paired_table(self, reference=None, n_boot=1000):
        """
        Diferença entre o modelo de referência (por omissão, o final) e cada um dos outros no
        teste, com o bootstrap emparelhado (as mesmas reamostragens nos dois)

        Regressão: MAE(referência) - MAE(outro): negativo = a referência erra menos.
        Classificação: F1 macro e AUC (referência - outro): positivo = a referência é melhor.
        A diferença é "clara" quando o intervalo de 95% não contém o 0.
        """
        reference = reference or self.final_model
        boot, test = self.bootstrap(n_boot), self.test_metrics()
        metrics = ["f1_macro", "auc"] if self.task.is_classification else ["mae"]
        rows = []
        for other in self.models:
            if other == reference:
                continue
            row = {"outro modelo": other}
            for metric in metrics:
                diff = boot[reference][metric] - boot[other][metric]
                low, high = np.nanpercentile(diff, [2.5, 97.5])
                label = METRIC_NAMES[metric]
                row[f"{label}: diferença"] = round(float(test.loc[reference, metric]
                                                         - test.loc[other, metric]), 4)
                row[f"IC 95% {label}"] = f"[{low:.4f}; {high:.4f}]"
                row[f"{label}: clara"] = "sim" if low > 0 or high < 0 else "não"
            rows.append(row)
        return pd.DataFrame(rows).set_index("outro modelo")

    def confusion(self, model=None, three_classes=False, percent=False):
        """
        Matriz de confusão no teste com o limiar congelado (linhas reais, colunas previstas)

        three_classes=True: linhas com as 3 classes do enunciado (Good standing, At risk,
        Default), para se ver onde caem os At risk. percent=True: % de cada linha.
        """
        model = model or self.final_model
        predicted = labels_from_proba(self.predict()[model], self.threshold(model))
        if three_classes:
            table = confusion_table(self.test[THREE_CLASS_COL], predicted,
                                    row_labels=["Good standing", "At risk", "Default"])
        else:
            table = confusion_table(self.y, predicted)
        if percent:
            return (100 * table.div(table.sum(axis=1).replace(0, np.nan), axis=0)).round(1)
        return table

    # ------------------------------------------------------------------ tempo, equidade, deriva

    def _frame(self, model):
        return pd.DataFrame({"y": self.y.to_numpy(), "p": self.predict()[model].to_numpy()})

    def _summary(self, part, model):
        """As medidas de um grupo de linhas do teste (um mês, uma região, ...)."""
        if self.task.is_classification:
            is_pos = part["y"].to_numpy() == POSITIVE_LABEL
            p = part["p"].to_numpy()
            flagged = p >= self.threshold(model)
            return {"n": len(part), "Default real (%)": 100 * is_pos.mean(),
                    "Default previstos (%)": 100 * flagged.mean(),
                    "P(Default) média (%)": 100 * p.mean(),
                    "recall Default": flagged[is_pos].mean() if is_pos.any() else np.nan,
                    "falsos alarmes (%)": 100 * flagged[~is_pos].mean() if (~is_pos).any() else np.nan,
                    "AUC": _weighted_auc(np.ones(len(p)), _auc_setup(is_pos, p))}
        residual = part["p"] - part["y"]
        return {"n": len(part), "taxa real média": part["y"].mean(),
                "taxa prevista média": part["p"].mean(), "MAE": residual.abs().mean(),
                "viés": residual.mean()}

    def by_period(self, model=None, freq="M"):
        """
        O modelo ao longo de 2015, por mês de emissão

        Regressão: taxa real e prevista, MAE e viés (> 0 = o modelo prevê a mais).
        Classificação: taxa real de Default, % previstos, P(Default) média e recall.
        """
        model = model or self.final_model
        frame = self._frame(model)
        frame["período"] = pd.DatetimeIndex(self.test[self.task.date_col]).to_period(freq).astype(str)
        rows = {period: self._summary(part, model) for period, part in frame.groupby("período")}
        return pd.DataFrame(rows).T.round(3)

    def groups(self, by, min_n=1000):
        """
        Grupos do teste para a análise de equidade

        by: "região" (regiões dos EUA a partir de addr_state), "rendimento" (quintis de
        annual_inc com os limites do TREINO) ou o nome de uma coluna do teste. Os grupos com
        menos de min_n empréstimos juntam-se em "outros".
        """
        if by == "região":
            groups = self.test["addr_state"].map(CENSUS_REGIONS).fillna("outros")
        elif by == "rendimento":
            edges = np.unique(np.nanquantile(self.train["annual_inc"], np.linspace(0, 1, 6)))
            inner = edges[1:-1]
            labels = ([f"Q1 (< {inner[0] / 1000:.0f} mil $)"]
                      + [f"Q{i + 2} ({a / 1000:.0f}–{b / 1000:.0f} mil $)"
                         for i, (a, b) in enumerate(zip(inner, inner[1:]))]
                      + [f"Q{len(inner) + 1} (≥ {inner[-1] / 1000:.0f} mil $)"])
            groups = (pd.cut(self.test["annual_inc"], [-np.inf, *inner, np.inf], labels=labels,
                             right=False).astype(str).replace("nan", "em falta"))
        else:
            groups = self.test[by].astype(str)
        counts = groups.value_counts()
        small = counts.index[counts < min_n]
        return groups.where(~groups.isin(small), "outros")

    def subgroup_table(self, by, model=None, min_n=1000):
        """
        O modelo final por grupo (equidade), com uma linha "total" no fim

        Regressão: n, taxa real e prevista, MAE e viés (> 0 = cobra a mais ao grupo).
        Classificação: n, taxa real de Default, % previstos como Default, recall de Default
        (defaults apanhados) e falsos alarmes (% dos bons classificados como Default).
        """
        model = model or self.final_model
        frame = self._frame(model)
        frame["grupo"] = self.groups(by, min_n).to_numpy()
        rows = {group: self._summary(part, model) for group, part in frame.groupby("grupo")}
        rows["total"] = self._summary(frame, model)
        return pd.DataFrame(rows).T.round(3)

    def drift_table(self, columns):
        """
        Mudança das variáveis entre o treino e o teste: média (ou categoria mais frequente),
        % em falta e PSI

        Returns:
            pd.DataFrame: Uma linha por coluna, a de maior PSI primeiro
        """
        def describe(s, column):
            if column == self.task.target and self.task.is_classification:
                return f"{100 * (s == POSITIVE_LABEL).mean():.1f}% Default"
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                return f"{s.mean():.2f}"
            top = s.mode().iloc[0]
            return f"{top} ({100 * (s == top).mean():.0f}%)"

        rows = []
        for column in [c for c in columns if c in self.train.columns and c in self.test.columns]:
            train, test = self.train[column], self.test[column]
            rows.append({"coluna": column, "treino": describe(train, column),
                         "teste": describe(test, column),
                         "em falta treino (%)": round(100 * train.isna().mean(), 1),
                         "em falta teste (%)": round(100 * test.isna().mean(), 1),
                         "PSI": round(population_stability(train, test), 3)})
        return pd.DataFrame(rows).set_index("coluna").sort_values("PSI", ascending=False)

    # ------------------------------------------------------------------ gráficos

    def plot_cv_vs_test(self, metric=None, ax=None):
        """Métrica de cada modelo: validação cruzada (média ± desvio) e teste (com IC 95%)."""
        metric = metric or self.task.metric
        boot, test = self.bootstrap(), self.test_metrics()
        if ax is None:
            _, ax = plt.subplots(figsize=(6.5, 0.45 * len(self.models) + 1.2))
        order = list(reversed(self.models))
        for i, name in enumerate(order):
            mean, std = parse_mean_std(self.entries[name]["cv"].get(metric))
            if not np.isnan(mean):
                ax.errorbar(mean, i + 0.15, xerr=0 if np.isnan(std) else std, fmt="o",
                            color=CATEGORICAL[0], capsize=3)
            low, high = np.nanpercentile(boot[name][metric], [2.5, 97.5])
            value = test.loc[name, metric]
            ax.errorbar(value, i - 0.15, xerr=[[value - low], [high - value]], fmt="s",
                        color=CATEGORICAL[1], capsize=3)
        ax.errorbar([], [], fmt="o", color=CATEGORICAL[0], label="validação cruzada (média ± desvio)")
        ax.errorbar([], [], fmt="s", color=CATEGORICAL[1], label="teste 2015 (IC 95%)")
        ax.set_yticks(range(len(order)), [f"{m} (final)" if m == self.final_model else m
                                          for m in order])
        ax.set_xlabel(METRIC_NAMES.get(metric, metric))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
        return ax

    def plot_by_period(self, model=None, axes=None):
        """Regressão: MAE e viés por mês. Classificação: % real e % prevista de Default por mês."""
        table = self.by_period(model)
        if axes is None:
            _, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        x = list(range(len(table)))
        if self.task.is_classification:
            axes[0].plot(x, table["Default real (%)"], marker="o", color=CATEGORICAL[0], label="real")
            axes[0].plot(x, table["Default previstos (%)"], marker="s", color=CATEGORICAL[1],
                         label="previstos como Default")
            axes[0].plot(x, table["P(Default) média (%)"], ls="--", color=MUTED,
                         label="P(Default) média")
            axes[0].set_ylabel("Default (%)")
            axes[0].set_ylim(0, 1.2 * table[["Default real (%)", "Default previstos (%)",
                                             "P(Default) média (%)"]].to_numpy().max())
            axes[0].legend(frameon=False, loc="lower center", ncol=3)
            axes[1].plot(x, table["recall Default"], marker="o", color=CATEGORICAL[0])
            axes[1].set_ylabel("recall de Default")
        else:
            axes[0].plot(x, table["MAE"], marker="o", color=CATEGORICAL[0])
            axes[0].set_ylabel("MAE (p.p.)")
            axes[1].axhline(0, color=MUTED, lw=0.8)
            axes[1].plot(x, table["viés"], marker="o", color=CATEGORICAL[1])
            axes[1].set_ylabel("viés: previsto − real (p.p.)")
        for ax in axes:
            ax.set_xticks(x, [p[-2:] for p in table.index])
            ax.set_xlabel("mês de emissão (2015)")
        return axes

    def plot_subgroups(self, groupings, columns, model=None, min_n=1000):
        """Uma linha de painéis por agrupamento e uma coluna por medida; a tracejado, o total."""
        fig, axes = plt.subplots(len(groupings), len(columns), squeeze=False,
                                 figsize=(4.4 * len(columns), 1.9 * len(groupings) + 0.6))
        for r, by in enumerate(groupings):
            table = self.subgroup_table(by, model, min_n)
            groups, total = table.drop(index="total"), table.loc["total"]
            for c, column in enumerate(columns):
                ax = axes[r, c]
                ax.barh(range(len(groups)), groups[column], color=CATEGORICAL[0])
                ax.axvline(total[column], color=CATEGORICAL[1], ls="--", lw=1)
                ax.set_yticks(range(len(groups)), [f"{g} (n = {int(n):,})".replace(",", " ")
                                                   for g, n in groups["n"].items()])
                ax.invert_yaxis()
                if r == 0:
                    ax.set_title(column)
                if c == 0:
                    ax.set_ylabel(by)
        fig.tight_layout()
        return fig
