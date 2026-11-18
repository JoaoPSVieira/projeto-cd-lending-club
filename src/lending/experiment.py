"""
Validação cruzada temporal de todos os modelos - Fases 4 e 5

Requisitos do enunciado cumpridos:
    - "Use the same eligible evaluation cohort and split definitions for all models within a
      task": todos os modelos de uma tarefa passam pelo mesmo runner, com as mesmas coortes,
      os mesmos folds (janela expansível) e o mesmo pré-processamento.
    - "Report training sample sizes, tuning procedures, and any computational limitations":
      cada linha de resultados guarda o tamanho do treino, a configuração e os tempos.
    - "Present the mean and the variation of the metric in the cross-validation procedures":
      summary() devolve a média e o desvio-padrão de cada métrica nos folds.

Como funciona (CrossValidationRunner.run):
    1. Ordena as linhas por issue_d (ordenação estável). É o contrato com os modelos com
       paragem antecipada (MLP e LightGBM): as últimas linhas do treino de cada fold são os
       empréstimos mais recentes. Os pickles NÃO vêm ordenados, por isso este passo é
       obrigatório e não depende de quem chama o runner.
    2. Divide em folds com ExpandingWindowSplit (6 trimestrais na regressão, 3 anuais na
       classificação).
    3. Em cada fold, ajusta o pré-processamento (FeatureBuilder + build_preprocessor) UMA vez,
       só no treino do fold, e reutiliza-o em todas as configurações. Dá o mesmo resultado que
       um Pipeline completo por configuração, porque o pré-processamento não depende do modelo,
       e poupa centenas de ajustes repetidos.
    4. Para cada configuração da grelha: amostra o treino se o modelo o pedir (kNN da
       regressão: 50 000 linhas, semente 42), ajusta, prevê a validação do fold, mede tempos
       e métricas, grava as previsões fora do fold e acrescenta uma linha ao CSV.
    5. Retoma: as combinações (modelo, configuração, fold) que já estão no CSV são saltadas.
       Uma grelha interrompida continua onde parou.

Ficheiros:
    reports/results/<tarefa>_cv.csv           uma linha por modelo x configuração x fold (entra no Git)
    data/processed/oof/<tarefa>/<modelo>/...  previsões fora do fold (fora do Git)

Limiar da classificação:
    Nas linhas do CSV, as métricas com limiar usam 0,5. O summary() escolhe, para cada
    configuração, o limiar que maximiza o F1 macro nas previsões fora do fold de todos os
    folds juntos, e recalcula as métricas de cada fold com esse limiar.

Exemplo:
    runner = CrossValidationRunner("classification")
    runner.run(data, classification_specs())          # model_catalog.py
    runner.summary()                                  # todas as configurações
    runner.comparison_table()                         # melhor configuração de cada modelo
    final = runner.refit(data, spec)                  # melhor configuração, todo o treino
"""

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from lending import config
from lending.evaluation import (CLASSIFICATION_METRICS, GREATER_IS_BETTER, REGRESSION_METRICS,
                                best_threshold, classification_metrics, format_mean_std,
                                positive_proba, regression_metrics)
from lending.feature_engineering import FeatureBuilder, add_engineered, input_columns
from lending.features import TaskFeatures
from lending.preprocessing import ExpandingWindowSplit, build_preprocessor
from lending.storage import PickleStore

RESULTS_DIR = config.REPORTS_DIR / "results"    # CSV de resultados (pequenos, entram no Git)
OOF_DIR = config.PROCESSED_DIR / "oof"          # previsões fora do fold (fora do Git)


@dataclass(frozen=True)
class TaskSetup:
    """
    Definição de uma tarefa supervisionada para o runner

    Attributes:
        name (str): "regression" ou "classification" (nomes do TaskFeatures)
        target (str): Coluna do alvo
        freq (str): Período dos folds ("Q" trimestre, "Y" ano)
        min_train_periods (int): Períodos de treino do primeiro fold
        metric (str): Métrica que escolhe a melhor configuração de cada modelo
        date_col (str): Coluna da data que ordena as linhas e define os folds
    """
    name: str
    target: str
    freq: str
    min_train_periods: int
    metric: str
    date_col: str = "issue_d"

    @property
    def is_classification(self):
        return self.name == "classification"

    @property
    def greater_is_better(self):
        return GREATER_IS_BETTER[self.metric]


TASKS = {
    # 6 folds trimestrais em 2013-2014: o 1.º treina em 2013T1-T2 e valida em 2013T3
    "regression": TaskSetup("regression", "int_rate", "Q", 2, "mae"),
    # 3 folds anuais: validam 2009, 2010 e 2011
    "classification": TaskSetup("classification", config.CLASS_COL, "Y", 2, "f1_macro"),
}


def config_id(model_name, params):
    """Identificador curto e estável de uma configuração (o mesmo em todas as execuções)."""
    text = json.dumps({"model": model_name, "params": params}, sort_keys=True, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


@dataclass(frozen=True)
class ModelSpec:
    """
    Um modelo candidato e a sua grelha de hiperparâmetros

    Attributes:
        name (str): Nome na tabela de comparação, ex.: "Random Forest" (único por tarefa)
        factory (callable): Função sem argumentos que devolve o estimador por treinar
        param_grid (dict): Grelha (ParameterGrid) com os nomes dos parâmetros do estimador,
            sem prefixo, ex.: {"max_depth": [4, 6], "min_samples_leaf": [50, 200]}
        preprocess (bool): False nas baselines que leem colunas originais (term, grade)
        max_train_rows (int | None): Se o treino do fold for maior, usa uma amostra aleatória
            deste tamanho (semente 42), mantendo a ordem temporal das linhas
        drop_features (tuple[str, ...]): Features a tirar depois do pré-processamento, com o
            respetivo indicador de falta (ex.: dti_with_loan nos modelos lineares e no kNN,
            por ser uma combinação linear de dti e installment_to_income)
    """
    name: str
    factory: Callable
    param_grid: dict = field(default_factory=dict)
    preprocess: bool = True
    max_train_rows: int | None = None
    drop_features: tuple = ()

    def configs(self):
        """
        Returns:
            list[tuple[str, dict]]: (config_id, parâmetros) de cada configuração da grelha
        """
        grid = list(ParameterGrid(self.param_grid)) if self.param_grid else [{}]
        return [(config_id(self.name, params), params) for params in grid]

    def make(self, params):
        """Estimador novo, por treinar, com os parâmetros da configuração."""
        estimator = self.factory()
        return estimator.set_params(**params) if params else estimator


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Tira colunas da saída do pré-processamento: a feature e o seu indicador de falta
    ("missingindicator_<feature>")

    Parameters:
        features (tuple[str, ...]): Nomes das features a tirar
    """

    def __init__(self, features=()):
        self.features = features

    def fit(self, X, y=None):
        columns = list(X.columns)
        self.dropped_ = [c for c in columns
                         if any(c == f or c.endswith(f"_{f}") for f in self.features)]
        self.kept_ = [c for c in columns if c not in self.dropped_]
        return self

    def transform(self, X):
        return X[self.kept_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_, dtype=object)


def default_preprocessor(task):
    """O pipeline da Parte 1 (sem o modelo): features novas e pré-processamento da tarefa."""
    base = TaskFeatures(task)
    return Pipeline([("features", FeatureBuilder(task)),
                     ("pre", build_preprocessor(add_engineered(base)))])


def _take(X, positions):
    return X.iloc[positions] if hasattr(X, "iloc") else X[positions]


class CrossValidationRunner:
    """
    Corre a validação cruzada temporal de uma tarefa e guarda os resultados

    Parameters:
        task (str | TaskSetup): "regression", "classification" ou um TaskSetup
        results_dir (Path): Pasta do CSV de resultados
        oof_dir (Path): Pasta das previsões fora do fold
        columns (list[str] | None): Colunas de entrada; por omissão input_columns(TaskFeatures(task))
        make_preprocessor (callable | None): Função que devolve o pré-processamento por ajustar;
            por omissão default_preprocessor(task)
        verbose (bool): Escreve uma linha por ajuste

    Methods:
        prepare(data), folds(data), run(data, specs), results(), oof(model, config_id),
        summary(), best(), comparison_table(), threshold(model, config_id), refit(data, spec)
    """

    def __init__(self, task, results_dir=RESULTS_DIR, oof_dir=OOF_DIR, columns=None,
                 make_preprocessor=None, verbose=True):
        self.task = TASKS[task] if isinstance(task, str) else task
        self.results_dir = Path(results_dir)
        self.oof_dir = Path(oof_dir) / self.task.name
        self._columns = list(columns) if columns is not None else None
        self.make_preprocessor = make_preprocessor or (lambda: default_preprocessor(self.task.name))
        self.verbose = verbose
        self.splitter = ExpandingWindowSplit(self.task.freq, self.task.min_train_periods)

    # ------------------------------------------------------------------ dados e folds

    @property
    def columns(self):
        if self._columns is None:
            self._columns = input_columns(TaskFeatures(self.task.name))
        return self._columns

    @property
    def metrics(self):
        return CLASSIFICATION_METRICS if self.task.is_classification else REGRESSION_METRICS

    @property
    def results_path(self):
        return self.results_dir / f"{self.task.name}_cv.csv"

    def prepare(self, data):
        """
        Confirma as colunas e o alvo e ORDENA as linhas por data (ordenação estável)

        Returns:
            pd.DataFrame: Cópia ordenada por issue_d, com o índice 0..n-1
        """
        needed = list(dict.fromkeys([self.task.date_col, self.task.target, *self.columns]))
        missing = [c for c in needed if c not in data.columns]
        if missing:
            raise ValueError(f"Faltam colunas nos dados: {missing}")
        target = data[self.task.target]
        if target.isna().any():
            raise ValueError(f"O alvo {self.task.target!r} tem valores em falta.")
        if self.task.is_classification:
            extra = sorted(set(target.unique()) - set(config.CLASS_ORDER))
            if extra:
                raise ValueError(f"Classes fora de {config.CLASS_ORDER} no treino: {extra}")
        return data.sort_values(self.task.date_col, kind="stable").reset_index(drop=True)

    def folds(self, data):
        """Tabela dos folds (períodos e tamanhos), com os dados já ordenados."""
        return self.splitter.describe(self.prepare(data)[self.task.date_col])

    def _sample(self, spec, A, y):
        """Amostra aleatória do treino (sem reposição), pela ordem temporal original."""
        if spec.max_train_rows is None or len(A) <= spec.max_train_rows:
            return A, y
        rng = np.random.default_rng(config.RANDOM_SEED)
        keep = np.sort(rng.choice(len(A), size=spec.max_train_rows, replace=False))
        return _take(A, keep), _take(y, keep)

    def _predict(self, model, A):
        return positive_proba(model, A) if self.task.is_classification else model.predict(A)

    def _score(self, y_true, prediction, threshold=0.5):
        if self.task.is_classification:
            return classification_metrics(y_true, prediction, threshold)
        return regression_metrics(y_true, prediction)

    # ------------------------------------------------------------------ correr

    def run(self, data, specs):
        """
        Corre (ou continua) a validação cruzada de todas as configurações dos modelos

        Parameters:
            data (pd.DataFrame): Coorte de treino da tarefa (ex.: o pickle REG_DEV_PKL)
            specs (list[ModelSpec]): Modelos e grelhas

        Returns:
            pd.DataFrame: Todas as linhas de resultados desta tarefa (results())
        """
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            raise ValueError(f"Há nomes de modelos repetidos: {names}")

        data = self.prepare(data)
        X, y, dates = data[self.columns], data[self.task.target], data[self.task.date_col]
        periods = pd.Series(pd.DatetimeIndex(dates).to_period(self.task.freq))
        folds = list(self.splitter.split(dates))
        done = {(r.model, r.config_id, r.fold) for r in self.results().itertuples()}

        for fold, (tr, va) in enumerate(folds, start=1):
            pending = [(spec, cid, params) for spec in specs for cid, params in spec.configs()
                       if (spec.name, cid, fold) not in done]
            if not pending:
                continue
            info = {"fold": fold,
                    "train_periods": f"{periods.iloc[tr].min()} a {periods.iloc[tr].max()}",
                    "valid_period": str(periods.iloc[va].iloc[0]),
                    "n_valid": len(va)}
            X_tr, X_va, y_tr, y_va = X.iloc[tr], X.iloc[va], y.iloc[tr], y.iloc[va]

            prep_seconds = 0.0
            if any(spec.preprocess for spec, _, _ in pending):
                start = time.perf_counter()
                prep = self.make_preprocessor()
                Xt_tr = prep.fit_transform(X_tr, y_tr)
                Xt_va = prep.transform(X_va)
                prep_seconds = time.perf_counter() - start

            for spec, cid, params in pending:
                A_tr, A_va = (Xt_tr, Xt_va) if spec.preprocess else (X_tr, X_va)
                if spec.drop_features:
                    dropper = ColumnDropper(spec.drop_features).fit(A_tr)
                    A_tr, A_va = dropper.transform(A_tr), dropper.transform(A_va)
                A_fit, y_fit = self._sample(spec, A_tr, y_tr)

                model = spec.make(params)
                start = time.perf_counter()
                model.fit(A_fit, y_fit)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                prediction = np.asarray(self._predict(model, A_va), dtype=float)
                predict_seconds = time.perf_counter() - start

                scores = self._score(y_va, prediction)
                self._save_oof(spec.name, cid, fold, va, dates.iloc[va], y_va, prediction)
                row = {"task": self.task.name, "model": spec.name, "config_id": cid,
                       "params": json.dumps(params, sort_keys=True, default=str), **info,
                       "n_train": len(A_fit),
                       "prep_seconds": round(prep_seconds if spec.preprocess else 0.0, 3),
                       "fit_seconds": round(fit_seconds, 3),
                       "predict_seconds": round(predict_seconds, 3)}
                if self.task.is_classification:
                    row["threshold"] = 0.5
                row.update(scores)
                row["timestamp"] = pd.Timestamp.now().isoformat(timespec="seconds")
                self._append(row)
                done.add((spec.name, cid, fold))
                if self.verbose:
                    note = " (limiar 0,5)" if self.task.is_classification else ""
                    print(f"[{self.task.name}] fold {fold}/{len(folds)} | {spec.name} {params} | "
                          f"{self.task.metric} {scores[self.task.metric]:.4f}{note} | "
                          f"{fit_seconds:.1f} s")
        return self.results()

    # ------------------------------------------------------------------ ficheiros

    def _oof_path(self, model, cid, fold):
        return self.oof_dir / _slug(model) / f"{cid}_fold{fold}.pkl"

    def _save_oof(self, model, cid, fold, rows, dates, y_true, prediction):
        column = "p_default" if self.task.is_classification else "y_pred"
        frame = pd.DataFrame({"row": rows, "issue_d": dates.to_numpy(),
                              "y_true": y_true.to_numpy(), column: prediction})
        path = self._oof_path(model, cid, fold)
        path.parent.mkdir(parents=True, exist_ok=True)
        PickleStore.save(frame, path)

    def _append(self, row):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        exists = self.results_path.exists()
        pd.DataFrame([row]).to_csv(self.results_path, mode="a", header=not exists, index=False)

    def results(self):
        """
        Returns:
            pd.DataFrame: Uma linha por modelo x configuração x fold (vazio se ainda não correu)
        """
        if not self.results_path.exists():
            return pd.DataFrame(columns=["model", "config_id", "fold"])
        # config_id como texto: um identificador como "1234567890" ou "12e4567890" seria lido
        # como número e deixaria de coincidir com o da grelha
        return pd.read_csv(self.results_path,
                           dtype={"model": str, "config_id": str, "params": str})

    def oof(self, model, cid):
        """
        Previsões fora do fold de uma configuração, todos os folds juntos

        Returns:
            pd.DataFrame: row, issue_d, y_true, y_pred (regressão) ou p_default, fold
        """
        folds = sorted(self.results().query("model == @model and config_id == @cid")["fold"])
        if not folds:
            raise KeyError(f"Não há resultados para {model!r} / {cid!r}.")
        frames = [PickleStore.load(self._oof_path(model, cid, f), expected_type=pd.DataFrame)
                  .assign(fold=f) for f in folds]
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------ resumos

    def summary(self):
        """
        Média e desvio-padrão (entre folds) de cada métrica, por configuração

        Na classificação, as métricas com limiar usam o limiar escolhido para a configuração
        (máximo do F1 macro nas previsões fora do fold); a AUC não depende do limiar.

        Returns:
            pd.DataFrame: Uma linha por modelo x configuração, ordenada pela métrica da
                tarefa (a melhor primeiro). "complete" indica se a configuração correu em
                todos os folds.
        """
        res = self.results()
        if res.empty:
            return pd.DataFrame()
        n_folds_total = res["fold"].nunique()
        rows = []
        for (model, cid), group in res.groupby(["model", "config_id"], sort=False):
            row = {"model": model, "config_id": cid, "params": group["params"].iloc[0],
                   "n_folds": len(group), "complete": len(group) == n_folds_total,
                   "n_train": group["n_train"].mean(), "fit_seconds": group["fit_seconds"].mean()}
            if self.task.is_classification:
                oof = self.oof(model, cid)
                row["threshold"], _ = best_threshold(oof["y_true"], oof["p_default"])
                per_fold = pd.DataFrame([classification_metrics(f["y_true"], f["p_default"],
                                                                row["threshold"])
                                         for _, f in oof.groupby("fold")])
            else:
                per_fold = group[REGRESSION_METRICS]
            for metric in self.metrics:
                row[f"{metric}_mean"] = per_fold[metric].mean()
                row[f"{metric}_std"] = per_fold[metric].std()   # desvio-padrão amostral (ddof=1)
            rows.append(row)
        return (pd.DataFrame(rows)
                .sort_values(f"{self.task.metric}_mean", ascending=not self.task.greater_is_better)
                .reset_index(drop=True))

    def best(self):
        """
        Returns:
            pd.DataFrame: A melhor configuração completa de cada modelo, a melhor primeiro
        """
        summary = self.summary()
        if summary.empty:
            return summary
        return summary[summary["complete"]].groupby("model", sort=False).head(1).reset_index(drop=True)

    def comparison_table(self, decimals=3):
        """
        Tabela de comparação da tarefa (requisito do enunciado): melhor configuração de cada
        modelo, com "média ± desvio" nos folds

        Returns:
            pd.DataFrame: Indexada pelo modelo
        """
        best = self.best()
        if best.empty:
            return best
        table = best[["model", "params", "n_train"]].copy()
        table["n_train"] = table["n_train"].round().astype(int)
        if self.task.is_classification:
            table["threshold"] = best["threshold"]
        for metric in self.metrics:
            table[metric] = [format_mean_std(m, s, decimals)
                             for m, s in zip(best[f"{metric}_mean"], best[f"{metric}_std"])]
        table["fit_seconds"] = best["fit_seconds"].round(1)
        return table.set_index("model")

    def threshold(self, model, cid):
        """Limiar escolhido na validação cruzada para uma configuração de classificação."""
        if not self.task.is_classification:
            raise ValueError("O limiar só existe na classificação.")
        oof = self.oof(model, cid)
        return best_threshold(oof["y_true"], oof["p_default"])[0]

    # ------------------------------------------------------------------ modelo final

    def refit(self, data, spec, cid=None):
        """
        Treina uma configuração em todo o treino da tarefa (para a avaliação final no teste)

        Usa as mesmas regras da validação cruzada: dados ordenados por data, pré-processamento
        ajustado em todas as linhas, mesma amostra máxima e mesmas colunas tiradas.

        Parameters:
            data (pd.DataFrame): Coorte de treino completa
            spec (ModelSpec): O modelo
            cid (str | None): Configuração; por omissão, a melhor desse modelo no summary()

        Returns:
            Pipeline: Pipeline já treinado (pré-processamento + modelo), pronto a prever
                sobre as colunas originais (runner.columns)
        """
        configs = dict(spec.configs())
        if cid is None:
            best = self.best()
            chosen = (best.loc[best["model"] == spec.name, "config_id"] if not best.empty
                      else pd.Series(dtype=str))
            if chosen.empty:
                raise ValueError(f"Não há uma configuração completa de {spec.name!r} nos resultados.")
            cid = chosen.iloc[0]
        if cid not in configs:
            raise KeyError(f"A configuração {cid!r} não pertence à grelha de {spec.name!r}.")

        data = self.prepare(data)
        X, y = data[self.columns], data[self.task.target]
        steps = []
        A = X
        if spec.preprocess:
            prep = self.make_preprocessor()
            A = prep.fit_transform(X, y)
            steps += prep.steps if isinstance(prep, Pipeline) else [("prep", prep)]
        if spec.drop_features:
            dropper = ColumnDropper(spec.drop_features).fit(A)
            A = dropper.transform(A)
            steps.append(("drop", dropper))
        A_fit, y_fit = self._sample(spec, A, y)
        model = spec.make(configs[cid]).fit(A_fit, y_fit)
        steps.append(("model", model))
        return Pipeline(steps)
