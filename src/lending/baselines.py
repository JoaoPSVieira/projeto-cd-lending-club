"""
Baselines - Fase 5 (comparação e avaliação)

Requisito do enunciado cumprido:
    - "Prepare one comparison table for each task, including baselines".

As baselines são as dos critérios de sucesso (docs/01_formulacao_problema/definicao_tarefas.md):
    regressão:      mediana do treino      -> DummyRegressor(strategy="median") do scikit-learn
                    mediana por prazo      -> TermMedianBaseline (aqui)
    classificação:  classe maioritária     -> DummyClassifier(strategy="most_frequent")
                    taxa de default do grau -> GradeRateBaseline (aqui)

As duas classes daqui leem uma coluna original (term ou grade), por isso recebem as colunas
de entrada da tarefa SEM pré-processamento (ModelSpec(..., preprocess=False) no runner).
Seguem a interface do scikit-learn para passarem pelo mesmo runner que os outros modelos.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from lending.evaluation import POSITIVE_LABEL


def _column(X, name):
    if not isinstance(X, pd.DataFrame) or name not in X.columns:
        raise ValueError(f"A baseline precisa de um DataFrame com a coluna {name!r} "
                         "(usar preprocess=False no ModelSpec).")
    return X[name]


class TermMedianBaseline(RegressorMixin, BaseEstimator):
    """
    Prevê a taxa de juro como a mediana do treino para o mesmo prazo (36 ou 60 meses)

    É a baseline exigente da regressão: o prazo, sozinho, separa a taxa em mais de 4 p.p.
    Um prazo que não exista no treino recebe a mediana global.

    Parameters:
        column (str): Coluna do prazo

    Attributes (depois do fit):
        medians_ (pd.Series): Mediana do alvo por prazo
        global_median_ (float): Mediana de todo o treino
    """

    def __init__(self, column="term"):
        self.column = column

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, dtype=float))
        groups = _column(X, self.column).to_numpy()
        self.medians_ = y.groupby(groups).median()
        self.global_median_ = float(y.median())
        return self

    def predict(self, X):
        check_is_fitted(self, "medians_")
        values = _column(X, self.column).map(self.medians_)
        return values.fillna(self.global_median_).to_numpy(dtype=float)


class GradeRateBaseline(ClassifierMixin, BaseEstimator):
    """
    Probabilidade de Default = taxa de default do grau (A a G) no treino

    Representa a avaliação de risco da própria LendingClub: comparar um modelo com esta
    baseline mostra quanto ele acrescenta ao grau. Um grau que não exista no treino recebe
    a taxa global.

    Parameters:
        column (str): Coluna do grau
        positive_label (str): Classe positiva

    Attributes (depois do fit):
        classes_ (np.ndarray): As duas classes, por ordem alfabética (como no scikit-learn)
        rates_ (pd.Series): Taxa de Default por grau
        global_rate_ (float): Taxa de Default de todo o treino
    """

    def __init__(self, column="grade", positive_label=POSITIVE_LABEL):
        self.column = column
        self.positive_label = positive_label

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2 or self.positive_label not in self.classes_:
            raise ValueError(f"Precisa de duas classes, incluindo {self.positive_label!r}; "
                             f"recebeu {list(self.classes_)}.")
        is_pos = pd.Series(y == self.positive_label, dtype=float)
        self.rates_ = is_pos.groupby(_column(X, self.column).to_numpy()).mean()
        self.global_rate_ = float(is_pos.mean())
        return self

    def predict_proba(self, X):
        """
        Returns:
            np.ndarray: (n, 2), pela ordem de classes_
        """
        check_is_fitted(self, "rates_")
        p = _column(X, self.column).map(self.rates_).fillna(self.global_rate_).to_numpy(dtype=float)
        pos = list(self.classes_).index(self.positive_label)
        proba = np.empty((len(p), 2))
        proba[:, pos], proba[:, 1 - pos] = p, 1 - p
        return proba

    def predict(self, X):
        """Classe com probabilidade >= 0,5 (o limiar final escolhe-se no runner)."""
        pos = list(self.classes_).index(self.positive_label)
        return self.classes_[np.where(self.predict_proba(X)[:, pos] >= 0.5, pos, 1 - pos)]
