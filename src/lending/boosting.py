"""
LightGBM com paragem antecipada temporal - Fase 4 (ensemble de boosting)

Requisito do enunciado cumprido:
    - "Apply one bagging and one boosting ensemble model for each task": este é o modelo de
      boosting (o de bagging é a Random Forest do scikit-learn, em model_catalog.py).

Porquê um wrapper:
    A secção 3 do relatório prevê LightGBM com paragem antecipada (no máximo 500 iterações).
    O LGBMRegressor/LGBMClassifier só para cedo se receber um conjunto de validação no fit,
    e o runner chama apenas fit(X, y). Este wrapper separa ele próprio a validação interna,
    com a mesma regra da MLP (lending/deep.py): as últimas validation_fraction linhas do
    treino, que são os empréstimos mais recentes porque o runner entrega os dados ordenados
    por issue_d. Assim, dentro do fold, também não se treina com o futuro.

    O critério de paragem é o erro quadrático (l2) na regressão e a AUC na classificação, a
    mesma métrica de paragem da MLP de classificação.

Os dados chegam ao LightGBM como matriz NumPy (os nomes das colunas do one-hot podem ter
caracteres que o LightGBM recusa); os nomes ficam em feature_names_in_ para as importâncias.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from lending.config import RANDOM_SEED


def last_rows_cut(n_rows, fraction):
    """
    Posição onde começa a validação interna (as últimas linhas)

    Returns:
        int: Linhas de treino; a validação interna são as linhas a partir daqui
    """
    if not 0 <= fraction < 1:
        raise ValueError(f"validation_fraction tem de estar em [0, 1); recebido {fraction}.")
    n_valid = int(round(n_rows * fraction))
    if fraction > 0 and (n_valid < 1 or n_rows - n_valid < 1):
        raise ValueError(f"Linhas insuficientes ({n_rows}) para a validação interna.")
    return n_rows - n_valid


class _LGBMEarlyStopping(BaseEstimator):
    """
    Parte comum. Parâmetros da grelha (secção 3): learning_rate e num_leaves.

    Attributes (depois do fit):
        model_ (lgb.LGBMModel): Modelo treinado
        best_iteration_ (int): Número de árvores usadas na previsão
        n_train_, n_valid_ (int): Linhas de treino e de validação interna
        feature_names_in_ (np.ndarray): Colunas vistas no fit (se X for um DataFrame)
    """

    _estimator = None   # LGBMRegressor ou LGBMClassifier
    _metric = None      # critério da paragem antecipada

    def __init__(self, learning_rate=0.1, num_leaves=31, n_estimators=500, stopping_rounds=20,
                 validation_fraction=0.15, random_state=RANDOM_SEED, n_jobs=-1):
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.stopping_rounds = stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _fit(self, X, y, **extra):
        X = validate_data(self, X, reset=True, dtype=np.float32)
        y = np.asarray(y)
        cut = last_rows_cut(len(X), self.validation_fraction)
        self.model_ = self._estimator(learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                                      n_estimators=self.n_estimators,
                                      random_state=self.random_state, n_jobs=self.n_jobs,
                                      metric=self._metric, verbose=-1, **extra)
        if cut < len(X):
            # eval_X/eval_y: forma do LightGBM 4.7 (o eval_set antigo está obsoleto)
            self.model_.fit(X[:cut], y[:cut], eval_X=(X[cut:],), eval_y=(y[cut:],),
                            callbacks=[lgb.early_stopping(self.stopping_rounds, verbose=False)])
            self.best_iteration_ = int(self.model_.best_iteration_)
        else:
            self.model_.fit(X, y)
            self.best_iteration_ = self.n_estimators
        self.n_train_, self.n_valid_ = cut, len(X) - cut
        return self

    def _X(self, X):
        check_is_fitted(self, "model_")
        return validate_data(self, X, reset=False, dtype=np.float32)

    def feature_importances(self, importance_type="gain"):
        """
        Importância de cada coluna (por omissão, o ganho total das divisões que a usam)

        Returns:
            pd.Series: Ordenada da mais importante para a menos importante
        """
        check_is_fitted(self, "model_")
        values = self.model_.booster_.feature_importance(importance_type=importance_type)
        names = getattr(self, "feature_names_in_", np.arange(len(values)))
        return pd.Series(values, index=names).sort_values(ascending=False)


class LGBMEarlyStoppingRegressor(RegressorMixin, _LGBMEarlyStopping):
    """LightGBM de regressão com paragem antecipada nas linhas mais recentes (critério l2)."""

    _estimator = lgb.LGBMRegressor
    _metric = "l2"

    def fit(self, X, y):
        return self._fit(X, np.asarray(y, dtype=float))

    def predict(self, X):
        return self.model_.predict(self._X(X))


class LGBMEarlyStoppingClassifier(ClassifierMixin, _LGBMEarlyStopping):
    """
    LightGBM de classificação binária com paragem antecipada nas linhas mais recentes
    (critério AUC)

    Parameters (além dos comuns):
        class_weight (str | dict | None): "balanced" pondera as classes pela frequência na
            parte de treino (sem a validação interna)

    Attributes (depois do fit):
        classes_ (np.ndarray): As duas classes, por ordem alfabética
    """

    _estimator = lgb.LGBMClassifier
    _metric = "auc"

    def __init__(self, learning_rate=0.1, num_leaves=31, n_estimators=500, stopping_rounds=20,
                 validation_fraction=0.15, class_weight=None, random_state=RANDOM_SEED, n_jobs=-1):
        super().__init__(learning_rate=learning_rate, num_leaves=num_leaves,
                         n_estimators=n_estimators, stopping_rounds=stopping_rounds,
                         validation_fraction=validation_fraction, random_state=random_state,
                         n_jobs=n_jobs)
        self.class_weight = class_weight

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"A classificação tem de ser binária; classes: {list(self.classes_)}.")
        # O LightGBM recebe 0/1 (o eval_y não aceita rótulos em texto); 1 = classes_[1]
        weights = self.class_weight
        if isinstance(weights, dict):
            weights = {i: weights[c] for i, c in enumerate(self.classes_)}
        return self._fit(X, (y == self.classes_[1]).astype(int), class_weight=weights)

    def predict_proba(self, X):
        """Probabilidades (n, 2), pela ordem de classes_."""
        return self.model_.predict_proba(self._X(X))

    def predict(self, X):
        """Classe com probabilidade >= 0,5 (o limiar final escolhe-se no runner)."""
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
