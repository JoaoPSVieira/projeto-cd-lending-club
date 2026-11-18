"""
Catálogo dos modelos e das grelhas de cada tarefa - Fase 4 (Model Building)

As grelhas são as da Tabela "Plano de modelos" da secção 3 do relatório. Estão aqui, num
só sítio, para a regressão (04b) e a classificação (04c) usarem exatamente as mesmas.

Requisitos do enunciado cumpridos:
    - "Apply two supervised models from scikit-learn (other than ensemble models) for each
      task": Ridge / regressão logística e árvore de decisão.
    - "Apply one bagging and one boosting ensemble model for each task": Random Forest e
      LightGBM (boosting.py).
    - Deep learning: MLP em Keras (deep.py).
    - kNN de raiz em NumPy (knn.py): k e pesos da secção 3; todas as configurações pesquisam
      os 101 vizinhos (search_k), por isso as distâncias de cada fold são calculadas uma vez.
    - Baselines: baselines.py e os Dummy do scikit-learn.

Contagem (sem baselines), igual nas duas tarefas:
    linear 6 + árvore 8 + Random Forest 4 + LightGBM 6 + MLP 6 = 30, mais 10 do kNN = 40,
    como na secção 3 (240 ajustes na regressão com 6 folds, 120 na classificação com 3).

Notas:
    - dti_with_loan = dti + installment_to_income (combinação linear): sai da regressão
      logística e do kNN (drop_features), fica nas árvores (secção 3). Só existe na
      classificação.
    - A MLP só importa o TensorFlow quando é criada, para o resto funcionar sem ele.
"""

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from lending.baselines import GradeRateBaseline, TermMedianBaseline
from lending.boosting import LGBMEarlyStoppingClassifier, LGBMEarlyStoppingRegressor
from lending.config import RANDOM_SEED
from lending.experiment import ModelSpec
from lending.knn import KNNClassifier, KNNRegressor

# Grelhas da secção 3
REGULARIZATION = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]         # 6 valores de 10^-3 a 10^2
TREE_GRID = {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [50, 200]}
FOREST_GRID = {"max_features": ["sqrt", 0.3], "min_samples_leaf": [10, 50]}
LGBM_GRID = {"learning_rate": [0.05, 0.1], "num_leaves": [15, 31, 63]}
MLP_GRID = {"hidden_layers": [(64,), (128, 64), (256, 128, 64)], "learning_rate": [1e-3, 3e-4]}

# kNN: k e pesos da secção 3; amostra por fold só na regressão
KNN_K = [5, 15, 31, 51, 101]
KNN_WEIGHTS = ["uniform", "distance"]
KNN_GRID = {"n_neighbors": KNN_K, "weights": KNN_WEIGHTS}
KNN_REGRESSION_SAMPLE = 50_000

# Features tiradas nos modelos sensíveis à colinearidade (regressão logística e kNN)
COLLINEAR_DROP = ("dti_with_loan",)

FOREST_SETTINGS = {"n_estimators": 300, "max_samples": 0.3, "n_jobs": -1,
                   "random_state": RANDOM_SEED}


def _mlp_regressor():
    from lending.deep import KerasMLPRegressor
    return KerasMLPRegressor()


def _mlp_classifier():
    from lending.deep import KerasMLPClassifier
    return KerasMLPClassifier()


def regression_specs(include_mlp=True):
    """
    Modelos da regressão (int_rate em T0), com as baselines primeiro

    Parameters:
        include_mlp (bool): False para correr sem TensorFlow

    Returns:
        list[ModelSpec]
    """
    specs = [
        ModelSpec("Mediana", lambda: DummyRegressor(strategy="median"), preprocess=False),
        ModelSpec("Mediana por prazo", TermMedianBaseline, preprocess=False),
        ModelSpec("kNN", lambda: KNNRegressor(search_k=max(KNN_K)), KNN_GRID,
                  max_train_rows=KNN_REGRESSION_SAMPLE),
        ModelSpec("Ridge", Ridge, {"alpha": REGULARIZATION}),
        ModelSpec("Árvore", lambda: DecisionTreeRegressor(random_state=RANDOM_SEED), TREE_GRID),
        ModelSpec("Random Forest", lambda: RandomForestRegressor(**FOREST_SETTINGS), FOREST_GRID),
        ModelSpec("LightGBM", LGBMEarlyStoppingRegressor, LGBM_GRID),
    ]
    if include_mlp:
        specs.append(ModelSpec("MLP", _mlp_regressor, MLP_GRID))
    return specs


def classification_specs(include_mlp=True):
    """
    Modelos da classificação (desfecho em T4), com as baselines primeiro

    Todos os modelos supervisionados ponderam as classes pela frequência no treino do fold.

    Parameters:
        include_mlp (bool): False para correr sem TensorFlow

    Returns:
        list[ModelSpec]
    """
    specs = [
        ModelSpec("Classe maioritária", lambda: DummyClassifier(strategy="most_frequent"),
                  preprocess=False),
        ModelSpec("Taxa do grau", GradeRateBaseline, preprocess=False),
        ModelSpec("kNN", lambda: KNNClassifier(search_k=max(KNN_K)), KNN_GRID,
                  drop_features=COLLINEAR_DROP),
        ModelSpec("Regressão logística",
                  lambda: LogisticRegression(class_weight="balanced", max_iter=2000),
                  {"C": REGULARIZATION}, drop_features=COLLINEAR_DROP),
        ModelSpec("Árvore", lambda: DecisionTreeClassifier(class_weight="balanced",
                                                            random_state=RANDOM_SEED), TREE_GRID),
        ModelSpec("Random Forest",
                  lambda: RandomForestClassifier(class_weight="balanced_subsample", **FOREST_SETTINGS),
                  FOREST_GRID),
        ModelSpec("LightGBM", lambda: LGBMEarlyStoppingClassifier(class_weight="balanced"),
                  LGBM_GRID),
    ]
    if include_mlp:
        specs.append(ModelSpec("MLP", _mlp_classifier, MLP_GRID))
    return specs
