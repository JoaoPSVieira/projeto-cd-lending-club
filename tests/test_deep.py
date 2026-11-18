"""
Testes da MLP em Keras (lending.deep)

Correr na raiz do repositório:  python -m pytest tests/test_deep.py

Usam dados sintéticos pequenos e redes pequenas, para correrem em poucos segundos.
Verificam o que o resto do projeto assume da MLP:
    - interface do scikit-learn (get_params, clone, Pipeline, NotFittedError)
    - previsões nas unidades certas e melhores do que um palpite constante
    - reprodutibilidade com a mesma semente
    - validação interna da paragem antecipada = ÚLTIMAS linhas do treino
    - convenções de classificação (classes_, predict_proba com 2 colunas, limiar 0,5)
    - pesos das classes calculados só nas linhas que treinam a rede
    - erros claros quando os dados não servem
"""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from lending.deep import KerasMLPClassifier, KerasMLPRegressor

# Rede pequena e poucas épocas: chega para testar o comportamento, não a qualidade
SMALL = dict(hidden_layers=(8,), max_epochs=10, batch_size=64)


@pytest.fixture(scope="module")
def regression_data():
    """600 linhas, 4 colunas; alvo à volta de 12 (como uma taxa de juro em p.p.)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 4)), columns=["a", "b", "c", "d"])
    y = 12 + 3 * X["a"] - 2 * X["b"] + rng.normal(scale=0.3, size=600)
    return X, y


@pytest.fixture(scope="module")
def classification_data():
    """800 linhas com ~20% de "Default" (classe minoritária, como no projeto)."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(800, 4)), columns=["a", "b", "c", "d"])
    logit = -1.8 + 2 * X["a"]
    y = np.where(rng.random(800) < 1 / (1 + np.exp(-logit)), "Default", "Good standing")
    return X, y


# ---------------------------------------------------------------- interface do scikit-learn

@pytest.mark.parametrize("cls", [KerasMLPRegressor, KerasMLPClassifier])
def test_get_params_e_clone(cls):
    model = cls(hidden_layers=(16, 8), learning_rate=3e-3)
    params = model.get_params()
    assert params["hidden_layers"] == (16, 8) and params["learning_rate"] == 3e-3
    copy = clone(model)
    assert copy.get_params() == params


def test_clone_nao_copia_a_rede_treinada(regression_data):
    X, y = regression_data
    model = KerasMLPRegressor(**SMALL).fit(X, y)
    assert hasattr(model, "model_")
    assert not hasattr(clone(model), "model_")


@pytest.mark.parametrize("cls", [KerasMLPRegressor, KerasMLPClassifier])
def test_predict_antes_do_fit_da_erro(cls, regression_data):
    with pytest.raises(NotFittedError):
        cls().predict(regression_data[0])


def test_funciona_dentro_de_um_pipeline(regression_data):
    X, y = regression_data
    pipe = Pipeline([("scale", StandardScaler()), ("model", KerasMLPRegressor(**SMALL))])
    pipe.set_params(model__learning_rate=1e-2)
    pipe.fit(X, y)
    assert pipe["model"].learning_rate == 1e-2
    assert pipe.predict(X).shape == (len(X),)


# ---------------------------------------------------------------- regressão

def test_regressor_aprende_e_preve_nas_unidades_originais(regression_data):
    X, y = regression_data
    model = KerasMLPRegressor(hidden_layers=(16,), max_epochs=40, batch_size=32).fit(X, y)
    pred = model.predict(X)
    mae = np.abs(pred - y).mean()
    mae_median = np.abs(y - np.median(y)).mean()
    assert mae < 0.5 * mae_median                       # muito melhor do que a mediana
    assert abs(pred.mean() - y.mean()) < 1              # escala de p.p., não padronizada
    assert model.y_mean_ == pytest.approx(y.mean(), rel=1e-4)


def test_regressor_aceita_huber_e_recusa_outras_perdas(regression_data):
    X, y = regression_data
    KerasMLPRegressor(loss="huber", **SMALL).fit(X, y)
    with pytest.raises(ValueError, match="loss"):
        KerasMLPRegressor(loss="mae", **SMALL).fit(X, y)


# ---------------------------------------------------------------- reprodutibilidade

@pytest.mark.parametrize("cls,data", [(KerasMLPRegressor, "regression_data"),
                                      (KerasMLPClassifier, "classification_data")])
def test_mesma_semente_mesmo_resultado(cls, data, request):
    X, y = request.getfixturevalue(data)
    predict = "predict_proba" if cls is KerasMLPClassifier else "predict"
    a = getattr(cls(**SMALL).fit(X, y), predict)(X)
    b = getattr(cls(**SMALL).fit(X, y), predict)(X)
    c = getattr(cls(random_state=7, **SMALL).fit(X, y), predict)(X)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


# ---------------------------------------------------------------- paragem antecipada

def test_validacao_interna_sao_as_ultimas_linhas(regression_data):
    X, y = regression_data
    model = KerasMLPRegressor(validation_fraction=0.15, **SMALL).fit(X, y)
    cut = math.ceil(len(X) * 0.85)                      # o mesmo arredondamento do Keras
    assert (model.n_fit_, model.n_stop_) == (cut, len(X) - cut)

    # Estragar o alvo só no fim faz a perda da validação interna disparar;
    # estragar só no início (linhas de treino da rede) quase não a afeta.
    end, start = y.copy(), y.copy()
    end.iloc[cut:] += 50
    start.iloc[:len(X) - cut] += 50
    val_end = min(KerasMLPRegressor(**SMALL).fit(X, end).history_["val_loss"])
    val_start = min(KerasMLPRegressor(**SMALL).fit(X, start).history_["val_loss"])
    assert val_end > 10 * val_start


def test_paragem_antecipada_guarda_a_melhor_epoca():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 3))
    y = rng.normal(size=400)                            # ruído puro: não há nada para aprender
    model = KerasMLPRegressor(hidden_layers=(32,), max_epochs=50, patience=3,
                              batch_size=32).fit(X, y)
    assert model.n_epochs_ < 50                         # parou antes do máximo
    assert model.n_epochs_ == model.best_epoch_ + 3     # parou 3 épocas depois da melhor
    assert model.best_epoch_ - 1 == int(np.argmin(model.history_["val_loss"]))


def test_fracao_de_validacao_invalida(regression_data):
    X, y = regression_data
    for fraction in (0, 1, 1.5):
        with pytest.raises(ValueError, match="validation_fraction"):
            KerasMLPRegressor(validation_fraction=fraction, **SMALL).fit(X, y)


# ---------------------------------------------------------------- classificação

def test_convencoes_do_classificador(classification_data):
    X, y = classification_data
    model = KerasMLPClassifier(**SMALL).fit(X, y)
    assert list(model.classes_) == ["Default", "Good standing"]
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert ((proba >= 0) & (proba <= 1)).all()
    expected = np.where(proba[:, 1] >= 0.5, "Good standing", "Default")
    np.testing.assert_array_equal(model.predict(X), expected)
    assert "val_auc" in model.history_


def test_classificador_aceita_rotulos_0_1(classification_data):
    X, y = classification_data
    model = KerasMLPClassifier(**SMALL).fit(X, (y == "Default").astype(int))
    assert list(model.classes_) == [0, 1]


def test_pesos_balanced_so_com_as_linhas_de_treino_da_rede(classification_data):
    X, y = classification_data
    model = KerasMLPClassifier(class_weight="balanced", **SMALL).fit(X, y)
    fitted = y[:model.n_fit_]
    expected = compute_class_weight("balanced", classes=model.classes_, y=fitted)
    assert model.class_weight_["Default"] == pytest.approx(expected[0])
    assert model.class_weight_["Good standing"] == pytest.approx(expected[1])
    assert model.class_weight_["Default"] > 1 > model.class_weight_["Good standing"]


def test_pesos_por_dicionario_e_sem_pesos(classification_data):
    X, y = classification_data
    weights = {"Default": 3.0, "Good standing": 1.0}
    assert KerasMLPClassifier(class_weight=weights, **SMALL).fit(X, y).class_weight_ == weights
    assert KerasMLPClassifier(class_weight=None, **SMALL).fit(X, y).class_weight_ is None
    with pytest.raises(ValueError, match="class_weight"):
        KerasMLPClassifier(class_weight="auto", **SMALL).fit(X, y)


def test_pesos_aumentam_as_previsoes_da_classe_minoritaria(classification_data):
    X, y = classification_data
    params = dict(hidden_layers=(16,), max_epochs=30, batch_size=32)
    with_weights = KerasMLPClassifier(class_weight="balanced", **params).fit(X, y)
    without = KerasMLPClassifier(class_weight=None, **params).fit(X, y)
    assert (with_weights.predict(X) == "Default").sum() > (without.predict(X) == "Default").sum()


def test_classificador_recusa_tres_classes(classification_data):
    X, y = classification_data
    y3 = y.copy()
    y3[:5] = "At risk"
    with pytest.raises(ValueError, match="binária"):
        KerasMLPClassifier(**SMALL).fit(X, y3)


# ---------------------------------------------------------------- dados que não servem

@pytest.mark.parametrize("cls", [KerasMLPRegressor, KerasMLPClassifier])
def test_recusa_valores_em_falta(cls, classification_data):
    X, y = classification_data
    with pytest.raises(ValueError, match="build_preprocessor"):
        cls(**SMALL).fit(X.assign(a=np.nan), y if cls is KerasMLPClassifier else X["b"])


def test_recusa_numero_de_colunas_diferente(regression_data):
    X, y = regression_data
    model = KerasMLPRegressor(**SMALL).fit(X, y)
    with pytest.raises(ValueError, match="colunas"):
        model.predict(X.iloc[:, :3])


def test_recusa_x_e_y_de_tamanhos_diferentes(regression_data):
    X, y = regression_data
    with pytest.raises(ValueError, match="linhas"):
        KerasMLPRegressor(**SMALL).fit(X, y.iloc[:-1])
