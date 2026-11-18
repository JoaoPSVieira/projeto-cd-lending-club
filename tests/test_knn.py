"""
Testes do kNN de raiz (src/lending/knn.py)

O kNeighbors do scikit-learn aparece aqui SÓ como referência para confirmar os resultados.
O código do kNN não o usa (test_knn_module_does_not_use_sklearn_neighbors).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lending.knn as knn_module
from lending.knn import NEIGHBOR_CACHE, KNNClassifier, KNNRegressor, NeighborSearch, fingerprint


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1_500, 8))
    Q = rng.normal(size=(400, 8))
    y = 12 + 3 * X[:, 0] - X[:, 1] ** 2 + rng.normal(0, 0.5, len(X))
    labels = np.where(X[:, 0] + X[:, 2] > 1.2, "Default", "Good standing")
    return X, Q, y, labels


@pytest.fixture(autouse=True)
def empty_cache():
    NEIGHBOR_CACHE.clear()
    yield
    NEIGHBOR_CACHE.clear()


# ---------------------------------------------------------------------------------------
# Estrutura de dados: NeighborSearch
# ---------------------------------------------------------------------------------------

def brute_force(X, Q, k):
    """Referência direta, sem truques: todas as distâncias e ordenação completa."""
    D = np.sqrt(((Q[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    order = np.argsort(D, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(D, order, axis=1), order


def test_query_matches_direct_computation(data):
    X, Q, _, _ = data
    distances, indices = NeighborSearch(X).query(Q, 7)
    ref_d, ref_i = brute_force(X, Q, 7)
    assert np.array_equal(indices, ref_i)
    assert distances == pytest.approx(ref_d, abs=1e-4)
    assert (np.diff(distances, axis=1) >= 0).all()          # do mais próximo para o mais afastado


@pytest.mark.parametrize("block_size", [1, 7, 128, 10_000])
def test_blocks_do_not_change_the_result(data, block_size):
    X, Q, _, _ = data
    reference = NeighborSearch(X, block_size=400).query(Q, 5)
    result = NeighborSearch(X, block_size=block_size).query(Q, 5)
    assert np.array_equal(result[1], reference[1])


def test_query_with_k_equal_to_training_size():
    X = np.arange(10, dtype=float).reshape(5, 2)
    distances, indices = NeighborSearch(X).query(X[:1], 5)
    assert indices[0].tolist() == [0, 1, 2, 3, 4]
    assert distances[0, 0] == 0.0


def test_training_point_is_its_own_nearest_neighbour(data):
    X, _, _, _ = data
    distances, indices = NeighborSearch(X).query(X[:50], 1)
    assert indices[:, 0].tolist() == list(range(50))
    assert distances[:, 0].max() == 0.0          # distância recalculada sem erro de arredondamento


def test_query_rejects_bad_k_and_columns(data):
    X, Q, _, _ = data
    search = NeighborSearch(X)
    with pytest.raises(ValueError):
        search.query(Q, 0)
    with pytest.raises(ValueError):
        search.query(Q, len(X) + 1)
    with pytest.raises(ValueError):
        search.query(Q[:, :3], 5)


# ---------------------------------------------------------------------------------------
# Regressão e classificação: iguais ao scikit-learn (só como referência)
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("k", [1, 5, 31])
def test_regressor_matches_sklearn(data, weights, k):
    X, Q, y, _ = data
    ours = KNNRegressor(n_neighbors=k, weights=weights).fit(X, y).predict(Q)
    ref = KNeighborsRegressor(n_neighbors=k, weights=weights, algorithm="brute").fit(X, y).predict(Q)
    assert ours == pytest.approx(ref, abs=1e-5)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_classifier_matches_sklearn(data, weights):
    X, Q, _, labels = data
    ours = KNNClassifier(n_neighbors=15, weights=weights).fit(X, labels)
    ref = KNeighborsClassifier(n_neighbors=15, weights=weights, algorithm="brute").fit(X, labels)
    assert list(ours.classes_) == list(ref.classes_) == ["Default", "Good standing"]
    assert ours.predict_proba(Q) == pytest.approx(ref.predict_proba(Q), abs=1e-6)
    assert np.array_equal(ours.predict(Q), ref.predict(Q))


def test_distance_weights_with_an_exact_match():
    X = np.array([[0.0], [1.0], [2.0]])
    model = KNNRegressor(n_neighbors=3, weights="distance").fit(X, [10.0, 20.0, 30.0])
    assert model.predict(np.array([[1.0]]))[0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------------------
# Vários k de uma vez (search_k e cache)
# ---------------------------------------------------------------------------------------

def test_search_k_gives_the_same_predictions(data):
    X, Q, y, _ = data
    for k in (5, 15, 31):
        direct = KNNRegressor(n_neighbors=k, use_cache=False).fit(X, y).predict(Q)
        shared = KNNRegressor(n_neighbors=k, search_k=101).fit(X, y).predict(Q)
        assert shared == pytest.approx(direct)


def test_grid_computes_distances_once(data):
    X, Q, y, _ = data
    for k in (5, 15, 31, 51, 101):
        for weights in ("uniform", "distance"):
            KNNRegressor(n_neighbors=k, weights=weights, search_k=101).fit(X, y).predict(Q)
    assert NEIGHBOR_CACHE.misses == 1 and NEIGHBOR_CACHE.hits == 9


def test_cache_is_not_reused_for_other_data(data):
    X, Q, y, _ = data
    KNNRegressor(search_k=101).fit(X, y).predict(Q)
    KNNRegressor(search_k=101).fit(X[:-1], y[:-1]).predict(Q)      # outro treino
    KNNRegressor(search_k=101).fit(X, y).predict(Q[:-1])           # outros pedidos
    assert NEIGHBOR_CACHE.misses == 3 and NEIGHBOR_CACHE.hits == 0


def test_fingerprint_depends_on_content():
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert fingerprint(a) == fingerprint(a.copy())
    b = a.copy()
    b[0, 0] = 99
    assert fingerprint(a) != fingerprint(b)


# ---------------------------------------------------------------------------------------
# Interface, erros e requisito "sem kNN de biblioteca"
# ---------------------------------------------------------------------------------------

def test_pipeline_clone_and_pickle(data):
    X, Q, _, labels = data
    pipe = Pipeline([("scale", StandardScaler()), ("model", KNNClassifier(n_neighbors=5))])
    fitted = clone(pipe).set_params(model__n_neighbors=15, model__weights="distance").fit(X, labels)
    assert fitted["model"].n_neighbors == 15
    restored = pickle.loads(pickle.dumps(fitted))
    assert np.array_equal(restored.predict_proba(Q), fitted.predict_proba(Q))


def test_dataframe_columns_are_checked(data):
    X, Q, y, _ = data
    columns = [f"x{i}" for i in range(X.shape[1])]
    model = KNNRegressor().fit(pd.DataFrame(X, columns=columns), y)
    assert list(model.feature_names_in_) == columns
    with pytest.raises(ValueError):
        model.predict(pd.DataFrame(Q, columns=columns[::-1]))


@pytest.mark.parametrize("params", [{"n_neighbors": 0}, {"n_neighbors": 5_000},
                                    {"weights": "gaussian"}, {"n_neighbors": 15, "search_k": 5}])
def test_invalid_parameters(data, params):
    X, _, y, _ = data
    with pytest.raises(ValueError):
        KNNRegressor(**params).fit(X, y)


def test_errors_on_unfitted_model_and_missing_values(data):
    X, Q, y, _ = data
    with pytest.raises(NotFittedError):
        KNNRegressor().predict(Q)
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        KNNRegressor().fit(X_nan, y)


def test_knn_module_does_not_use_sklearn_neighbors():
    source = Path(knn_module.__file__).read_text(encoding="utf-8")
    code = [line for line in source.splitlines() if line.lstrip().startswith(("import", "from"))]
    assert not any("neighbors" in line for line in code)
    assert not any(line.startswith(("import scipy", "from scipy")) for line in code)
