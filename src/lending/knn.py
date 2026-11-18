"""
kNN de raiz em NumPy - Fase 4 (Model Building)

Requisitos do enunciado cumpridos:
    - "Implement the k-nearest neighbors (kNN) algorithm from scratch in Python, using NumPy
      arrays and a suitable data structure. Do not substitute a library kNN estimator":
      este módulo só usa NumPy e não importa nada de sklearn.neighbors (há um teste que o
      confirma). Do scikit-learn só se herda a interface (BaseEstimator), para o kNN passar
      pelo mesmo Pipeline e pelo mesmo runner dos outros modelos.
    - "Explain and document your implementation": docstrings, docs/04_modelos/knn.md e o
      notebook 04a_knn.ipynb.
    - "Apply your kNN implementation to regression and classification, evaluate it, and
      discuss the results": KNNRegressor e KNNClassifier entram no catálogo (model_catalog.py)
      e são avaliados pelo runner, nos mesmos folds dos outros modelos.

Algoritmo (T04, p. 34-38):
    Treino: guardar os dados de treino; não há parâmetros a aprender.
    Previsão: para cada pedido, calcular a distância a TODOS os pontos de treino, escolher os
    k mais próximos e devolver a média do alvo (regressão) ou a proporção de cada classe entre
    eles (classificação), simples ("uniform") ou ponderada pelo inverso da distância
    ("distance").

Estrutura de dados (NeighborSearch):
    - Matriz de treino float32, contígua em memória, com a norma ao quadrado de cada linha
      calculada uma vez no fit.
    - Distâncias por blocos de pedidos: d^2(q, x) = ||q||^2 + ||x||^2 - 2 q.x. O termo q.x de
      um bloco inteiro é um único produto de matrizes (rápido, em BLAS), e a memória fica
      limitada ao tamanho do bloco.
    - Seleção parcial com np.argpartition: encontra os k menores sem ordenar a linha toda
      (custo linear) e só os k escolhidos são ordenados.
    - A distância dos k escolhidos é recalculada diretamente (||q - x||), porque a fórmula
      acima perde precisão em float32 para pontos muito próximos. Isto importa nos pesos 1/d.
    Porquê não uma KD-tree: com cerca de 100 a 120 colunas, uma KD-tree quase não consegue
    descartar ramos e acaba por visitar quase todos os pontos (maldição da dimensionalidade),
    ficando tão lenta como a força bruta. O notebook 04a mostra-o com tempos medidos.

Distância euclidiana (T04, p. 35): adequada a variáveis numéricas. O pipeline padroniza todas
as colunas, por isso nenhuma domina a distância só por causa da escala.

Vários k de uma vez (secção 3 do relatório):
    Com search_k (ex.: 101), a pesquisa devolve sempre os 101 vizinhos mais próximos e cada
    modelo usa os primeiros n_neighbors. O resultado da última pesquisa fica numa cache
    (NEIGHBOR_CACHE), identificada pelo conteúdo do treino e dos pedidos. Na grelha do runner,
    as 10 configurações de um fold (5 valores de k x 2 pesos) usam o mesmo treino e a mesma
    validação, por isso as distâncias são calculadas uma só vez por fold. O resultado é igual
    ao de calcular tudo de novo.
"""

import hashlib

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import check_is_fitted, column_or_1d, validate_data

WEIGHTS = ("uniform", "distance")

# Distância mínima nos pesos 1/d: um vizinho exatamente igual ao pedido fica com um peso
# enorme e domina a previsão, como no scikit-learn
DISTANCE_EPS = 1e-12


class NeighborSearch:
    """
    Pesquisa dos k vizinhos mais próximos por força bruta, vetorizada por blocos

    Attributes:
        X (np.ndarray): Matriz de treino (n, p), float32, contígua
        sq_norms (np.ndarray): ||x||^2 de cada linha de treino (n,)
        block_size (int): Pedidos processados de cada vez. A memória de um bloco é cerca de
            block_size x n x 12 bytes (distâncias em float32 e índices do argpartition em
            int64): 256 x 50 000 x 12 = 154 MB.

    Methods:
        squared_distances(Q): Distâncias ao quadrado entre um bloco de pedidos e o treino
        query(Q, k): Distâncias e índices dos k vizinhos de cada pedido, do mais próximo
            para o mais afastado
    """

    def __init__(self, X, block_size=256):
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        if self.X.ndim != 2 or len(self.X) == 0:
            raise ValueError("O treino tem de ser uma matriz 2D com pelo menos uma linha.")
        if block_size < 1:
            raise ValueError("block_size tem de ser >= 1.")
        self.sq_norms = np.einsum("ij,ij->i", self.X, self.X)
        self.block_size = int(block_size)

    @property
    def n_samples(self):
        return self.X.shape[0]

    def squared_distances(self, Q):
        """
        Parameters:
            Q (np.ndarray): Bloco de pedidos (m, p), float32

        Returns:
            np.ndarray: (m, n) com ||q - x||^2, nunca negativo
        """
        D = Q @ self.X.T                                   # q.x: um produto de matrizes
        D *= -2.0
        D += self.sq_norms[None, :]                        # + ||x||^2
        D += np.einsum("ij,ij->i", Q, Q)[:, None]          # + ||q||^2
        np.maximum(D, 0.0, out=D)                          # arredondamentos podem dar -1e-7
        return D

    def query(self, Q, k):
        """
        Parameters:
            Q (np.ndarray): Pedidos (m, p)
            k (int): Vizinhos por pedido, 1 <= k <= n

        Returns:
            tuple[np.ndarray, np.ndarray]: distâncias (m, k) float32 e índices (m, k) int64,
                ordenados do mais próximo para o mais afastado (empates: menor índice primeiro)
        """
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        if Q.ndim != 2 or Q.shape[1] != self.X.shape[1]:
            raise ValueError(f"Os pedidos têm {Q.shape[-1]} colunas; o treino tem {self.X.shape[1]}.")
        if not 1 <= k <= self.n_samples:
            raise ValueError(f"k tem de estar entre 1 e {self.n_samples}; recebido {k}.")

        distances = np.empty((len(Q), k), dtype=np.float32)
        indices = np.empty((len(Q), k), dtype=np.int64)
        for start in range(0, len(Q), self.block_size):
            block = slice(start, start + self.block_size)
            D = self.squared_distances(Q[block])
            if k < self.n_samples:
                chosen = np.argpartition(D, k - 1, axis=1)[:, :k]     # os k menores, sem ordem
            else:
                chosen = np.broadcast_to(np.arange(self.n_samples), D.shape)
            # A fórmula ||q||^2 + ||x||^2 - 2 q.x perde precisão em float32 quando q e x estão
            # muito próximos (diferença de números grandes). Serve para ESCOLHER os k vizinhos;
            # a distância dos k escolhidos é recalculada diretamente, ||q - x||, sem esse erro.
            d = np.sqrt(((self.X[chosen] - Q[block][:, None, :]) ** 2).sum(axis=2))
            order = np.lexsort((chosen, d), axis=-1)                 # por distância, depois índice
            distances[block] = np.take_along_axis(d, order, axis=1)
            indices[block] = np.take_along_axis(chosen, order, axis=1)
        return distances, indices


def fingerprint(array):
    """Identificador do conteúdo de uma matriz (forma, tipo e hash dos bytes)."""
    array = np.ascontiguousarray(array)
    digest = hashlib.blake2b(memoryview(array).cast("B"), digest_size=16).hexdigest()
    return array.shape, str(array.dtype), digest


class NeighborCache:
    """
    Guarda os vizinhos da última pesquisa, para outras configurações a reutilizarem

    Só há uma entrada: a da pesquisa mais recente. Serve para a grelha do runner, em que as
    configurações de um fold pesquisam o mesmo treino com os mesmos pedidos.

    Attributes:
        hits, misses (int): Pesquisas reutilizadas e pesquisas calculadas
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self._key, self._distances, self._indices = None, None, None
        self.hits = self.misses = 0

    def get(self, key, k):
        """Os k primeiros vizinhos guardados, ou None se a pesquisa guardada não servir."""
        if self._key == key and self._indices.shape[1] >= k:
            self.hits += 1
            return self._distances[:, :k], self._indices[:, :k]
        return None

    def put(self, key, distances, indices):
        self._key, self._distances, self._indices = key, distances, indices
        self.misses += 1


NEIGHBOR_CACHE = NeighborCache()


class _KNNBase(BaseEstimator):
    """
    Parte comum do regressor e do classificador

    Parameters:
        n_neighbors (int): k, número de vizinhos usados na previsão
        weights (str): "uniform" (média simples) ou "distance" (pesos 1/d)
        search_k (int | None): Vizinhos pesquisados (>= n_neighbors). Com o mesmo search_k
            em todas as configurações, a pesquisa é feita uma vez e reutilizada
        block_size (int): Pedidos por bloco no cálculo das distâncias
        use_cache (bool): Reutilizar a pesquisa anterior quando o treino e os pedidos são iguais

    Attributes (depois do fit):
        search_ (NeighborSearch): Os dados de treino, prontos a pesquisar
        n_features_in_ (int), feature_names_in_ (np.ndarray): Colunas vistas no fit
    """

    def __init__(self, n_neighbors=15, weights="uniform", search_k=None, block_size=256,
                 use_cache=True):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.search_k = search_k
        self.block_size = block_size
        self.use_cache = use_cache

    def _fit_X(self, X):
        if self.weights not in WEIGHTS:
            raise ValueError(f"weights tem de ser um de {WEIGHTS}; recebido {self.weights!r}.")
        if self.search_k is not None and self.search_k < self.n_neighbors:
            raise ValueError("search_k tem de ser >= n_neighbors.")
        X = validate_data(self, X, reset=True, dtype=np.float32)
        if not 1 <= self.n_neighbors <= len(X):
            raise ValueError(f"n_neighbors tem de estar entre 1 e {len(X)} (linhas de treino); "
                             f"recebido {self.n_neighbors}.")
        self.search_ = NeighborSearch(X, self.block_size)
        self._train_key = fingerprint(self.search_.X)
        return X

    def kneighbors(self, X, n_neighbors=None):
        """
        Os vizinhos de cada pedido no treino

        Parameters:
            X (pd.DataFrame | np.ndarray): Pedidos, com as mesmas colunas do fit
            n_neighbors (int | None): Por omissão, self.n_neighbors

        Returns:
            tuple[np.ndarray, np.ndarray]: distâncias euclidianas (m, k) e índices no treino
                (m, k), do mais próximo para o mais afastado
        """
        check_is_fitted(self, "search_")
        X = validate_data(self, X, reset=False, dtype=np.float32)
        k = n_neighbors or self.n_neighbors
        k_search = min(max(k, self.search_k or k), self.search_.n_samples)
        if not self.use_cache:
            distances, indices = self.search_.query(X, k_search)
            return distances[:, :k], indices[:, :k]
        key = (self._train_key, fingerprint(X))
        cached = NEIGHBOR_CACHE.get(key, k)
        if cached is not None:
            return cached
        distances, indices = self.search_.query(X, k_search)
        NEIGHBOR_CACHE.put(key, distances, indices)
        return distances[:, :k], indices[:, :k]

    def _neighbor_weights(self, distances):
        if self.weights == "uniform":
            return np.ones_like(distances, dtype=np.float64)
        return 1.0 / np.maximum(distances.astype(np.float64), DISTANCE_EPS)


class KNNRegressor(RegressorMixin, _KNNBase):
    """
    kNN para regressão: média (simples ou ponderada) do alvo dos k vizinhos

    Parameters: os de _KNNBase.

    Attributes (depois do fit):
        y_fit_ (np.ndarray): Alvo das linhas de treino
    """

    def fit(self, X, y):
        """
        Parameters:
            X (pd.DataFrame | np.ndarray): Saída do pipeline (colunas padronizadas)
            y (array-like): Alvo numérico

        Returns:
            KNNRegressor: O próprio modelo
        """
        X = self._fit_X(X)
        y = column_or_1d(y).astype(np.float64)
        check_consistent_length(X, y)
        self.y_fit_ = y
        return self

    def predict(self, X):
        distances, indices = self.kneighbors(X)
        w = self._neighbor_weights(distances)
        return (w * self.y_fit_[indices]).sum(axis=1) / w.sum(axis=1)


class KNNClassifier(ClassifierMixin, _KNNBase):
    """
    kNN para classificação: proporção (simples ou ponderada) de cada classe entre os k vizinhos

    A proporção da classe Default serve de probabilidade, para a AUC e para o limiar escolhido
    pelo runner. predict devolve a classe mais frequente entre os vizinhos.

    Pesos das classes (desequilíbrio): com class_weight="balanced", o voto de cada vizinho é
    multiplicado pelo peso da sua classe, n / (n_classes x n_classe), calculado no treino. É a
    mesma regra do scikit-learn e dos outros modelos do catálogo. Sem pesos, a proporção de
    Default fica na escala da taxa de default do treino, e com um limiar fixo a percentagem de
    Default previstos acompanha essa taxa de fold para fold. Dentro de um treino, os pesos são
    uma transformação crescente da proporção simples, por isso não mudam a AUC.

    Parameters: os de _KNNBase, mais
        class_weight (str | dict | None): "balanced", {classe: peso} ou None (votos iguais)

    Attributes (depois do fit):
        classes_ (np.ndarray): Classes por ordem alfabética (como no scikit-learn)
        y_codes_ (np.ndarray): Índice da classe de cada linha de treino em classes_
        class_weight_ (np.ndarray): Peso de cada classe, pela ordem de classes_
    """

    def __init__(self, n_neighbors=15, weights="uniform", search_k=None, block_size=256,
                 use_cache=True, class_weight=None):
        super().__init__(n_neighbors=n_neighbors, weights=weights, search_k=search_k,
                         block_size=block_size, use_cache=use_cache)
        self.class_weight = class_weight

    def fit(self, X, y):
        X = self._fit_X(X)
        y = column_or_1d(y)
        check_consistent_length(X, y)
        self.classes_, self.y_codes_ = np.unique(y, return_inverse=True)
        self.class_weight_ = self._class_weights()
        return self

    def _class_weights(self):
        if self.class_weight is None:
            return np.ones(len(self.classes_))
        if isinstance(self.class_weight, str) and self.class_weight == "balanced":
            counts = np.bincount(self.y_codes_, minlength=len(self.classes_))
            return len(self.y_codes_) / (len(self.classes_) * counts)
        if isinstance(self.class_weight, dict):
            return np.array([self.class_weight.get(c, 1.0) for c in self.classes_], dtype=np.float64)
        raise ValueError("class_weight tem de ser 'balanced', um dicionário ou None; "
                         f"recebido {self.class_weight!r}.")

    def predict_proba(self, X):
        """
        Returns:
            np.ndarray: (m, n_classes), proporção ponderada de cada classe, pela ordem de classes_
        """
        distances, indices = self.kneighbors(X)
        codes = self.y_codes_[indices]
        w = self._neighbor_weights(distances) * self.class_weight_[codes]
        proba = np.column_stack([(w * (codes == c)).sum(axis=1) for c in range(len(self.classes_))])
        return proba / w.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Classe com maior proporção ponderada entre os vizinhos (votação da maioria, T04, p. 37)."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
