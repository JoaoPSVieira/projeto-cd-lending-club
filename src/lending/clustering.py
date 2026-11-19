"""
Clustering: medidas internas de avaliação e estabilidade

Requisitos do enunciado que este módulo serve:
    - "Apply at least two clustering algorithms. Vary their relevant hyperparameters,
      including the number of clusters": cada configuração das grelhas (k-means, GMM) é
      medida da mesma forma, com o ClusterEvaluator.
    - "For clustering, use appropriate internal measures alongside substantive
      interpretation": cotovelo (soma dos quadrados dentro dos grupos), silhueta,
      Davies-Bouldin e Dunn.
    - "Describe the resulting groups and assess their stability": o ARI (adjusted Rand
      index) compara agrupamentos dos mesmos pontos (ex.: sementes ou amostras diferentes).

Medidas internas (só usam os dados e os grupos, porque no clustering não há alvo):
    - WSS (within-cluster sum of squares, a "inércia" do k-means): soma das distâncias ao
      quadrado de cada ponto ao centro do seu grupo. Desce sempre quando k sobe; o k
      escolhe-se no "cotovelo", onde a descida deixa de compensar.
    - Silhueta: para cada ponto, a = distância média aos pontos do seu grupo e b = distância
      média aos pontos do grupo vizinho mais próximo; s = (b - a) / max(a, b), entre -1 e 1.
      Perto de 1: o ponto está bem dentro do seu grupo; perto de 0: está na fronteira.
      A medida é a média de s. Mais alto = melhor.
    - Davies-Bouldin: para cada grupo, o pior rácio (dispersão do grupo + dispersão de
      outro) / distância entre os dois centros; depois a média por grupo. Mede se há grupos
      largos demasiado próximos. Mais baixo = melhor (0 é o mínimo).
    - Dunn (Dunn, 1974): menor distância entre dois pontos de grupos diferentes / maior
      distância entre dois pontos do mesmo grupo (o "diâmetro"). Mais alto = melhor. É uma
      medida de pior caso: um único ponto afastado aumenta o diâmetro e um único par de
      pontos próximos em grupos diferentes reduz a separação. Em dados contínuos e sem
      fronteiras nítidas (como os perfis de crédito) dá valores muito pequenos.

O índice de Dunn não existe no scikit-learn e está implementado aqui com NumPy
(dunn_index). As restantes medidas e o ARI vêm do scikit-learn.

Amostras:
    A silhueta e o Dunn calculam todas as distâncias entre pares de pontos (n² operações),
    o que é impossível com 466 124 empréstimos. Calculam-se numa amostra aleatória (por
    omissão 20 000 e 5 000 pontos). Os índices da amostra só dependem do número de linhas e
    da semente, por isso todas as configurações (cada k, cada algoritmo) são medidas nos
    MESMOS pontos e são comparáveis entre si. A WSS e o Davies-Bouldin usam todos os pontos.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score

from lending import config

# Tamanhos das amostras da silhueta e do Dunn (plano da Parte 2)
SILHOUETTE_SAMPLE = 20_000
DUNN_SAMPLE = 5_000

# Linhas por bloco no cálculo das distâncias do Dunn: 1 000 x 5 000 distâncias em float64
# ocupam 40 MB, em vez dos 200 MB da matriz completa
DUNN_CHUNK = 1_000


# ======================
# = Funções auxiliares =
# ======================

def _check_inputs(X, labels):
    """Converte para NumPy e confirma que há uma etiqueta por linha."""
    X = np.asarray(X, dtype="float64")
    labels = np.asarray(labels)
    if X.ndim != 2:
        raise ValueError(f"X tem de ser uma matriz (2 dimensões); tem {X.ndim}.")
    if len(X) != len(labels):
        raise ValueError(f"X tem {len(X)} linhas, mas há {len(labels)} etiquetas.")
    return X, labels


def euclidean_distances(A, B):
    """
    Distâncias euclidianas entre cada linha de A e cada linha de B

    Usa ||a - b||² = ||a||² + ||b||² - 2 a·b, que se calcula com um produto de matrizes
    (rápido) em vez de um ciclo por par. Os arredondamentos podem dar quadrados
    ligeiramente negativos (ex.: -1e-15) quando a = b, por isso cortam-se em 0.

    Parameters:
        A (np.ndarray): Matriz n x d
        B (np.ndarray): Matriz m x d

    Returns:
        np.ndarray: Matriz n x m
    """
    squared = (A ** 2).sum(axis=1)[:, None] + (B ** 2).sum(axis=1)[None, :] - 2 * A @ B.T
    return np.sqrt(np.maximum(squared, 0.0))


# =============================
# = Medidas de um agrupamento =
# =============================

def dunn_index(X, labels, chunk_size=DUNN_CHUNK):
    """
    Índice de Dunn: menor separação entre grupos / maior diâmetro de um grupo

    Definição original (Dunn, 1974):
        separação(i, j) = menor distância entre um ponto do grupo i e um ponto do grupo j
        diâmetro(k)     = maior distância entre dois pontos do grupo k
        Dunn            = min separação(i, j) / max diâmetro(k)

    Cálculo: para cada grupo, as distâncias dos seus pontos (em blocos de chunk_size
    linhas, para limitar a memória) a todos os pontos. As distâncias aos pontos do mesmo
    grupo dão o diâmetro; as distâncias aos dos outros grupos dão a separação. No total
    são n x n distâncias, como na matriz completa, mas nunca estão todas em memória.

    Parameters:
        X (array-like): Dados n x d (os mesmos que o algoritmo usou, já padronizados)
        labels (array-like): Grupo de cada linha (qualquer tipo de etiqueta)
        chunk_size (int): Linhas por bloco

    Returns:
        float: Índice de Dunn (infinito se todos os grupos tiverem um único ponto, porque
            nesse caso o maior diâmetro é 0)

    Raises:
        ValueError: Se houver menos de 2 grupos
    """
    X, labels = _check_inputs(X, labels)
    groups = np.unique(labels)
    if len(groups) < 2:
        raise ValueError("O índice de Dunn precisa de pelo menos 2 grupos.")

    max_diameter = 0.0
    min_separation = np.inf
    for group in groups:
        inside = labels == group
        members, others = X[inside], X[~inside]
        for start in range(0, len(members), chunk_size):
            block = members[start:start + chunk_size]
            max_diameter = max(max_diameter, euclidean_distances(block, members).max())
            min_separation = min(min_separation, euclidean_distances(block, others).min())

    if max_diameter == 0.0:
        return np.inf
    return float(min_separation / max_diameter)


def within_cluster_ss(X, labels):
    """
    Soma dos quadrados dentro dos grupos (WSS)

    Soma, em todos os grupos, das distâncias ao quadrado de cada ponto à média do seu
    grupo. Num k-means convergido é igual à inertia_; calculada assim serve também para
    agrupamentos de outros algoritmos (ex.: GMM), que não têm inertia_.

    Parameters:
        X (array-like): Dados n x d
        labels (array-like): Grupo de cada linha

    Returns:
        float
    """
    X, labels = _check_inputs(X, labels)
    total = 0.0
    for group in np.unique(labels):
        members = X[labels == group]
        total += ((members - members.mean(axis=0)) ** 2).sum()
    return float(total)


# ====================
# = ClusterEvaluator =
# ====================

class ClusterEvaluator:
    """
    Mede agrupamentos com as medidas internas e a estabilidade, sempre nos mesmos pontos

    Attributes:
        silhouette_sample (int | None): Pontos da amostra da silhueta (None = todos)
        dunn_sample (int | None): Pontos da amostra do Dunn (None = todos)
        random_state (int): Semente das amostras

    Methods:
        sample_index(n, size): Índices da amostra (iguais para todas as configurações)
        evaluate(X, labels): Medidas internas de um agrupamento
        stability(labelings): ARI entre todos os pares de agrupamentos dos mesmos pontos

    Exemplo:
        evaluator = ClusterEvaluator()
        for k in range(2, 11):
            labels = KMeans(k, n_init=10, random_state=42).fit_predict(X)
            rows.append({"k": k, **evaluator.evaluate(X, labels)})
    """

    def __init__(self, silhouette_sample=SILHOUETTE_SAMPLE, dunn_sample=DUNN_SAMPLE,
                 random_state=config.RANDOM_SEED):
        self.silhouette_sample = silhouette_sample
        self.dunn_sample = dunn_sample
        self.random_state = random_state

    def sample_index(self, n, size):
        """
        Índices de uma amostra aleatória sem reposição, por ordem crescente

        Só dependem de n, de size e da semente (não dos grupos): todas as configurações
        medidas nos mesmos dados usam os mesmos pontos.

        Parameters:
            n (int): Número de linhas dos dados
            size (int | None): Tamanho da amostra; None ou size >= n devolve todas as linhas

        Returns:
            np.ndarray
        """
        if size is None or size >= n:
            return np.arange(n)
        rng = np.random.default_rng(self.random_state)
        return np.sort(rng.choice(n, size=size, replace=False))

    def evaluate(self, X, labels):
        """
        Medidas internas de um agrupamento

        Parameters:
            X (array-like): Dados n x d usados no ajuste (já padronizados)
            labels (array-like): Grupo de cada linha

        Returns:
            dict: n_clusters, smallest_cluster_pct (% de linhas no grupo mais pequeno),
                wss, silhouette, davies_bouldin, dunn. A silhueta e o Dunn ficam NaN se a
                amostra tiver menos de 2 grupos.

        Raises:
            ValueError: Se houver menos de 2 grupos
        """
        X, labels = _check_inputs(X, labels)
        groups, counts = np.unique(labels, return_counts=True)
        if len(groups) < 2:
            raise ValueError("Um agrupamento precisa de pelo menos 2 grupos para ser avaliado.")

        result = {
            "n_clusters": len(groups),
            "smallest_cluster_pct": 100 * counts.min() / len(labels),
            "wss": within_cluster_ss(X, labels),
            "silhouette": np.nan,
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "dunn": np.nan,
        }

        idx = self.sample_index(len(X), self.silhouette_sample)
        n_groups = len(np.unique(labels[idx]))
        if 2 <= n_groups < len(idx):
            result["silhouette"] = float(silhouette_score(X[idx], labels[idx]))

        idx = self.sample_index(len(X), self.dunn_sample)
        if len(np.unique(labels[idx])) >= 2:
            result["dunn"] = dunn_index(X[idx], labels[idx])
        return result

    @staticmethod
    def stability(labelings):
        """
        ARI entre todos os pares de agrupamentos dos mesmos pontos

        O ARI conta os pares de pontos em que dois agrupamentos concordam (os dois juntos
        ou os dois separados) e desconta a concordância esperada por acaso. Não depende
        dos nomes dos grupos: o "grupo 0" de uma semente pode ser o "grupo 3" de outra.
        1 = agrupamentos iguais; cerca de 0 = concordância ao acaso.

        Parameters:
            labelings (list[array-like] | dict[str, array-like]): Pelo menos 2
                agrupamentos, todos com uma etiqueta para cada um dos mesmos pontos

        Returns:
            pd.DataFrame: Uma linha por par (a, b, ari)

        Raises:
            ValueError: Com menos de 2 agrupamentos ou tamanhos diferentes
        """
        if not isinstance(labelings, dict):
            labelings = {i: labels for i, labels in enumerate(labelings)}
        if len(labelings) < 2:
            raise ValueError("A estabilidade precisa de pelo menos 2 agrupamentos.")
        sizes = {len(labels) for labels in labelings.values()}
        if len(sizes) != 1:
            raise ValueError(f"Os agrupamentos têm tamanhos diferentes: {sorted(sizes)}.")

        names = list(labelings)
        rows = [{"a": a, "b": b, "ari": adjusted_rand_score(labelings[a], labelings[b])}
                for i, a in enumerate(names) for b in names[i + 1:]]
        return pd.DataFrame(rows)
