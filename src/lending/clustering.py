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

Perfis dos mutuários (BorrowerProfiles):
    As 8 variáveis do ajuste seguem o exemplo do enunciado (FICO, DTI, antiguidade do
    crédito, utilização) e o plano da Parte 2. O pré-processamento repete os passos do
    pipeline da Parte 1 (log nos montantes, clipping aos percentis 0,5-99,5, imputação pela
    mediana, padronização) com uma diferença: SEM indicadores de "em falta". Um indicador
    0/1 raro, depois de padronizado, vale dezenas de desvios nas poucas linhas com 1 (com
    0,07% de uns, cerca de 37) e dominaria a distância euclidiana. Pela mesma razão, as
    binárias (ever_delinquent, has_public_record) ficam fora do ajuste e só descrevem os
    grupos.

Amostras:
    A silhueta e o Dunn calculam todas as distâncias entre pares de pontos (n² operações),
    o que é impossível com 466 124 empréstimos. Calculam-se numa amostra aleatória (por
    omissão 20 000 e 5 000 pontos). Os índices da amostra só dependem do número de linhas e
    da semente, por isso todas as configurações (cada k, cada algoritmo) são medidas nos
    MESMOS pontos e são comparáveis entre si. A WSS e o Davies-Bouldin usam todos os pontos.
"""

import json
import re
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from lending import config
from lending.cohorts import UNKNOWN_OUTCOME, make_target
from lending.experiment import ModelSpec
from lending.feature_engineering import FeatureBuilder, specs_for
from lending.features import TaskFeatures
from lending.preprocessing import QuantileClipper
from lending.storage import PickleStore

# Tamanhos das amostras da silhueta e do Dunn (plano da Parte 2)
SILHOUETTE_SAMPLE = 20_000
DUNN_SAMPLE = 5_000

# Número de grupos testado no clustering de mutuários (plano da Parte 2)
K_VALUES = list(range(2, 11))

# Onde ficam os resultados das grelhas (CSV, entram no Git) e os grupos de cada ponto
# (pickles, fora do Git, como as previsões fora do fold da classificação)
RESULTS_DIR = config.REPORTS_DIR / "results"
LABELS_DIR = config.PROCESSED_DIR / "clustering"

# Linhas por bloco no cálculo das distâncias do Dunn: 1 000 x 5 000 distâncias em float64
# ocupam 40 MB, em vez dos 200 MB da matriz completa
DUNN_CHUNK = 1_000

# Variáveis do perfil do mutuário que entram no ajuste (unidades originais).
# fico_range_low representa o FICO: o fico_range_high é sempre low + 4 (+5 em 34 casos),
# por isso as duas dariam o mesmo valor padronizado e o FICO contaria a dobrar.
PROFILE_FEATURES = ["fico_range_low", "dti", "revol_util", "annual_inc", "loan_to_income",
                    "revol_to_income", "open_acc_pct", "credit_history_years"]
# Montantes em USD: log(1 + x) antes do resto, como no pipeline da Parte 1
PROFILE_LOG = ["annual_inc"]
# Binárias: só descrevem os grupos depois do ajuste
PROFILE_DESCRIPTORS = ["ever_delinquent", "has_public_record"]

# Nomes legíveis para tabelas e figuras
PROFILE_LABELS = {
    "fico_range_low": "FICO",
    "dti": "DTI (%)",
    "revol_util": "Utilização rotativa (%)",
    "annual_inc": "Rendimento anual (USD)",
    "loan_to_income": "Montante / rendimento",
    "revol_to_income": "Saldo rotativo / rendimento",
    "open_acc_pct": "Contas abertas (%)",
    "credit_history_years": "Antiguidade do crédito (anos)",
    "ever_delinquent": "Já teve atrasos",
    "has_public_record": "Tem registo público",
}


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


def match_labels(reference, other):
    """
    Renomeia os grupos de `other` para corresponderem aos de `reference`

    Os nomes dos grupos são arbitrários: o "grupo 0" do k-means pode ser o "grupo 2" do
    GMM. Para comparar dois agrupamentos dos mesmos pontos grupo a grupo, emparelha-se
    cada grupo de `other` com um grupo de `reference` de forma a maximizar o número de
    pontos em comum (problema de afetação, resolvido pelo algoritmo húngaro sobre a tabela
    de contingência). Se `other` tiver mais grupos, os que sobram recebem nomes novos.

    Parameters:
        reference, other (array-like): Grupos dos mesmos pontos, pela mesma ordem, com
            etiquetas inteiras (como as do k-means e do GMM)

    Returns:
        np.ndarray: `other` com os nomes dos grupos de `reference`
    """
    reference, other = np.asarray(reference), np.asarray(other)
    if len(reference) != len(other):
        raise ValueError(f"Tamanhos diferentes: {len(reference)} e {len(other)}.")
    if not (np.issubdtype(reference.dtype, np.integer) and np.issubdtype(other.dtype, np.integer)):
        raise ValueError("As etiquetas dos grupos têm de ser números inteiros.")
    ref_groups, ref_idx = np.unique(reference, return_inverse=True)
    oth_groups, oth_idx = np.unique(other, return_inverse=True)
    counts = np.zeros((len(ref_groups), len(oth_groups)), dtype=np.int64)
    np.add.at(counts, (ref_idx, oth_idx), 1)

    rows, cols = linear_sum_assignment(-counts)   # maximizar = minimizar o simétrico
    mapping = {oth_groups[c]: ref_groups[r] for r, c in zip(rows, cols)}
    # Grupos de `other` sem par (quando tem mais grupos): nomes novos, a seguir aos existentes
    next_label = int(ref_groups.max()) + 1
    for group in oth_groups:
        if group not in mapping:
            mapping[group] = next_label
            next_label += 1
    return np.array([mapping[g] for g in oth_groups], dtype=np.int64)[oth_idx]


def matched_share(reference, other):
    """
    Fração dos pontos que ficam no grupo correspondente depois de emparelhar os grupos

    Complementa o ARI com uma leitura direta: 0,75 = três em cada quatro empréstimos estão
    no "mesmo" grupo nos dois agrupamentos.

    Returns:
        float: Entre 0 e 1
    """
    return float(np.mean(np.asarray(reference) == match_labels(reference, other)))


def group_profiles(table, labels, descriptors=PROFILE_DESCRIPTORS):
    """
    Perfil de cada grupo em unidades originais (para interpretar os grupos)

    Parameters:
        table (pd.DataFrame): Variáveis em unidades originais (ex.: BorrowerProfiles.table)
        labels (array-like): Grupo de cada linha de table (números ou nomes)
        descriptors (list[str]): Colunas binárias (0/1): em vez da mediana, a % de 1

    Returns:
        pd.DataFrame: Uma linha por grupo, com "empréstimos", "% dos empréstimos", a mediana
            de cada variável e, nas binárias, a percentagem de empréstimos com 1 (o nome da
            coluna mantém-se)
    """
    labels = pd.Series(np.asarray(labels), index=table.index, name="grupo")
    if labels.isna().any():
        raise ValueError("Há linhas sem grupo.")
    grouped = table.groupby(labels)
    binary = [c for c in table.columns if c in descriptors]
    profile = grouped[[c for c in table.columns if c not in descriptors]].median()
    for column in binary:
        profile[column] = 100 * grouped[column].mean()
    sizes = labels.value_counts()
    profile.insert(0, "% dos empréstimos", 100 * sizes / len(labels))
    profile.insert(0, "empréstimos", sizes)
    return profile


# =================================
# = Risco dos grupos (pós-ajuste) =
# =================================
#
# A taxa de juro e o desfecho NÃO entram no ajuste: são calculados depois de os grupos
# estarem fixos, para lhes dar significado de negócio.

# Corte de observação comum (Fase 1, docs/01_formulacao_problema/desenho_split.md): só se
# conhece o desfecho dos empréstimos cujo prazo termina até esta data
OBSERVATION_CUTOFF = pd.Timestamp("2018-12-31")


def outcome_observed(df, cutoff=OBSERVATION_CUTOFF):
    """
    Empréstimos com o desfecho observável: prazo terminado até ao corte e estado conhecido

    É a regra de observabilidade da classificação (36 meses com o prazo terminado),
    alargada aos empréstimos de 60 meses que também terminam antes do corte (concedidos
    até 2013-12). Ficam de fora os de 60 meses concedidos a partir de 2014: desses, só
    teriam acabado os que pagaram ou falharam cedo (follow-up desigual), o que enviesaria a
    taxa de default. Também ficam de fora os Current e os estados em falta.

    Parameters:
        df (pd.DataFrame): Com issue_d (data), term (meses) e loan_status
        cutoff (pd.Timestamp): Corte de observação

    Returns:
        pd.Series: bool, mesmo índice que df
    """
    issue = pd.to_datetime(df["issue_d"])
    # Mês da última prestação, contado em meses desde o ano 0 (a 1.ª prestação é no mês
    # seguinte ao da concessão, por isso a última é issue_d + term meses)
    last_payment = issue.dt.year * 12 + issue.dt.month - 1 + df["term"].astype("float64")
    ended = last_payment <= cutoff.year * 12 + cutoff.month - 1
    known = df["loan_status"].notna() & ~df["loan_status"].isin(UNKNOWN_OUTCOME)
    return (ended & known).astype(bool)


def proportion_ci(successes, n, z=1.96):
    """
    Proporção com intervalo de confiança de Wilson (95% por omissão)

    O intervalo de Wilson comporta-se melhor do que o p ± z·√(p(1-p)/n) quando a proporção
    está perto de 0 ou de 1 ou quando n é pequeno; com n grande, os dois coincidem.

    Parameters:
        successes, n (array-like): Contagens

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (proporção, limite inferior, limite superior)
    """
    successes = np.asarray(successes, dtype="float64")
    n = np.asarray(n, dtype="float64")
    p = successes / n
    denominator = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
    return p, center - half, center + half


def group_outcomes(df, labels, cutoff=OBSERVATION_CUTOFF):
    """
    Taxa de juro e taxa de default de cada grupo

    - Taxa de juro: em todos os empréstimos do grupo (é fixada na concessão, não depende do
      tempo de observação).
    - Default: só nos empréstimos com o desfecho observável (outcome_observed), com o alvo da
      classificação (make_target: Charged Off / Default, e os At risk depois do fim do prazo,
      contam como Default).

    Parameters:
        df (pd.DataFrame): Com int_rate, term, issue_d e loan_status
        labels (array-like): Grupo de cada linha de df

    Returns:
        pd.DataFrame: Uma linha por grupo: empréstimos, juro médio (%), juro mediano (%),
            60 meses (%), com desfecho, defaults, default (%) e o IC 95% (Wilson)
    """
    labels = pd.Series(np.asarray(labels), index=df.index, name="grupo")
    observed = outcome_observed(df, cutoff)
    is_default = pd.Series(False, index=df.index)
    is_default[observed] = make_target(df.loc[observed, "loan_status"]) == "Default"

    grouped = df.groupby(labels)
    table = pd.DataFrame({
        "empréstimos": labels.value_counts(),
        "juro médio (%)": grouped["int_rate"].mean(),
        "juro mediano (%)": grouped["int_rate"].median(),
        "60 meses (%)": 100 * grouped["term"].apply(lambda t: (t == 60).mean()),
        "com desfecho": observed.groupby(labels).sum(),
        "defaults": is_default.groupby(labels).sum(),
    })
    rate, low, high = proportion_ci(table["defaults"], table["com desfecho"])
    table["default (%)"] = 100 * rate
    table["IC 95% inferior"] = 100 * low
    table["IC 95% superior"] = 100 * high
    return table


# ================
# = Estabilidade =
# ================
#
# "Assess their stability" (enunciado): um agrupamento é estável se quase não muda quando se
# repete o procedimento com outra semente ou com outros dados da mesma população. Em todos os
# testes, cada modelo ajustado atribui um grupo a TODOS os pontos (predict), e as atribuições
# comparam-se nos mesmos pontos com o ARI e a concordância (matched_share).

# Sementes do teste de estabilidade (5, como no plano)
STABILITY_SEEDS = [config.RANDOM_SEED + i for i in range(5)]


def fit_predict(spec, params, X, fit_rows=None, seed=None):
    """
    Ajusta uma configuração (opcionalmente só em algumas linhas e com outra semente) e
    devolve o grupo de todos os pontos de X

    Parameters:
        spec (ModelSpec), params (dict): Algoritmo e configuração
        X (np.ndarray): Todos os pontos
        fit_rows (np.ndarray | None): Linhas do ajuste (None = todas)
        seed (int | None): Substitui o random_state da fábrica

    Returns:
        np.ndarray: Uma etiqueta por linha de X
    """
    X = np.asarray(X, dtype="float64")
    model = spec.make(params)
    if seed is not None:
        model.set_params(random_state=seed)
    model.fit(X if fit_rows is None else X[fit_rows])
    return model.predict(X)


def pairwise_agreement(labelings):
    """
    ARI e concordância entre todos os pares de agrupamentos dos mesmos pontos

    Parameters:
        labelings (dict[str, np.ndarray]): Pelo menos 2 agrupamentos

    Returns:
        pd.DataFrame: Uma linha por par (a, b, ari, matched)
    """
    table = ClusterEvaluator.stability(labelings)
    table["matched"] = [matched_share(labelings[a], labelings[b])
                        for a, b in zip(table["a"], table["b"])]
    return table


def seed_stability(spec, params, X, seeds=STABILITY_SEEDS):
    """
    Estabilidade à semente: o mesmo procedimento, nos mesmos dados, com sementes diferentes

    Mede quanto os grupos dependem do início aleatório (centros iniciais do k-means++, início
    do EM). Com n_init > 1, cada ajuste já escolhe o melhor de vários inícios.

    Returns:
        pd.DataFrame: Uma linha por par de sementes (10 pares com 5 sementes)
    """
    return pairwise_agreement({seed: fit_predict(spec, params, X, seed=seed) for seed in seeds})


def disjoint_halves(n, n_splits=5, seed=config.RANDOM_SEED):
    """
    Divisões aleatórias dos n pontos em duas metades sem pontos em comum

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: n_splits pares (metade A, metade B)
    """
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(n_splits):
        order = rng.permutation(n)
        splits.append((np.sort(order[: n // 2]), np.sort(order[n // 2:])))
    return splits


def subset_stability(spec, params, X, subsets):
    """
    Estabilidade aos dados: ajusta em cada subconjunto e compara os grupos de todos os pontos

    Com duas metades disjuntas, mede se outra amostra da mesma população daria os mesmos
    grupos. Com dois períodos (ex.: 2007-2012 e 2013-2014), mede se os grupos sobrevivem à
    mudança da população no tempo.

    Parameters:
        subsets (dict[str, np.ndarray]): Nome -> linhas do ajuste

    Returns:
        pd.DataFrame: Uma linha por par de subconjuntos
    """
    return pairwise_agreement({name: fit_predict(spec, params, X, fit_rows=rows)
                               for name, rows in subsets.items()})


def cached_csv(path, compute, recompute=False):
    """
    Lê um resultado guardado em CSV ou, se não existir (ou recompute=True), calcula-o e grava-o

    É a cache das análises que não passam pela ClusterGrid (ex.: a estabilidade): o
    RECALCULAR do notebook passa como recompute.

    Parameters:
        path (str | Path): Ficheiro CSV
        compute (Callable[[], pd.DataFrame]): Cálculo, só chamado quando é preciso
        recompute (bool): True ignora o ficheiro guardado

    Returns:
        pd.DataFrame
    """
    path = Path(path)
    if path.exists() and not recompute:
        return pd.read_csv(path)
    table = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


# ====================
# = BorrowerProfiles =
# ====================

class BorrowerProfiles:
    """
    Variáveis do clustering de mutuários e o seu pré-processamento

    Passos do pipeline (o que é aprendido no fit está entre parênteses):
        FeatureBuilder("clustering")   calcula as features novas (rácios, antiguidade, binárias)
        Montantes (PROFILE_LOG)        log(1 + x) -> clipping (percentis) -> imputação (mediana)
                                       -> padronização (média, desvio)
        Restantes variáveis            clipping (percentis) -> imputação (mediana) -> padronização
    Sem indicadores de "em falta" e sem as binárias (ver a docstring do módulo).

    O clustering é descritivo e não tem teste: o pré-processamento é ajustado em todos os
    empréstimos do clustering (2007-06 a 2014-12). O teste de 2015 nunca é usado.

    Attributes:
        features (list[str]): Variáveis do ajuste
        log_features (list[str]): Variáveis com log(1 + x) (subconjunto de features)
        descriptors (list[str]): Variáveis que só descrevem os grupos

    Methods:
        source_columns(): Colunas dos dados de que o pipeline precisa
        table(df): Variáveis do ajuste + descritores, em unidades originais
        preprocessor(): ColumnTransformer por ajustar
        pipeline(): FeatureBuilder + preprocessor, por ajustar

    Exemplo:
        profiles = BorrowerProfiles()
        data = PickleStore.load(config.ACCEPTED_DEV_PKL, expected_type=pd.DataFrame)
        X = profiles.pipeline().fit_transform(data[profiles.source_columns()])
        original = profiles.table(data)
    """

    def __init__(self, features=PROFILE_FEATURES, log_features=PROFILE_LOG,
                 descriptors=PROFILE_DESCRIPTORS):
        """
        Raises:
            ValueError: Se uma variável não for elegível no clustering (tabela de
                elegibilidade da Fase 1 ou features novas do clustering) ou se uma
                variável com log não estiver entre as do ajuste
        """
        self.features = list(features)
        self.log_features = list(log_features)
        self.descriptors = list(descriptors)

        not_in_fit = sorted(set(self.log_features) - set(self.features))
        if not_in_fit:
            raise ValueError(f"Variáveis com log que não entram no ajuste: {not_in_fit}")
        overlap = sorted(set(self.features) & set(self.descriptors))
        if overlap:
            raise ValueError(f"Variáveis no ajuste e nos descritores ao mesmo tempo: {overlap}")

        engineered = {spec.name for spec in specs_for("clustering")}
        eligible = set(TaskFeatures("clustering").columns) | engineered
        ineligible = [c for c in self.features + self.descriptors if c not in eligible]
        if ineligible:
            raise ValueError(f"Variáveis não elegíveis no clustering: {ineligible}")

    def source_columns(self):
        """
        Colunas dos dados de que o pipeline precisa

        As variáveis originais + as colunas de origem de TODAS as features novas do
        clustering (o FeatureBuilder calcula-as todas, mesmo as que não são usadas).

        Returns:
            list[str]: Sem repetições
        """
        engineered = {spec.name for spec in specs_for("clustering")}
        direct = [c for c in self.features + self.descriptors if c not in engineered]
        sources = [c for spec in specs_for("clustering") for c in spec.sources]
        return list(dict.fromkeys(direct + sources))

    def table(self, df):
        """
        Variáveis do ajuste + descritores em unidades originais (antes do pré-processamento)

        Serve para descrever os grupos (medianas por grupo, taxas das binárias).

        Parameters:
            df (pd.DataFrame): Dados com as colunas de source_columns()

        Returns:
            pd.DataFrame: Mesmo índice que df; colunas features + descriptors (float64)
        """
        built = FeatureBuilder("clustering").fit_transform(df[self.source_columns()])
        return built[self.features + self.descriptors].astype("float64")

    def preprocessor(self):
        """
        ColumnTransformer das variáveis do ajuste, por ajustar

        Uma transformação por coluna, pela ordem de features, para a saída ter as colunas
        por essa ordem (sem funções anónimas, para o pipeline se poder gravar com joblib).

        Returns:
            ColumnTransformer: Devolve um DataFrame (set_output pandas) com as colunas pela
                ordem de features
        """
        def steps(log):
            chain = [("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))] if log else []
            return Pipeline(chain + [
                ("clip", QuantileClipper()),
                ("impute", SimpleImputer(strategy="median")),   # sem add_indicator
                ("scale", StandardScaler()),
            ])

        return ColumnTransformer(
            [(column, steps(log=column in self.log_features), [column])
             for column in self.features],
            remainder="drop",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

    def pipeline(self):
        """
        FeatureBuilder + pré-processamento, por ajustar

        Pode receber um modelo no fim (ex.: Pipeline(profiles.pipeline().steps +
        [("model", KMeans(...))])) para guardar tudo junto no deployment.

        Returns:
            Pipeline
        """
        return Pipeline([("features", FeatureBuilder("clustering")),
                         ("pre", self.preprocessor())])


# ===========
# = Grelhas =
# ===========

def kmeans_spec(k_values=K_VALUES):
    """
    Grelha do k-means: número de grupos k

    n_init=10: o k-means converge para um mínimo local que depende dos centros iniciais;
    corre 10 inícios (k-means++) e fica o de menor WSS. random_state fixa os inícios.

    Returns:
        ModelSpec: Mesmo formato das grelhas da classificação e da regressão
    """
    factory = partial(KMeans, n_init=10, random_state=config.RANDOM_SEED)
    return ModelSpec("k-means", factory, {"n_clusters": list(k_values)})


def gmm_spec(k_values=K_VALUES, covariance_types=("full", "diag"), max_train_rows=None):
    """
    Grelha da mistura de gaussianas (GMM): número de componentes x tipo de covariância

    - full: cada grupo tem a sua matriz de covariância completa (elipses com qualquer
      orientação: as variáveis podem estar correlacionadas dentro do grupo).
      Parâmetros por grupo com d = 8 variáveis: 8 médias + 36 covariâncias.
    - diag: só as variâncias (elipses alinhadas com os eixos): 8 médias + 8 variâncias.
    O EM começa nos centros de um k-means (init_params="kmeans", o padrão) e, como o
    k-means, só garante um ótimo local: n_init=3 corre 3 inícios e fica o de maior
    verosimilhança.

    Parameters:
        max_train_rows (int | None): Ajustar numa amostra deste tamanho (o plano previa
            100 000 se o tempo apertasse; com os 466 124 o maior ajuste demora ~35 s)

    Returns:
        ModelSpec
    """
    factory = partial(GaussianMixture, n_init=3, random_state=config.RANDOM_SEED)
    grid = {"n_components": list(k_values), "covariance_type": list(covariance_types)}
    return ModelSpec("GMM", factory, grid, max_train_rows=max_train_rows)


def model_info(model, X_fit):
    """
    Informação própria de cada algoritmo, depois do ajuste

    Returns:
        dict:
            n_iter: iterações até convergir (k-means e GMM)
            converged: se o EM convergiu antes do limite de iterações (GMM)
            bic, aic: verosimilhança penalizada pelo número de parâmetros, nos dados do
                ajuste (GMM); mais baixo = melhor
            mean_max_proba: média, nos dados do ajuste, da probabilidade do grupo atribuído
                (GMM). 1 = cada ponto pertence claramente a um grupo; perto de 1/k = os
                grupos sobrepõem-se e a atribuição é quase ao acaso
    """
    info = {}
    if hasattr(model, "n_iter_"):
        info["n_iter"] = int(model.n_iter_)
    if hasattr(model, "converged_"):
        info["converged"] = bool(model.converged_)
    if hasattr(model, "bic"):
        info["bic"] = float(model.bic(X_fit))
        info["aic"] = float(model.aic(X_fit))
    if hasattr(model, "predict_proba"):
        info["mean_max_proba"] = float(model.predict_proba(X_fit).max(axis=1).mean())
    return info


def _slug(name):
    """Nome seguro para pastas e ficheiros (ex.: "k-means" -> "k_means")."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


class ClusterGrid:
    """
    Corre (ou continua) as grelhas dos algoritmos de clustering e mede cada agrupamento

    Para cada configuração de cada ModelSpec:
        1. ajusta o modelo em X (ou numa amostra de spec.max_train_rows linhas, semente 42);
        2. atribui um grupo a TODOS os pontos de X (predict);
        3. mede o agrupamento com o ClusterEvaluator (sempre nos mesmos pontos);
        4. grava uma linha no CSV de resultados e os grupos de cada ponto num pickle.

    Retomável, como o runner da validação cruzada: as configurações que já estão no CSV e
    têm os grupos gravados não voltam a correr. Com recompute=True (o RECALCULAR dos
    notebooks) apaga o CSV e os grupos desta grelha e calcula tudo de novo.

    Attributes:
        name (str): Nome da grelha (ex.: "borrowers"); dá o nome aos ficheiros
        evaluator (ClusterEvaluator): Medidas internas
        results_path (Path): reports/results/clustering_<name>.csv
        labels_dir (Path): data/processed/clustering/<name>/

    Methods:
        run(X, specs, recompute=False), results(), labels(model, config_id), clear()

    Exemplo:
        grid = ClusterGrid("borrowers")
        table = grid.run(X, [kmeans_spec()], recompute=RECALCULAR)
    """

    def __init__(self, name, evaluator=None, results_dir=RESULTS_DIR, labels_dir=LABELS_DIR,
                 verbose=True):
        self.name = name
        self.evaluator = ClusterEvaluator() if evaluator is None else evaluator
        self.results_path = Path(results_dir) / f"clustering_{_slug(name)}.csv"
        self.labels_dir = Path(labels_dir) / _slug(name)
        self.verbose = verbose

    # ------------------------------------------------------------------ correr

    def run(self, X, specs, recompute=False):
        """
        Parameters:
            X (pd.DataFrame | np.ndarray): Matriz já pré-processada (ex.: BorrowerProfiles)
            specs (list[ModelSpec]): Algoritmos e grelhas
            recompute (bool): True apaga os resultados guardados desta grelha e recomeça

        Returns:
            pd.DataFrame: Todas as linhas de resultados desta grelha (results())
        """
        if recompute:
            self.clear()
        X = np.asarray(X, dtype="float64")
        done = {(r.model, r.config_id) for r in self.results().itertuples()
                if self._labels_path(r.model, r.config_id).exists()}

        for spec in specs:
            fit_rows = self._fit_rows(len(X), spec.max_train_rows)
            for cid, params in spec.configs():
                if (spec.name, cid) in done:
                    continue
                model = spec.make(params)
                start = time.perf_counter()
                model.fit(X[fit_rows])
                fit_seconds = time.perf_counter() - start

                start = time.perf_counter()
                labels = model.predict(X).astype("int16")
                predict_seconds = time.perf_counter() - start

                start = time.perf_counter()
                scores = self.evaluator.evaluate(X, labels)
                eval_seconds = time.perf_counter() - start

                PickleStore.save(labels, self._labels_path(spec.name, cid))
                row = {"model": spec.name, "config_id": cid,
                       "params": json.dumps(params, sort_keys=True, default=str),
                       "n_points": len(X), "n_fit": len(fit_rows),
                       "fit_seconds": round(fit_seconds, 2),
                       "predict_seconds": round(predict_seconds, 2),
                       "eval_seconds": round(eval_seconds, 2),
                       **model_info(model, X[fit_rows]), **scores,
                       "timestamp": pd.Timestamp.now().isoformat(timespec="seconds")}
                self._save_row(row)
                done.add((spec.name, cid))
                if self.verbose:
                    print(f"[{self.name}] {spec.name} {params} | silhueta "
                          f"{scores['silhouette']:.3f} | {fit_seconds:.1f} s")
        return self.results()

    @staticmethod
    def _fit_rows(n, max_rows):
        """Linhas do ajuste: todas, ou uma amostra fixa (semente 42) de max_rows linhas."""
        if max_rows is None or max_rows >= n:
            return np.arange(n)
        rng = np.random.default_rng(config.RANDOM_SEED)
        return np.sort(rng.choice(n, size=max_rows, replace=False))

    # ------------------------------------------------------------------ ficheiros

    def _labels_path(self, model, cid):
        return self.labels_dir / _slug(model) / f"{cid}.pkl"

    def _save_row(self, row):
        """Acrescenta a linha ao CSV, substituindo uma linha antiga da mesma configuração."""
        old = self.results()
        old = old[~((old["model"] == row["model"]) & (old["config_id"] == row["config_id"]))]
        table = pd.DataFrame([row]) if old.empty else pd.concat([old, pd.DataFrame([row])],
                                                                ignore_index=True)
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(self.results_path, index=False)

    def results(self):
        """
        Returns:
            pd.DataFrame: Uma linha por algoritmo x configuração (vazio se ainda não correu)
        """
        if not self.results_path.exists():
            return pd.DataFrame(columns=["model", "config_id", "params"])
        # config_id como texto: um identificador só com dígitos seria lido como número
        return pd.read_csv(self.results_path, dtype={"model": str, "config_id": str, "params": str})

    def labels(self, model, cid):
        """
        Grupo de cada ponto de X numa configuração já corrida

        Returns:
            np.ndarray: int16, uma etiqueta por linha de X (pela ordem de X)
        """
        path = self._labels_path(model, cid)
        if not path.exists():
            raise FileNotFoundError(f"Sem grupos gravados para {model} {cid}: correr run() primeiro.")
        return PickleStore.load(path)

    def clear(self):
        """Apaga o CSV e os grupos gravados desta grelha."""
        if self.results_path.exists():
            self.results_path.unlink()
        if self.labels_dir.exists():
            shutil.rmtree(self.labels_dir)
