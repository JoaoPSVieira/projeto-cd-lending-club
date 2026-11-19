"""
Testes das medidas de avaliação do clustering (src/lending/clustering.py)

O cdist do SciPy e as medidas do scikit-learn aparecem aqui SÓ como referência para
confirmar os resultados do índice de Dunn e da WSS, que são calculados por nós.
"""

import json
from functools import partial

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from lending.clustering import (PROFILE_DESCRIPTORS, PROFILE_FEATURES, BorrowerProfiles,
                                ClusterEvaluator, ClusterGrid, dunn_index, euclidean_distances,
                                kmeans_spec, within_cluster_ss)
from lending.experiment import ModelSpec


@pytest.fixture
def blobs():
    """Três grupos bem separados em 4 dimensões."""
    X, labels = make_blobs(n_samples=600, centers=3, n_features=4, cluster_std=0.6,
                           random_state=0)
    return X, labels


def dunn_brute_force(X, labels):
    """Referência direta: matriz de distâncias completa e a definição à letra."""
    D = cdist(X, X)
    same = labels[:, None] == labels[None, :]
    return D[~same].min() / D[same].max()


# ---------------------------------------------------------------------------------------
# Distâncias
# ---------------------------------------------------------------------------------------

def test_euclidean_distances_match_scipy():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(50, 6)), rng.normal(size=(30, 6))
    np.testing.assert_allclose(euclidean_distances(A, B), cdist(A, B), atol=1e-10)


def test_euclidean_distance_of_a_point_to_itself_is_zero():
    A = np.array([[1e3, -2e3, 5.5]])
    assert euclidean_distances(A, A)[0, 0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------------------
# Índice de Dunn
# ---------------------------------------------------------------------------------------

def test_dunn_hand_computed_example():
    # Grupo A = {0, 1} (diâmetro 1); grupo B = {5, 7} (diâmetro 2); separação = 5 - 1 = 4
    X = np.array([[0.0], [1.0], [5.0], [7.0]])
    assert dunn_index(X, ["A", "A", "B", "B"]) == pytest.approx(4 / 2)


def test_dunn_matches_brute_force(blobs):
    X, labels = blobs
    assert dunn_index(X, labels) == pytest.approx(dunn_brute_force(X, labels))


def test_dunn_matches_brute_force_with_overlapping_groups():
    # Grupos ao acaso (sem estrutura): separação quase 0, Dunn muito pequeno
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 5))
    labels = rng.integers(0, 4, size=400)
    assert dunn_index(X, labels) == pytest.approx(dunn_brute_force(X, labels))
    assert dunn_index(X, labels) < 0.1


def test_dunn_chunk_size_does_not_change_the_result(blobs):
    X, labels = blobs
    assert dunn_index(X, labels, chunk_size=7) == pytest.approx(dunn_index(X, labels))


def test_dunn_does_not_depend_on_label_names(blobs):
    X, labels = blobs
    renamed = np.array(["c", "a", "b"])[labels]
    assert dunn_index(X, renamed) == pytest.approx(dunn_index(X, labels))


def test_dunn_is_higher_for_well_separated_groups(blobs):
    X, labels = blobs
    shuffled = np.random.default_rng(0).permutation(labels)
    assert dunn_index(X, labels) > 10 * dunn_index(X, shuffled)


def test_dunn_needs_two_groups():
    with pytest.raises(ValueError, match="2 grupos"):
        dunn_index(np.zeros((5, 2)), [0] * 5)


def test_dunn_with_single_point_groups_is_infinite():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [3.0, 0.0]])
    assert dunn_index(X, [0, 1, 2]) == np.inf


def test_dunn_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="etiquetas"):
        dunn_index(np.zeros((5, 2)), [0, 1])


# ---------------------------------------------------------------------------------------
# Soma dos quadrados dentro dos grupos (WSS)
# ---------------------------------------------------------------------------------------

def test_wss_equals_kmeans_inertia(blobs):
    X, _ = blobs
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    assert within_cluster_ss(X, km.labels_) == pytest.approx(km.inertia_, rel=1e-6)


def test_wss_hand_computed_example():
    # Grupo A = {0, 2} (média 1: 1 + 1); grupo B = {10} (0)
    X = np.array([[0.0], [2.0], [10.0]])
    assert within_cluster_ss(X, [0, 0, 1]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------------------
# ClusterEvaluator
# ---------------------------------------------------------------------------------------

def test_evaluate_returns_every_measure(blobs):
    X, labels = blobs
    result = ClusterEvaluator().evaluate(X, labels)
    assert list(result) == ["n_clusters", "smallest_cluster_pct", "wss", "silhouette",
                            "davies_bouldin", "dunn"]
    assert result["n_clusters"] == 3
    assert result["smallest_cluster_pct"] == pytest.approx(100 / 3)


def test_evaluate_without_sampling_matches_full_measures(blobs):
    # As amostras (20 000 e 5 000) são maiores do que os dados: usa todos os pontos
    X, labels = blobs
    result = ClusterEvaluator().evaluate(X, labels)
    assert result["silhouette"] == pytest.approx(silhouette_score(X, labels))
    assert result["dunn"] == pytest.approx(dunn_brute_force(X, labels))


def test_evaluate_prefers_the_true_groups(blobs):
    X, labels = blobs
    evaluator = ClusterEvaluator()
    good = evaluator.evaluate(X, labels)
    bad = evaluator.evaluate(X, np.random.default_rng(0).permutation(labels))
    assert good["silhouette"] > bad["silhouette"]
    assert good["davies_bouldin"] < bad["davies_bouldin"]
    assert good["dunn"] > bad["dunn"]
    assert good["wss"] < bad["wss"]


def test_sample_is_the_same_for_every_configuration():
    evaluator = ClusterEvaluator(silhouette_sample=100, dunn_sample=50)
    first = evaluator.sample_index(1_000, 100)
    second = evaluator.sample_index(1_000, 100)
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 100
    assert np.all(np.diff(first) > 0)


def test_evaluate_with_sampling_is_reproducible(blobs):
    X, labels = blobs
    evaluator = ClusterEvaluator(silhouette_sample=200, dunn_sample=100)
    first, second = evaluator.evaluate(X, labels), evaluator.evaluate(X, labels)
    assert first == second
    # Com amostra, a silhueta é uma estimativa próxima da de todos os pontos
    assert first["silhouette"] == pytest.approx(silhouette_score(X, labels), abs=0.05)


def test_evaluate_needs_two_groups():
    with pytest.raises(ValueError, match="2 grupos"):
        ClusterEvaluator().evaluate(np.zeros((5, 2)), [1] * 5)


# ---------------------------------------------------------------------------------------
# Estabilidade (ARI)
# ---------------------------------------------------------------------------------------

def test_stability_ignores_group_names(blobs):
    _, labels = blobs
    renamed = np.array([2, 0, 1])[labels]
    table = ClusterEvaluator.stability({"semente 1": labels, "semente 2": renamed})
    assert list(table.columns) == ["a", "b", "ari"]
    assert table.loc[0, "ari"] == pytest.approx(1.0)


def test_stability_of_random_groups_is_near_zero(blobs):
    _, labels = blobs
    rng = np.random.default_rng(0)
    table = ClusterEvaluator.stability([labels, rng.permutation(labels)])
    assert abs(table.loc[0, "ari"]) < 0.05


def test_stability_has_one_row_per_pair(blobs):
    _, labels = blobs
    table = ClusterEvaluator.stability([labels] * 5)
    assert len(table) == 5 * 4 // 2
    assert isinstance(table, pd.DataFrame)


def test_stability_rejects_bad_inputs(blobs):
    _, labels = blobs
    with pytest.raises(ValueError, match="pelo menos 2"):
        ClusterEvaluator.stability([labels])
    with pytest.raises(ValueError, match="tamanhos diferentes"):
        ClusterEvaluator.stability([labels, labels[:10]])


# ---------------------------------------------------------------------------------------
# Perfis dos mutuários (BorrowerProfiles)
# ---------------------------------------------------------------------------------------

@pytest.fixture
def loans():
    """Empréstimos sintéticos com as colunas de origem do clustering, com alguns em falta."""
    rng = np.random.default_rng(0)
    n = 2_000
    issue = pd.to_datetime("2012-01-01") + pd.to_timedelta(rng.integers(0, 1_000, n), unit="D")
    issue = issue.to_period("M").to_timestamp()
    earliest = (issue - pd.to_timedelta(rng.integers(400, 12_000, n), unit="D")).to_period("M").to_timestamp()
    df = pd.DataFrame({
        "fico_range_low": rng.choice(np.arange(660, 850, 5), n).astype(float),
        "dti": rng.uniform(0, 40, n),
        "revol_util": rng.uniform(0, 100, n),
        "annual_inc": np.exp(rng.normal(11, 0.6, n)),
        "loan_amnt": rng.uniform(1_000, 35_000, n),
        "revol_bal": rng.exponential(15_000, n),
        "open_acc": rng.integers(1, 20, n).astype(float),
        "total_acc": rng.integers(20, 60, n).astype(float),
        "issue_d": issue,
        "earliest_cr_line": earliest,
        "delinq_2yrs": rng.poisson(0.3, n).astype(float),
        "mths_since_last_delinq": np.where(rng.random(n) < 0.5, np.nan, rng.integers(1, 100, n)),
        "pub_rec": rng.poisson(0.1, n).astype(float),
        "int_rate": rng.uniform(5, 25, n),   # coluna que não pode entrar
    })
    df.loc[:9, "revol_util"] = np.nan
    df.loc[10:12, "annual_inc"] = np.nan
    return df


def test_profiles_default_variables():
    profiles = BorrowerProfiles()
    assert profiles.features == PROFILE_FEATURES
    assert len(profiles.features) == 8
    assert "credit_history_years" in profiles.features
    assert profiles.log_features == ["annual_inc"]


def test_profile_pipeline_gives_standardized_columns_without_missing(loans):
    profiles = BorrowerProfiles()
    X = profiles.pipeline().fit_transform(loans[profiles.source_columns()])
    assert list(X.columns) == PROFILE_FEATURES   # sem indicadores de falta nem binárias
    assert not X.isna().any().any()
    np.testing.assert_allclose(X.mean(), 0, atol=1e-9)
    np.testing.assert_allclose(X.std(ddof=0), 1, atol=1e-9)


def test_profile_pipeline_logs_only_the_income(loans):
    profiles = BorrowerProfiles()
    X = profiles.pipeline().fit_transform(loans[profiles.source_columns()])
    income = np.log1p(loans["annual_inc"])
    income = income.clip(income.quantile(0.005), income.quantile(0.995))
    income = income.fillna(income.median())
    expected = (income - income.mean()) / income.std(ddof=0)
    np.testing.assert_allclose(X["annual_inc"], expected, atol=1e-9)
    # O DTI não leva log: a ordem dos valores e a forma da distribuição mantêm-se
    dti = loans["dti"].clip(loans["dti"].quantile(0.005), loans["dti"].quantile(0.995))
    np.testing.assert_allclose(X["dti"], (dti - dti.mean()) / dti.std(ddof=0), atol=1e-9)


def test_profile_pipeline_ignores_other_columns(loans):
    profiles = BorrowerProfiles()
    X = profiles.pipeline().fit_transform(loans)   # com int_rate e tudo o resto
    assert "int_rate" not in X.columns
    assert list(X.columns) == PROFILE_FEATURES


def test_profile_table_is_in_original_units(loans):
    profiles = BorrowerProfiles()
    table = profiles.table(loans)
    assert list(table.columns) == PROFILE_FEATURES + PROFILE_DESCRIPTORS
    np.testing.assert_allclose(table["fico_range_low"], loans["fico_range_low"])
    years = ((loans["issue_d"].dt.year - loans["earliest_cr_line"].dt.year) * 12
             + loans["issue_d"].dt.month - loans["earliest_cr_line"].dt.month) / 12
    np.testing.assert_allclose(table["credit_history_years"], years)
    assert set(table["has_public_record"].dropna().unique()) <= {0.0, 1.0}
    assert table["revol_util"].isna().sum() == 10   # a tabela não imputa


def test_profile_pipeline_can_be_pickled(loans):
    import pickle
    profiles = BorrowerProfiles()
    fitted = profiles.pipeline().fit(loans[profiles.source_columns()])
    restored = pickle.loads(pickle.dumps(fitted))
    pd.testing.assert_frame_equal(restored.transform(loans), fitted.transform(loans))


def test_profiles_reject_ineligible_variables():
    # total_pymnt: pós-concessão (proibida); int_rate: ❌ no clustering (tabela da Fase 1)
    with pytest.raises(ValueError, match="não elegíveis"):
        BorrowerProfiles(features=["fico_range_low", "total_pymnt"], log_features=[])
    with pytest.raises(ValueError, match="não elegíveis"):
        BorrowerProfiles(features=["fico_range_low", "int_rate"], log_features=[])


def test_profiles_reject_inconsistent_lists():
    with pytest.raises(ValueError, match="log"):
        BorrowerProfiles(features=["fico_range_low", "dti"], log_features=["annual_inc"])
    with pytest.raises(ValueError, match="descritores"):
        BorrowerProfiles(descriptors=["dti"])


# ---------------------------------------------------------------------------------------
# Grelhas (ClusterGrid)
# ---------------------------------------------------------------------------------------

class CountingKMeans(KMeans):
    """KMeans que conta quantas vezes foi ajustado (para testar a retoma)."""
    fits = 0

    def fit(self, X, y=None, sample_weight=None):
        CountingKMeans.fits += 1
        return super().fit(X, y, sample_weight)


def counting_spec(k_values=(2, 3, 4)):
    factory = partial(CountingKMeans, n_init=3, random_state=0)
    return ModelSpec("k-means", factory, {"n_clusters": list(k_values)})


@pytest.fixture
def grid(tmp_path):
    CountingKMeans.fits = 0
    return ClusterGrid("teste", results_dir=tmp_path / "results", labels_dir=tmp_path / "labels",
                       verbose=False)


def test_kmeans_spec_default_grid():
    spec = kmeans_spec()
    assert spec.name == "k-means"
    assert [params["n_clusters"] for _, params in spec.configs()] == list(range(2, 11))
    model = spec.make({"n_clusters": 4})
    assert model.n_init == 10 and model.random_state == 42


def test_grid_runs_every_configuration(grid, blobs):
    X, _ = blobs
    table = grid.run(X, [counting_spec()])
    assert len(table) == 3
    assert grid.results_path.name == "clustering_teste.csv"
    assert {"n_iter", "wss", "silhouette", "davies_bouldin", "dunn", "fit_seconds"} <= set(table.columns)
    # Os blobs têm 3 grupos: a silhueta é máxima em k = 3
    best = table.loc[table["silhouette"].idxmax(), "params"]
    assert json.loads(best) == {"n_clusters": 3}
    # A WSS medida é a inertia do k-means (ajustado em todos os pontos)
    for row in table.itertuples():
        labels = grid.labels(row.model, row.config_id)
        assert len(labels) == len(X)
        assert row.wss == pytest.approx(within_cluster_ss(X, labels))


def test_grid_resumes_without_refitting(grid, blobs):
    X, _ = blobs
    first = grid.run(X, [counting_spec()])
    assert CountingKMeans.fits == 3
    second = grid.run(X, [counting_spec((2, 3, 4, 5))])   # só k = 5 é novo
    assert CountingKMeans.fits == 4
    assert len(second) == 4
    pd.testing.assert_frame_equal(second.iloc[:3].reset_index(drop=True), first)


def test_grid_recompute_starts_again(grid, blobs):
    X, _ = blobs
    grid.run(X, [counting_spec()])
    table = grid.run(X, [counting_spec((2,))], recompute=True)
    assert CountingKMeans.fits == 4
    assert len(table) == 1


def test_grid_refits_a_configuration_without_labels(grid, blobs):
    X, _ = blobs
    table = grid.run(X, [counting_spec()])
    lost = table.iloc[0]
    grid._labels_path(lost["model"], lost["config_id"]).unlink()
    again = grid.run(X, [counting_spec()])
    assert CountingKMeans.fits == 4
    assert len(again) == 3                                  # sem linhas repetidas
    assert again["config_id"].is_unique


def test_grid_fits_on_a_sample_but_labels_every_point(grid, blobs):
    X, _ = blobs
    spec = ModelSpec("GMM", partial(GaussianMixture, random_state=0),
                     {"n_components": [3]}, max_train_rows=200)
    table = grid.run(X, [spec])
    row = table.iloc[0]
    assert row["n_fit"] == 200 and row["n_points"] == len(X)
    assert {"bic", "aic", "converged"} <= set(table.columns)
    assert len(grid.labels(row["model"], row["config_id"])) == len(X)


def test_grid_labels_must_exist(grid):
    with pytest.raises(FileNotFoundError):
        grid.labels("k-means", "nao_existe")
