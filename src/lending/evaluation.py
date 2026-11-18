"""
Métricas de avaliação - Fase 5 (comparação e avaliação dos modelos)

Requisitos do enunciado cumpridos:
    - Regressão: "MAE, RMSE, and R2" (regression_metrics).
    - Classificação: "confusion matrix, precision, recall, F1 (macro and micro averages),
      AUC, and accuracy" (classification_metrics, confusion_table).
    - "Present the mean and the variation of the metric in the cross-validation procedures
      and the mean in the test dataset": média e desvio vêm do runner (experiment.py); a
      incerteza no teste vem do bootstrap (bootstrap_ci, T03, p. 41 e 62).

Nomes das métricas (fixos, iguais nos CSV de resultados):
    regressão:      mae, rmse, r2
    classificação:  f1_macro, f1_micro, precision_macro, precision_micro, recall_macro,
                    recall_micro, accuracy, recall_default, auc

Nota para o relatório: em classificação binária, o F1, a precisão e o recall "micro" são
iguais à accuracy (cada erro é ao mesmo tempo um falso positivo de uma classe e um falso
negativo da outra). Não é um erro de cálculo.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, mean_absolute_error, precision_recall_fscore_support,
                             r2_score, roc_auc_score, root_mean_squared_error)

from lending import config

POSITIVE_LABEL = "Default"
NEGATIVE_LABEL = "Good standing"

REGRESSION_METRICS = ["mae", "rmse", "r2"]
CLASSIFICATION_METRICS = ["f1_macro", "f1_micro", "precision_macro", "precision_micro",
                          "recall_macro", "recall_micro", "accuracy", "recall_default", "auc"]

# Métrica que escolhe a configuração de cada modelo, e se mais alto é melhor
GREATER_IS_BETTER = {"mae": False, "rmse": False, "r2": True, "f1_macro": True, "auc": True}

# Limiares testados na escolha do limiar (secção "Classificação" do plano da Parte 2)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


# ---------------------------------------------------------------------------------------
# Regressão
# ---------------------------------------------------------------------------------------

def regression_metrics(y_true, y_pred):
    """
    Parameters:
        y_true, y_pred (array-like): Taxa real e prevista (pontos percentuais)

    Returns:
        dict[str, float]: mae, rmse (na unidade do alvo) e r2
    """
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred))}


# ---------------------------------------------------------------------------------------
# Classificação
# ---------------------------------------------------------------------------------------

def positive_proba(model, X, positive_label=POSITIVE_LABEL):
    """
    Probabilidade da classe positiva, qualquer que seja a ordem de classes_

    No scikit-learn, as colunas de predict_proba seguem classes_ (ordem alfabética), por isso
    "Default" é a primeira coluna e não a segunda. Esta função escolhe a coluna pelo nome.

    Returns:
        np.ndarray: P(Default) para cada linha
    """
    classes = list(model.classes_)
    if positive_label not in classes:
        raise ValueError(f"O modelo não tem a classe {positive_label!r}; classes: {classes}.")
    return np.asarray(model.predict_proba(X))[:, classes.index(positive_label)]


def labels_from_proba(p_positive, threshold=0.5, positive_label=POSITIVE_LABEL,
                      negative_label=NEGATIVE_LABEL):
    """Classe prevista: positiva quando P(positiva) >= limiar."""
    return np.where(np.asarray(p_positive) >= threshold, positive_label, negative_label)


def classification_metrics(y_true, p_positive, threshold=0.5, positive_label=POSITIVE_LABEL,
                           labels=None):
    """
    Métricas de classificação com um limiar sobre P(positiva)

    Parameters:
        y_true (array-like): Classe real (rótulos em texto)
        p_positive (array-like): P(Default) de cada linha
        threshold (float): Limiar de decisão
        positive_label (str): Classe positiva
        labels (list[str] | None): Classes, por omissão config.CLASS_ORDER

    Returns:
        dict[str, float]: As métricas de CLASSIFICATION_METRICS. A AUC não depende do limiar
            e fica NaN se o y_true só tiver uma classe.
    """
    labels = list(labels or config.CLASS_ORDER)
    negative_label = next(c for c in labels if c != positive_label)
    y_true = np.asarray(y_true)
    y_pred = labels_from_proba(p_positive, threshold, positive_label, negative_label)

    out = {}
    for average in ("macro", "micro"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                     average=average, zero_division=0)
        out[f"f1_{average}"], out[f"precision_{average}"], out[f"recall_{average}"] = f, p, r
    out["accuracy"] = accuracy_score(y_true, y_pred)
    is_pos = y_true == positive_label
    out["recall_default"] = float((y_pred[is_pos] == positive_label).mean()) if is_pos.any() else np.nan
    out["auc"] = (roc_auc_score(is_pos, p_positive) if 0 < is_pos.sum() < len(is_pos) else np.nan)
    return {k: float(out[k]) for k in CLASSIFICATION_METRICS}


def f1_macro_by_threshold(y_true, p_positive, thresholds=THRESHOLD_GRID,
                          positive_label=POSITIVE_LABEL):
    """
    F1 macro (média do F1 das duas classes) para cada limiar, de forma vetorizada

    Returns:
        np.ndarray: Um F1 macro por limiar
    """
    is_pos = np.asarray(y_true) == positive_label
    pred_pos = np.asarray(p_positive)[None, :] >= np.asarray(thresholds)[:, None]
    tp = (pred_pos & is_pos).sum(axis=1)
    fp = (pred_pos & ~is_pos).sum(axis=1)
    fn = (~pred_pos & is_pos).sum(axis=1)
    tn = (~pred_pos & ~is_pos).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f1_pos + f1_neg) / 2


def best_threshold(y_true, p_positive, thresholds=THRESHOLD_GRID, positive_label=POSITIVE_LABEL):
    """
    Limiar que maximiza o F1 macro (em caso de empate, o mais baixo)

    Usa-se nas previsões fora do fold de todos os folds juntos. Fica fixo e nunca é mudado
    com o teste.

    Returns:
        tuple[float, float]: (limiar, F1 macro com esse limiar)
    """
    scores = f1_macro_by_threshold(y_true, p_positive, thresholds, positive_label)
    i = int(np.argmax(scores))
    return float(np.asarray(thresholds)[i]), float(scores[i])


def confusion_table(y_true, y_pred, row_labels=None, col_labels=None):
    """
    Matriz de confusão com nomes (linhas = classe real, colunas = classe prevista)

    As linhas e as colunas podem ser diferentes. No teste de 2015, as linhas têm as três
    classes reais (Good standing, At risk, Default) e as colunas as duas previstas, para se
    ver onde caem os At risk.

    Returns:
        pd.DataFrame: Contagens, com as linhas e colunas pedidas (zeros onde não há casos)
    """
    row_labels = list(row_labels or config.CLASS_ORDER)
    col_labels = list(col_labels or config.CLASS_ORDER)
    table = pd.crosstab(pd.Series(np.asarray(y_true), name="real"),
                        pd.Series(np.asarray(y_pred), name="previsto"))
    return table.reindex(index=row_labels, columns=col_labels, fill_value=0).astype(int)


# ---------------------------------------------------------------------------------------
# Incerteza: bootstrap (T03, p. 41 e 62)
# ---------------------------------------------------------------------------------------

def bootstrap_ci(metric, y_true, y_score, n_boot=1000, alpha=0.05, seed=config.RANDOM_SEED):
    """
    Intervalo de confiança de uma métrica por bootstrap das linhas

    Reamostra as linhas com reposição n_boot vezes, calcula a métrica em cada reamostragem
    e usa os percentis alpha/2 e 1 - alpha/2 (método dos percentis).

    Parameters:
        metric (callable): metric(y_true, y_score) -> float, ex.: mean_absolute_error ou
            lambda y, p: classification_metrics(y, p, limiar)["f1_macro"]
        y_true, y_score (array-like): Valores reais e previsões (ou probabilidades)
        n_boot (int): Número de reamostragens
        alpha (float): 0,05 dá um intervalo a 95%
        seed (int): Semente

    Returns:
        dict[str, float]: estimate (métrica nos dados originais), low, high
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.array([metric(y_true[idx], y_score[idx])
                      for idx in (rng.integers(0, n, n) for _ in range(n_boot))])
    low, high = np.nanpercentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(metric(y_true, y_score)), "low": float(low), "high": float(high)}


def paired_bootstrap_diff(metric, y_true, score_a, score_b, n_boot=1000, alpha=0.05,
                          seed=config.RANDOM_SEED):
    """
    Diferença entre dois modelos (A - B) na mesma métrica, com bootstrap emparelhado

    As duas previsões são reamostradas com as mesmas linhas, por isso a diferença só reflete
    os modelos e não a amostra. Se o intervalo contiver o 0, não há um vencedor claro.

    Returns:
        dict[str, float]: estimate, low, high (da diferença A - B)
    """
    y_true, score_a, score_b = map(np.asarray, (y_true, score_a, score_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(metric(y_true[idx], score_a[idx]) - metric(y_true[idx], score_b[idx]))
    low, high = np.nanpercentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    estimate = metric(y_true, score_a) - metric(y_true, score_b)
    return {"estimate": float(estimate), "low": float(low), "high": float(high)}


def format_mean_std(mean, std, decimals=3):
    """Texto "média ± desvio" para as tabelas do relatório."""
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
