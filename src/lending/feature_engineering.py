"""
Engenharia de features - Fase 3 do enunciado (Model Selection)

Requisitos do enunciado que este módulo cumpre:
    - "Engineer at least 10 meaningful new features."
    - "For each, record its source columns, formula, rationale, missing-value handling, and
      eligibility for each prediction time": cada feature é um FeatureSpec com estes campos;
      catalog() devolve a ficha de todas (docs/03_features/ e secção 3 do relatório).
    - "Do not keep flags, ratios (...) computed from excluded fields, for example (...)
      installment-to-income in rate regression": uma feature herda a elegibilidade das
      colunas de onde vem ("só vale o que já nasceu"). O FeatureBuilder verifica, com
      forbidden_columns (features.py), que nenhuma feature de uma tarefa usa uma coluna
      proibida nessa tarefa, e dá erro se usar.

Como entra no pipeline:
    As features são cálculos linha a linha (rácios, diferenças de datas, indicadores) e não
    aprendem nada com os dados, por isso não passa informação da validação para o treino.
    Mesmo assim, o FeatureBuilder é um transformer do scikit-learn e fica DENTRO do pipeline,
    antes do build_preprocessor, para ser aplicado sempre da mesma forma no treino, na
    validação e no teste final:

        features = add_engineered(TaskFeatures("classification"))
        pipe = Pipeline([("features", FeatureBuilder("classification")),
                         ("pre", build_preprocessor(features)),
                         ("model", ...)])
        pipe.fit(train[input_columns(TaskFeatures("classification"))], y_train)

    Valores em falta: uma divisão por zero (ou por um valor negativo) e uma data impossível
    dão NaN, nunca infinito nem 0. Os NaN são tratados pelo pipeline como os das outras
    colunas: imputação pela mediana do treino de cada fold + indicador de "em falta".

Unidades: as percentagens seguem a convenção do schema (0 a 100, como dti e int_rate).
"""

import copy
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from lending.features import TASKS, forbidden_columns

# Momentos da linha temporal (docs/01_formulacao_problema/momentos_previsao.md)
MOMENTS = {
    "T0": "T0 Pedido",
    "T2": "T2 Pricing",
    "T4": "T4 Concessão",
}


@dataclass(frozen=True)
class FeatureSpec:
    """
    Definição ("ficha") de uma feature nova, com tudo o que o enunciado pede registar

    Attributes:
        name (str): Nome da coluna criada
        sources (tuple[str]): Colunas de origem
        formula (str): Fórmula, em texto
        rationale (str): Porque é útil
        missing (str): O que acontece aos valores em falta e aos casos impossíveis
        moment (str): Momento a partir do qual o valor existe ("T0", "T2" ou "T4")
        tasks (tuple[str]): Tarefas em que é elegível
        compute (Callable[[pd.DataFrame], pd.Series]): Cálculo, linha a linha
        kind (str): Grupo do pipeline em que entra ("numeric" ou "categorical")
        author (str): Quem a propôs (para a divisão do trabalho no relatório)
    """
    name: str
    sources: tuple
    formula: str
    rationale: str
    missing: str
    moment: str
    tasks: tuple
    compute: Callable
    kind: str = "numeric"
    author: str = ""


# =============================================================================
# Funções auxiliares (sempre NaN em vez de infinito)
# =============================================================================

def safe_ratio(numerator, denominator):
    """
    numerator / denominator, com NaN quando o denominador falta ou é <= 0

    Parameters:
        numerator, denominator (pd.Series): Colunas numéricas

    Returns:
        pd.Series: float64; nunca infinito
    """
    numerator = numerator.astype("float64")
    denominator = denominator.astype("float64")
    return numerator / denominator.where(denominator > 0)


def months_between(later, earlier):
    """
    Número de meses de calendário entre duas datas mês/ano (later - earlier)

    As duas colunas são datas do tipo "mês/ano" (o dia é sempre 1), por isso a diferença
    é feita em meses inteiros, sem arredondamentos de dias.

    Returns:
        pd.Series: float64; NaN se alguma data faltar
    """
    later, earlier = pd.to_datetime(later), pd.to_datetime(earlier)
    months = (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)
    return months.astype("float64")


# =============================================================================
# Cálculos de cada feature
# =============================================================================

def _loan_to_income(df):
    return safe_ratio(df["loan_amnt"], df["annual_inc"])


def _installment_to_income(df):
    # Prestação mensal em % do rendimento mensal (mesma unidade que o dti)
    return 100 * safe_ratio(df["installment"], df["annual_inc"] / 12)


def _dti_with_loan(df):
    # O dti da LendingClub não inclui o próprio empréstimo pedido
    return df["dti"].astype("float64") + _installment_to_income(df)


def _credit_history_years(df):
    months = months_between(df["issue_d"], df["earliest_cr_line"])
    return (months.where(months >= 0) / 12).astype("float64")


def _issue_year(df):
    return pd.to_datetime(df["issue_d"]).dt.year.astype("float64")


# =============================================================================
# Catálogo das features (10 no total: 5 de cada elemento do grupo)
# =============================================================================

FEATURES = [
    # ------------------------------------------------------------------ João (5)
    FeatureSpec(
        name="loan_to_income",
        sources=("loan_amnt", "annual_inc"),
        formula="loan_amnt / annual_inc",
        rationale="Peso do montante pedido face ao rendimento: o mesmo montante é um esforço "
                  "diferente para rendimentos diferentes. Na EDA, o montante e o rendimento "
                  "estão correlacionados (ρ = 0,50), mas nenhum dos dois, sozinho, mede o esforço.",
        missing="NaN se annual_inc faltar ou for <= 0 (sem divisão por zero); depois, "
                "imputação pela mediana do treino + indicador no pipeline.",
        moment="T0",
        tasks=("regression", "classification", "clustering"),
        compute=_loan_to_income,
        author="João",
    ),
    FeatureSpec(
        name="installment_to_income",
        sources=("installment", "annual_inc"),
        formula="100 × installment / (annual_inc / 12)",
        rationale="Prestação mensal em % do rendimento mensal: mede o peso do empréstimo no "
                  "orçamento, que nem a prestação nem o rendimento medem sozinhos (na EDA, a "
                  "prestação em valor absoluto não tem relação com o default).",
        missing="NaN se annual_inc faltar ou for <= 0; imputação + indicador no pipeline.",
        moment="T2",
        tasks=("classification",),   # proibida na regressão (enunciado): usa installment
        compute=_installment_to_income,
        author="João",
    ),
    FeatureSpec(
        name="dti_with_loan",
        sources=("dti", "installment", "annual_inc"),
        formula="dti + 100 × installment / (annual_inc / 12)",
        rationale="O dti da LendingClub exclui o empréstimo pedido; somar a nova prestação dá o "
                  "endividamento do mutuário depois de o receber. Nota: é uma combinação linear "
                  "de dti e installment_to_income; em modelos lineares e no kNN usa-se só uma das "
                  "duas (decisão tomada na validação cruzada).",
        missing="NaN se dti, installment ou annual_inc faltarem; imputação + indicador.",
        moment="T2",
        tasks=("classification",),
        compute=_dti_with_loan,
        author="João",
    ),
    FeatureSpec(
        name="credit_history_years",
        sources=("issue_d", "earliest_cr_line"),
        formula="(issue_d − earliest_cr_line) em meses / 12",
        rationale="Antiguidade do historial de crédito na data de concessão, como pede o "
                  "enunciado para o classificador. Um historial curto dá menos informação sobre "
                  "o comportamento do mutuário.",
        missing="NaN se alguma das datas faltar ou se a diferença for negativa (data "
                "impossível); imputação + indicador.",
        moment="T4",
        tasks=("classification",),   # regressão: issue_d é uma data futura face ao pedido
        compute=_credit_history_years,
        author="João",
    ),
    FeatureSpec(
        name="issue_year",
        sources=("issue_d",),
        formula="ano de issue_d",
        rationale="Capta mudanças de política e de contexto económico ao longo do tempo (na EDA, "
                  "a população de 2013–2014 é mais arriscada do que a de 2007–2011). O teste "
                  "(2015) está fora dos anos de treino; como o pipeline limita cada variável aos "
                  "percentis do treino, um ano posterior é tratado como o último ano do treino "
                  "(a tendência não é extrapolada). A validação temporal, em que cada fold valida "
                  "um ano que o treino não tem, mostra se a feature ajuda; a decisão de a manter "
                  "é tomada aí.",
        missing="Sem valores em falta (issue_d existe sempre).",
        moment="T4",
        tasks=("classification",),   # condicional (⚠️): permitida no classificador se justificada
        compute=_issue_year,
        author="João",
    ),
    # COLOCAR AQUI AS OUTRAS 5
]


# =============================================================================
# Transformer e funções de apoio
# =============================================================================

def specs_for(task, specs=None):
    """
    Features elegíveis numa tarefa

    Parameters:
        task (str): "regression", "classification" ou "clustering"
        specs (list[FeatureSpec] | None): Por omissão, FEATURES

    Returns:
        list[FeatureSpec]
    """
    if task not in TASKS:
        raise ValueError(f"Tarefa desconhecida: {task!r}; use uma de {TASKS}.")
    return [s for s in (FEATURES if specs is None else specs) if task in s.tasks]


def check_leakage(task, specs=None):
    """
    Dá erro se alguma feature elegível na tarefa usar uma coluna proibida nessa tarefa

    Aplica a regra do enunciado: rácios ou indicadores calculados a partir de colunas
    excluídas continuam proibidos (ex.: installment_to_income na regressão).

    Raises:
        ValueError: Com a feature e as colunas proibidas encontradas
    """
    for spec in specs_for(task, specs):
        leaked = forbidden_columns(task, list(spec.sources))
        if leaked:
            raise ValueError(f"A feature {spec.name!r} usa colunas proibidas em {task}: {leaked}")


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Acrescenta ao DataFrame as features novas elegíveis numa tarefa

    Não aprende nada no fit (os cálculos são linha a linha): o fit só confirma que as
    colunas de origem existem e que nenhuma feature usa colunas proibidas na tarefa.

    Attributes:
        task (str): "regression", "classification" ou "clustering"
        specs (list[FeatureSpec] | None): Por omissão, FEATURES
        names_ (list[str]): Nomes das features criadas (depois do fit)

    Methods:
        fit(X), transform(X), get_feature_names_out()

    Exemplo:
        cols = input_columns(TaskFeatures("classification"))   # inclui issue_d, earliest_cr_line
        X = FeatureBuilder("classification").fit_transform(cls[cols])
    """

    def __init__(self, task, specs=None):
        self.task = task
        self.specs = specs

    def fit(self, X, y=None):
        active = specs_for(self.task, self.specs)
        check_leakage(self.task, self.specs)
        missing = sorted({c for s in active for c in s.sources} - set(X.columns))
        if missing:
            raise ValueError(f"Faltam colunas de origem nos dados: {missing}")
        clash = [s.name for s in active if s.name in X.columns]
        if clash:
            raise ValueError(f"Já existem colunas com o nome destas features: {clash}")
        self.names_ = [s.name for s in active]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = X.copy()
        for spec in specs_for(self.task, self.specs):
            X[spec.name] = spec.compute(X).replace([np.inf, -np.inf], np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(self.feature_names_in_) + self.names_, dtype=object)


def add_engineered(task_features, specs=None):
    """
    Cópia de um TaskFeatures com as features novas da tarefa nos grupos certos

    Assim, o build_preprocessor (preprocessing.py) trata as features novas como as outras
    colunas: clipping, imputação + indicador e padronização (numéricas) ou one-hot
    (categorias).

    Parameters:
        task_features (TaskFeatures): Colunas de entrada de uma tarefa
        specs (list[FeatureSpec] | None): Por omissão, FEATURES

    Returns:
        TaskFeatures: Cópia com as listas numeric/categorical alargadas
    """
    extended = copy.copy(task_features)
    active = specs_for(task_features.task, specs)
    extended.numeric = list(task_features.numeric) + [s.name for s in active if s.kind == "numeric"]
    extended.categorical = (list(task_features.categorical)
                            + [s.name for s in active if s.kind == "categorical"])
    return extended


def input_columns(task_features, specs=None):
    """
    Colunas a passar ao pipeline: as de entrada da tarefa + as de origem das features novas

    Algumas fontes não entram diretamente como preditores (ex.: issue_d e earliest_cr_line,
    que só entram transformadas), mas o FeatureBuilder precisa delas para calcular as
    features. Depois do FeatureBuilder, o build_preprocessor descarta-as (remainder="drop").

    Parameters:
        task_features (TaskFeatures): Colunas de entrada ORIGINAIS da tarefa (sem add_engineered)
        specs (list[FeatureSpec] | None): Por omissão, FEATURES

    Returns:
        list[str]: Sem repetições, pela ordem: colunas da tarefa, depois as fontes em falta
    """
    active = specs_for(task_features.task, specs)
    new_names = {s.name for s in active}
    base = [c for c in task_features.columns if c not in new_names]
    sources = [c for s in active for c in s.sources]
    return list(dict.fromkeys(base + sources))


def catalog(specs=None):
    """
    Ficha de todas as features: fontes, fórmula, motivo, valores em falta e elegibilidade

    Returns:
        pd.DataFrame: Uma linha por feature; ✅/❌ por tarefa
    """
    rows = []
    for s in (FEATURES if specs is None else specs):
        row = {"feature": s.name, "autor": s.author, "fontes": ", ".join(s.sources),
               "fórmula": s.formula, "momento": MOMENTS.get(s.moment, s.moment)}
        row.update({task: "✅" if task in s.tasks else "❌" for task in TASKS})
        row.update({"motivo": s.rationale, "valores em falta": s.missing})
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")
