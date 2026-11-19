"""
Feature Engineering - Fase 3 (Model Selection)

Requisitos do enunciado cumpridos:
    - "Engineer at least 10 meaningful new features"
    - "For each, record its source columns, formula, rationale, missing-value handling, and
      eligibility for each prediction time": cada feature é um FeatureSpec com estes campos;
      catalog() devolve a ficha de todas.
    - "Do not keep flags, ratios, text, embeddings, or missingness counts computed from
      excluded fields, for example has_recovery, a hardship-field nonmissing count, or
      installment-to-income in rate regression": uma feature herda a elegibilidade das
      colunas de onde vem. O FeatureBuilder verifica, com forbidden_columns (features.py),
      que nenhuma feature de uma tarefa usa uma coluna proibida nessa tarefa, e se usar dá erro.

Como entra no pipeline:
    As features são cálculos linha a linha e não
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

Unidades: as percentagens seguem a convenção do schema (0 a 100)
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
    Definição de uma feature nova

    Attributes:
        name (str): Nome da coluna criada
        sources (tuple[str]): Colunas de origem
        formula (str): Fórmula
        rationale (str): Porque é útil
        missing (str): O que acontece aos valores em falta e aos casos impossíveis
        moment (str): Momento a partir do qual o valor existe ("T0", "T2" ou "T4")
        tasks (tuple[str]): Tarefas em que é elegível
        compute (Callable[[pd.DataFrame], pd.Series]): Cálculo, linha a linha
        kind (str): Grupo do pipeline em que entra ("numeric" ou "categorical")
        author (str): Quem propôs (para o relatório)
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


# ======================
# = Funções auxiliares =
# ======================

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
    Número de meses entre duas datas mês/ano

    As duas colunas são datas do tipo "mês/ano" (o dia é sempre 1), por isso a diferença
    é feita em meses inteiros, sem arredondamentos de dias.

    Returns:
        pd.Series: float64; NaN se alguma data faltar
    """
    later, earlier = pd.to_datetime(later), pd.to_datetime(earlier)
    months = (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)
    return months.astype("float64")


# ============================
# = Cálculos de cada feature =
# ============================

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


def _revol_to_income(df):
    return safe_ratio(df["revol_bal"], df["annual_inc"])


def _open_acc_pct(df):
    # % das linhas de crédito do historial que ainda estão abertas (0 a 100). Mais contas
    # abertas do que no total é impossível (2 empréstimos de 2007, valores_suspeitos.md): NaN
    pct = 100 * safe_ratio(df["open_acc"], df["total_acc"])
    return pct.where(pct <= 100)


def _ever_delinquent(df):
    # mths_since_last_delinq vazio = nunca houve atraso (os zeros disfarçados de 2007-2008
    # já passaram a vazio na limpeza); delinq_2yrs só cobre os últimos 2 anos
    delinq = df["delinq_2yrs"].astype("float64")
    months = df["mths_since_last_delinq"]
    flag = ((delinq > 0) | months.notna()).astype("float64")
    # Sem nenhuma das duas não se sabe: NaN (não 0)
    return flag.where(delinq.notna() | months.notna())


def _has_public_record(df):
    pub_rec = df["pub_rec"].astype("float64")
    return (pub_rec > 0).astype("float64").where(pub_rec.notna())


def _total_credit_util(df):
    # Saldo de todas as linhas de crédito em % do limite total (0 a 100; pode passar 100)
    return 100 * safe_ratio(df["tot_cur_bal"], df["tot_hi_cred_lim"])


# =========================
# = Catálogo das features =
# =========================

FEATURES = [
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
                  "o comportamento do mutuário. No clustering de mutuários é uma das variáveis "
                  "do perfil que o enunciado dá como exemplo (FICO, DTI, antiguidade do crédito, "
                  "utilização); o clustering é descritivo e não tem momento de previsão.",
        missing="NaN se alguma das datas faltar ou se a diferença for negativa (data "
                "impossível); imputação + indicador.",
        moment="T4",
        # Regressão: issue_d é uma data futura face ao pedido. Clustering: acrescentado na
        # Parte 2 (o issue_d passou a ⚠️ no clustering, só como referência desta feature)
        tasks=("classification", "clustering"),
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
    FeatureSpec(
        name="revol_to_income",
        sources=("revol_bal", "annual_inc"),
        formula="revol_bal / annual_inc",
        rationale="Saldo em crédito rotativo (cartões) face ao rendimento: o revol_util mede a "
                  "parte do limite usada, mas não diz se a dívida é grande para o rendimento. "
                  "No treino da classificação, o quintil mais alto tem 14,9% de default contra "
                  "cerca de 11,3% nos três primeiros; a correlação com o revol_util (ρ = 0,34) e "
                  "com o dti (ρ = 0,40) é moderada, por isso acrescenta informação.",
        missing="NaN se revol_bal faltar ou se annual_inc faltar ou for <= 0 (sem divisão por "
                "zero); depois, imputação pela mediana do treino + indicador no pipeline.",
        moment="T0",
        tasks=("regression", "classification", "clustering"),
        compute=_revol_to_income,
        author="Rodrigo",
    ),
    FeatureSpec(
        name="open_acc_pct",
        sources=("open_acc", "total_acc"),
        formula="100 × open_acc / total_acc",
        rationale="Parte do historial de crédito que continua aberta: o mesmo número de contas "
                  "abertas significa coisas diferentes para quem teve 5 ou 50 contas. Uma "
                  "percentagem alta indica crédito recente ou acumulado. No treino da "
                  "classificação, o quintil mais alto tem 15,5% de default contra 10,5% a 12,1% "
                  "nos outros (relação não linear); a correlação com o open_acc é só ρ = 0,26.",
        missing="NaN se alguma das colunas faltar, se total_acc for <= 0 ou se open_acc for "
                "maior do que total_acc (caso impossível: 2 empréstimos de 2007); imputação "
                "pela mediana do treino + indicador no pipeline.",
        moment="T0",
        tasks=("regression", "classification", "clustering"),
        compute=_open_acc_pct,
        author="Rodrigo",
    ),
    FeatureSpec(
        name="ever_delinquent",
        sources=("delinq_2yrs", "mths_since_last_delinq"),
        formula="1 se delinq_2yrs > 0 ou mths_since_last_delinq preenchido; senão 0",
        rationale="Indica se o mutuário já teve algum atraso de pagamento. O delinq_2yrs só conta "
                  "os últimos 2 anos e o mths_since_last_delinq vazio quer dizer \"nunca\", por "
                  "isso juntar as duas colunas dá o historial completo num valor simples, sem o "
                  "problema de um vazio que significa ausência de evento. 48% dos empréstimos "
                  "da regressão têm 1; no treino da classificação o default é 13,5% com atraso e "
                  "11,7% sem.",
        missing="NaN só se as duas colunas faltarem (não se sabe se houve atraso); "
                "mths_since_last_delinq vazio sozinho significa \"nunca\" e dá 0. Os NaN são "
                "imputados pela mediana do treino + indicador no pipeline.",
        moment="T0",
        tasks=("regression", "classification", "clustering"),
        compute=_ever_delinquent,
        author="Rodrigo",
    ),
    FeatureSpec(
        name="has_public_record",
        sources=("pub_rec",),
        formula="1 se pub_rec > 0; senão 0",
        rationale="Indica se o mutuário tem algum registo público negativo (ex.: falência, "
                  "penhora). A contagem tem uma cauda longa e quase todos têm 0, por isso a "
                  "diferença que importa é entre 0 e pelo menos 1. No treino da classificação, "
                  "o default é 19,9% com registo e 11,9% sem; na regressão, a taxa média é "
                  "14,70% contra 13,93%.",
        missing="NaN se pub_rec faltar; imputação pela mediana do treino + indicador.",
        moment="T0",
        tasks=("regression", "classification", "clustering"),
        compute=_has_public_record,
        author="Rodrigo",
    ),
    FeatureSpec(
        name="total_credit_util",
        sources=("tot_cur_bal", "tot_hi_cred_lim"),
        formula="100 × tot_cur_bal / tot_hi_cred_lim",
        rationale="Utilização de todo o crédito (cartões, empréstimos e hipotecas), e não só do "
                  "rotativo como o revol_util: mede quanto do crédito total ainda está em dívida. "
                  "Na regressão, é a feature nova com mais relação com a taxa a seguir ao "
                  "loan_to_income (ρ = 0,11 com o int_rate), e a correlação com o revol_util é "
                  "só ρ = 0,39.",
        missing="NaN se alguma das colunas faltar ou se tot_hi_cred_lim for <= 0 (15 casos na "
                "regressão); imputação pela mediana do treino + indicador. Valores acima de 100% "
                "(saldo acima do limite, 5 536 casos) ficam; o clipping aos "
                "percentis do treino limita os extremos.",
        moment="T0",
        # As duas colunas são das 37 do bureau que só existem a partir de 2012
        # (valores_em_falta.md, secção 4): vazias no treino da classificação (2007-2011) e em
        # 2007-2012 no clustering. Por isso só entra na regressão (treino 2013-2014)
        tasks=("regression",),
        compute=_total_credit_util,
        author="Rodrigo",
    ),
]


# ==================================
# = Transformer e funções de apoio =
# ==================================

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

    Não aprende nada no fit: o fit só confirma que as colunas de origem
    existem e que nenhuma feature usa colunas proibidas na tarefa.

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

    Assim o build_preprocessor (preprocessing.py) trata as features novas como as outras
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
        pd.DataFrame: Uma linha por feature; ✓ / ✗ por tarefa
    """
    rows = []
    for s in (FEATURES if specs is None else specs):
        row = {"feature": s.name, "autor": s.author, "fontes": ", ".join(s.sources),
               "fórmula": s.formula, "momento": MOMENTS.get(s.moment, s.moment)}
        row.update({task: "✓" if task in s.tasks else "✗" for task in TASKS})
        row.update({"motivo": s.rationale, "valores em falta": s.missing})
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")
