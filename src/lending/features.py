"""
Colunas de entrada de cada tarefa, a partir da tabela de elegibilidade da Fase 1

Motivo:
    A elegibilidade de cada uma das 151 colunas foi decidida na Fase 1, antes de olhar para
    os dados (docs/01_formulacao_problema/elegibilidade_colunas.md). Em vez de copiar essa
    decisão para o código, a tabela é lida diretamente: se for alterada, o pipeline muda
    com ela, e não há duas versões que possam divergir.

Regras do pipeline base (as colunas ✅ de cada tarefa, menos):
    - ⚠️ condicionais (zip_code, addr_state, issue_d, earliest_cr_line na regressão): ficam
      de fora até haver a análise que a condição exige (ex.: fairness da geografia);
    - texto livre (emp_title, title): muitas categorias; só entram como features novas;
    - datas (earliest_cr_line, ...): só entram transformadas (ex.: antiguidade do crédito),
      como features novas na seleção de modelos.

Segurança contra leakage:
    FORBIDDEN_ALWAYS e FORBIDDEN_REGRESSION repetem as listas do enunciado ("Note on data
    leakage"). Se alguma coluna proibida aparecer nas colunas de uma tarefa (ex.: por um
    erro na tabela), o código dá erro.
"""

import re

import pandas as pd

from lending import config
from lending.schema import ACCEPTED_SCHEMA, ACCEPTED_UNITS

ELIGIBILITY_DOC = config.DOCS_DIR / "01_formulacao_problema" / "elegibilidade_colunas.md"

# Coluna da tabela de elegibilidade de cada tarefa (a seguir a Coluna, Descrição e Momento)
TASKS = ("regression", "classification", "clustering")
ELIGIBLE, NOT_ELIGIBLE, CONDITIONAL, TARGET = "✅", "❌", "⚠️", "🎯"

# Proibidas como preditores em qualquer tarefa (enunciado), incluindo o alvo da classificação
FORBIDDEN_ALWAYS = [
    "loan_status",
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
    "debt_settlement_flag", "debt_settlement_flag_date",
]
FORBIDDEN_PREFIXES = ("hardship_", "settlement_")

# Proibidas só na regressão (enunciado): definem ou permitem reconstruir a taxa, ou são
# posteriores ao pricing; int_rate é o alvo
FORBIDDEN_REGRESSION = ["int_rate", "grade", "sub_grade", "installment", "funded_amnt",
                        "funded_amnt_inv", "issue_d"]

# Tipos do schema que o pipeline base não usa diretamente
TEXT_KINDS = ("text",)
DATE_KINDS = ("month_year",)

# Tipos que entram como números e unidades em que se aplica o logaritmo
NUMERIC_KINDS = ("number", "term", "emp_length")
LOG_UNITS = ("USD", "USD/ano")


def load_eligibility(path=ELIGIBILITY_DOC):
    """
    Lê a tabela de elegibilidade (uma linha por coluna do accepted)

    Parameters:
        path (str | Path): Ficheiro Markdown da Fase 1

    Returns:
        pd.DataFrame: Índice = coluna; colunas regression, classification, clustering com
            o símbolo de cada tarefa (✅, ❌, ⚠️ ou 🎯)

    Raises:
        ValueError: Se a tabela não tiver exatamente as colunas do schema do accepted
    """
    rows = {}
    with open(path, encoding="utf-8") as doc:
        for line in doc:
            match = re.match(r"^\| `([a-z0-9_]+)` \|", line)
            if match:
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                rows[match.group(1)] = dict(zip(TASKS, cells[3:6]))

    table = pd.DataFrame.from_dict(rows, orient="index")
    if set(table.index) != set(ACCEPTED_SCHEMA) or len(table) != len(ACCEPTED_SCHEMA):
        raise ValueError("A tabela de elegibilidade não tem as 151 colunas do schema.")
    return table


def forbidden_columns(task, columns):
    """
    Colunas de columns que o enunciado proíbe como preditores na tarefa

    Parameters:
        task (str): "regression", "classification" ou "clustering"
        columns (list[str]): Colunas a verificar

    Returns:
        list[str]: As colunas proibidas encontradas (vazia se não houver nenhuma)
    """
    forbidden = set(FORBIDDEN_ALWAYS)
    if task == "regression":
        forbidden |= set(FORBIDDEN_REGRESSION)
    return [c for c in columns if c in forbidden or c.startswith(FORBIDDEN_PREFIXES)]


class TaskFeatures:
    """
    Colunas de entrada de uma tarefa, agrupadas pelo tratamento que recebem no pipeline

    Attributes:
        task (str): "regression", "classification" ou "clustering"
        money (list[str]): Numéricas em USD (logaritmo antes das outras transformações)
        numeric (list[str]): Restantes numéricas
        categorical (list[str]): Categorias (one-hot)
        excluded (pd.DataFrame): Colunas de fora, com o símbolo da tabela e a razão

    Methods:
        columns: Todas as colunas de entrada (money + numeric + categorical)
    """

    def __init__(self, task, eligibility=None):
        """
        Parameters:
            task (str): "regression", "classification" ou "clustering"
            eligibility (pd.DataFrame | None): Resultado de load_eligibility(); por omissão lê o doc

        Raises:
            ValueError: Se a tarefa não existir ou se uma coluna proibida for elegível
        """
        if task not in TASKS:
            raise ValueError(f"Tarefa desconhecida: {task!r}; use uma de {TASKS}.")
        self.task = task
        status = (load_eligibility() if eligibility is None else eligibility)[task]

        self.money, self.numeric, self.categorical = [], [], []
        excluded = {}
        for col in ACCEPTED_SCHEMA:   # ordem do ficheiro
            kind = ACCEPTED_SCHEMA[col]
            if status[col] != ELIGIBLE:
                excluded[col] = (status[col], {NOT_ELIGIBLE: "não elegível (Fase 1)",
                                               CONDITIONAL: "condicional: fica de fora até "
                                                            "a condição ser analisada",
                                               TARGET: "alvo"}[status[col]])
            elif kind in TEXT_KINDS:
                excluded[col] = (ELIGIBLE, "texto livre: só como feature nova")
            elif kind in DATE_KINDS:
                excluded[col] = (ELIGIBLE, "data: só transformada, como feature nova")
            elif kind in NUMERIC_KINDS and ACCEPTED_UNITS[col] in LOG_UNITS:
                self.money.append(col)
            elif kind in NUMERIC_KINDS:
                self.numeric.append(col)
            else:
                self.categorical.append(col)

        self.excluded = pd.DataFrame.from_dict(excluded, orient="index",
                                               columns=["tabela", "razão"])
        leaked = forbidden_columns(task, self.columns)
        if leaked:
            raise ValueError(f"Colunas proibidas pelo enunciado entre as de {task}: {leaked}")

    @property
    def columns(self):
        """Todas as colunas de entrada, pela ordem money, numeric, categorical."""
        return self.money + self.numeric + self.categorical

    def summary(self):
        """
        Returns:
            pd.Series: Número de colunas em cada grupo
        """
        return pd.Series({"montantes (log)": len(self.money), "numéricas": len(self.numeric),
                          "categorias": len(self.categorical), "excluídas": len(self.excluded)},
                         name=self.task)
