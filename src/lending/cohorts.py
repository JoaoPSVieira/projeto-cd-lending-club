"""
Coortes de cada tarefa e alvo da classificação

As regras vêm da Fase 1 (docs/01_formulacao_problema/coorte_alvo.md e desenho_split.md)
e são determinísticas: não dependem de nenhuma estatística dos dados. O mesmo
CohortBuilder constrói as coortes do desenvolvimento e, no fim, as do teste.

Coortes:
    Regressão       todos os empréstimos com int_rate (36 e 60 meses, incluindo Current)
    Classificação   36 meses, prazo terminado (treino 2007-06 a 2011-12; 2012-2014 é
                    embargo), loan_status preenchido e diferente de Current
    Clustering      todos os empréstimos do conjunto (mutuários); os estados juntam
                    também os pedidos rejeitados

Alvo da classificação (3 classes do enunciado):
    Good standing   Fully Paid
    At risk         In Grace Period, Late (16-30 days), Late (31-120 days)
    Default         Charged Off, Default
    Os "Does not meet the credit policy" entram pelo desfecho (Fully Paid / Charged Off);
    a indicação "does not meet" nunca é usada como preditor.
"""

import pandas as pd

from lending import config
from lending.partition import TemporalPartitioner

TARGET_CLASSES = {
    "Fully Paid": "Good standing",
    "Does not meet the credit policy. Status:Fully Paid": "Good standing",
    "In Grace Period": "At risk",
    "Late (16-30 days)": "At risk",
    "Late (31-120 days)": "At risk",
    "Charged Off": "Default",
    "Default": "Default",
    "Does not meet the credit policy. Status:Charged Off": "Default",
}
CLASS_ORDER = ["Good standing", "At risk", "Default"]

# Desfecho ainda desconhecido: fora da classificação (enunciado)
UNKNOWN_OUTCOME = ["Current"]

CLASSIFICATION_TERM = 36
TARGET_COLUMN = "target"


def make_target(loan_status):
    """
    Converte loan_status nas 3 classes do enunciado

    Parameters:
        loan_status (pd.Series): Estado de cada empréstimo (sem Current nem valores em falta)

    Returns:
        pd.Series: Categórica ordenada (Good standing < At risk < Default)

    Raises:
        ValueError: Se aparecer um estado que não está em TARGET_CLASSES
    """
    target = loan_status.map(TARGET_CLASSES)
    unknown = loan_status.notna() & target.isna()
    if unknown.any() or loan_status.isna().any():
        raise ValueError(f"Estados sem classe: {loan_status[unknown].unique().tolist()}; "
                         f"em falta: {int(loan_status.isna().sum())}")
    return target.astype(pd.CategoricalDtype(CLASS_ORDER, ordered=True)).rename(TARGET_COLUMN)


class CohortBuilder:
    """
    Constrói a coorte de cada tarefa e regista cada passo no registo de exclusões

    Attributes:
        classification_partitioner (TemporalPartitioner): Treino / embargo / teste da
            classificação (config.CLASSIFICATION_PARTITIONS)

    Methods:
        regression(df, log): Coorte da regressão
        classification(df, log, partition): Coorte da classificação, com a coluna target
        clustering(df, log): Coorte do clustering de mutuários
    """

    def __init__(self, classification_partitions=None):
        partitions = (config.CLASSIFICATION_PARTITIONS if classification_partitions is None
                      else classification_partitions)
        self.classification_partitioner = TemporalPartitioner(partitions)

    def regression(self, df, log):
        """
        Todos os empréstimos com int_rate preenchido (o alvo da regressão)

        Parameters:
            df (pd.DataFrame): Conjunto limpo
            log (ExclusionLog): Registo da tarefa (é acrescentado)

        Returns:
            pd.DataFrame: Coorte da regressão
        """
        return log.apply(df, df["int_rate"].isna(),
                         step="Sem taxa de juro", kind="remoção",
                         rule="int_rate em falta",
                         reason="int_rate é o alvo da regressão")

    def classification(self, df, log, partition="train"):
        """
        Empréstimos de 36 meses com o prazo terminado e o desfecho conhecido, com o alvo

        Passos, por esta ordem:
            1. Prazo de 60 meses (fora do âmbito: não terminam a tempo)
            2. Fora da partição pedida (ex.: no treino, o embargo de 2012-2014)
            3. Desfecho desconhecido (Current ou loan_status em falta)

        Parameters:
            df (pd.DataFrame): Conjunto limpo
            log (ExclusionLog): Registo da tarefa (é acrescentado)
            partition (str): "train" ou "test" (config.CLASSIFICATION_PARTITIONS)

        Returns:
            pd.DataFrame: Coorte, com a coluna target (3 classes)
        """
        cohort = log.apply(df, df["term"] != CLASSIFICATION_TERM,
                           step="Prazo de 60 meses", kind="remoção",
                           rule=f"term != {CLASSIFICATION_TERM}",
                           reason="Não terminam a tempo de o desfecho ser observado "
                                  "(classificação só para 36 meses, desenho_split.md)")

        labels = self.classification_partitioner.assign(cohort["issue_d"])
        start, end = config.CLASSIFICATION_PARTITIONS[partition]
        reason = ("Embargo: em 01/01/2015 o prazo ainda não tinha terminado, o rótulo não "
                  "era observável na data de treino simulada" if partition == "train"
                  else f"Só os empréstimos do período de {partition}")
        cohort = log.apply(cohort, labels != partition,
                           step=f"Fora do período de {partition}", kind="partição",
                           rule=f"issue_d fora de {start} a {end}", reason=reason)

        unknown = cohort["loan_status"].isin(UNKNOWN_OUTCOME) | cohort["loan_status"].isna()
        cohort = log.apply(cohort, unknown,
                           step="Desfecho desconhecido", kind="remoção",
                           rule="loan_status = Current ou em falta",
                           reason="O desfecho final ainda não é conhecido (enunciado)")

        return cohort.assign(**{TARGET_COLUMN: make_target(cohort["loan_status"])})

    def clustering(self, df, log):
        """
        Clustering de mutuários: todo o conjunto (nenhum filtro adicional)

        Regista o passo com 0 linhas para a tabela de exclusões mostrar que a coorte é o
        conjunto limpo inteiro.
        """
        return log.apply(df, pd.Series(False, index=df.index),
                         step="Coorte do clustering de mutuários", kind="remoção",
                         rule="nenhum filtro adicional",
                         reason="Usa todos os empréstimos do desenvolvimento (coorte_alvo.md)")
