"""
Coortes de cada tarefa e alvo da classificação

As regras vêm da Fase 1 (docs/01_formulacao_problema/coorte_alvo.md e desenho_split.md),
com as decisões do grupo tomadas na limpeza (docs/02_preprocessamento/coortes.md), e são
determinísticas: não dependem de nenhuma estatística dos dados. O mesmo CohortBuilder
constrói as coortes do desenvolvimento e, no fim, as do teste.

Coortes:
    Regressão       todos os empréstimos com int_rate (36 e 60 meses, incluindo Current),
                    treino 2013-01 a 2014-12 (antes de 2013 faltam 37 colunas do bureau)
    Classificação   36 meses, prazo terminado (treino 2007-06 a 2011-12; 2012-2014 é
                    embargo), desfecho conhecido e final (sem Current nem At risk)
    Clustering      todos os empréstimos do conjunto (mutuários); os estados juntam
                    também os pedidos rejeitados

Alvo da classificação:
    O enunciado define 3 classes:
        Good standing   Fully Paid
        At risk         In Grace Period, Late (16-30 days), Late (31-120 days)
        Default         Charged Off, Default
    A classificação é binária (Good standing vs Default): At risk só existe em empréstimos
    ainda ativos e tem 0 casos no treino (regra de observabilidade). Os At risk saem da
    coorte num passo registado, como os Current.
    Os "Does not meet the credit policy" entram pelo desfecho (Fully Paid / Charged Off);
    a indicação "does not meet" nunca é usada como preditor.
"""

import pandas as pd

from lending import config
from lending.partition import TemporalPartitioner

# Mapeamento do enunciado (3 classes). Fica completo para que cada estado tenha uma classe
# conhecida; a classe At risk é depois retirada da coorte.
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

# Desfecho ainda desconhecido: fora da classificação (enunciado)
UNKNOWN_OUTCOME = ["Current"]

# Estados intermédios (empréstimo ativo e em atraso): fora da classificação binária
AT_RISK_STATUSES = [status for status, cls in TARGET_CLASSES.items() if cls == "At risk"]

CLASSIFICATION_TERM = 36


def make_target(loan_status):
    """
    Converte loan_status nas classes da classificação (config.CLASS_ORDER)

    Parameters:
        loan_status (pd.Series): Estado de cada empréstimo (sem Current, At risk nem valores
            em falta)

    Returns:
        pd.Series: Categórica ordenada (Good standing < Default), com o nome config.CLASS_COL

    Raises:
        ValueError: Se aparecer um estado sem classe ou com uma classe fora de CLASS_ORDER
    """
    target = loan_status.map(TARGET_CLASSES)
    invalid = ~target.isin(config.CLASS_ORDER)
    if invalid.any():
        raise ValueError(f"Estados sem classe na classificação: "
                         f"{loan_status[invalid].unique().tolist()}")
    return (target.astype(pd.CategoricalDtype(config.CLASS_ORDER, ordered=True))
            .rename(config.CLASS_COL))


class CohortBuilder:
    """
    Constrói a coorte de cada tarefa e regista cada passo no registo de exclusões

    Attributes:
        regression_partitioner (TemporalPartitioner): Antes do bureau / treino / teste da
            regressão (config.REGRESSION_PARTITIONS)
        classification_partitioner (TemporalPartitioner): Treino / embargo / teste da
            classificação (config.CLASSIFICATION_PARTITIONS)

    Methods:
        regression(df, log, partition): Coorte da regressão
        classification(df, log, partition): Coorte da classificação, com a coluna do alvo
        clustering(df, log): Coorte do clustering de mutuários
    """

    def __init__(self, regression_partitions=None, classification_partitions=None):
        self.regression_partitions = (config.REGRESSION_PARTITIONS
                                      if regression_partitions is None else regression_partitions)
        self.classification_partitions = (config.CLASSIFICATION_PARTITIONS
                                          if classification_partitions is None
                                          else classification_partitions)
        self.regression_partitioner = TemporalPartitioner(self.regression_partitions)
        self.classification_partitioner = TemporalPartitioner(self.classification_partitions)

    def regression(self, df, log, partition="train"):
        """
        Empréstimos do período pedido com int_rate preenchido (o alvo da regressão)

        Passos, por esta ordem:
            1. Fora da partição pedida (no treino, os anos antes de 2013)
            2. int_rate em falta

        Parameters:
            df (pd.DataFrame): Conjunto limpo
            log (ExclusionLog): Registo da tarefa (é acrescentado)
            partition (str): "train" ou "test" (config.REGRESSION_PARTITIONS)

        Returns:
            pd.DataFrame: Coorte da regressão
        """
        labels = self.regression_partitioner.assign(df["issue_d"])
        start, end = self.regression_partitions[partition]
        reason = ("Antes de 2013 faltam as 37 colunas do bureau registadas só a partir de "
                  "2012 (decisão do grupo, valores_em_falta.md)" if partition == "train"
                  else f"Só os empréstimos do período de {partition}")
        cohort = log.apply(df, labels != partition,
                           step=f"Fora do período de {partition} da regressão", kind="partição",
                           rule=f"issue_d fora de {start} a {end}", reason=reason)

        return log.apply(cohort, cohort["int_rate"].isna(),
                         step="Sem taxa de juro", kind="remoção",
                         rule="int_rate em falta",
                         reason="int_rate é o alvo da regressão")

    def classification(self, df, log, partition="train"):
        """
        Empréstimos de 36 meses com o prazo terminado e o desfecho final conhecido, com o alvo

        Passos, por esta ordem:
            1. Prazo de 60 meses (fora do âmbito: não terminam a tempo)
            2. Fora da partição pedida (ex.: no treino, o embargo de 2012-2014)
            3. Desfecho desconhecido (Current ou loan_status em falta)
            4. Classe At risk (fora da classificação binária)

        Parameters:
            df (pd.DataFrame): Conjunto limpo
            log (ExclusionLog): Registo da tarefa (é acrescentado)
            partition (str): "train" ou "test" (config.CLASSIFICATION_PARTITIONS)

        Returns:
            pd.DataFrame: Coorte, com a coluna config.CLASS_COL (Good standing / Default)
        """
        cohort = log.apply(df, df["term"] != CLASSIFICATION_TERM,
                           step="Prazo de 60 meses", kind="remoção",
                           rule=f"term != {CLASSIFICATION_TERM}",
                           reason="Não terminam a tempo de o desfecho ser observado "
                                  "(classificação só para 36 meses, desenho_split.md)")

        labels = self.classification_partitioner.assign(cohort["issue_d"])
        start, end = self.classification_partitions[partition]
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

        cohort = log.apply(cohort, cohort["loan_status"].isin(AT_RISK_STATUSES),
                           step="Classe At risk", kind="remoção",
                           rule="loan_status = In Grace Period ou Late",
                           reason="Classificação binária (decisão do grupo): At risk só existe "
                                  "em empréstimos ativos e tem 0 casos no treino")

        return cohort.assign(**{config.CLASS_COL: make_target(cohort["loan_status"])})

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
