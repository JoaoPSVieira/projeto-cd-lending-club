"""
Dados para o trabalho em R: testes de hipóteses (R/testes_hipoteses.Rmd) e pipeline com PCA
(R/pipeline_pca.Rmd)

Motivo:
    O R não lê os pickles do pandas. Esta classe escolhe, nos conjuntos de desenvolvimento já
    limpos, só as colunas que cada análise usa e grava-as em CSV. Nenhum valor é alterado: as
    decisões de limpeza já foram tomadas no pré-processamento.

Regras:
    - Só dados de desenvolvimento (2007-06 a 2014-12). O teste final (2015) nunca é lido.
    - Hipóteses com o desfecho (default): só o treino da classificação, o único conjunto em
      que o desfecho de cada empréstimo já era observável na data de treino simulada.

Tabelas:
    default    (classificacao_treino.pkl)      dti, purpose, target
    aceites vs rejeitados (aceites_dev.pkl + rejeitados_dev.pkl):
               group, amount, risk_score, date, same_scale_period
    pipeline   (classificacao_treino.pkl)      issue_d, colunas numéricas indicadas, target
               (issue_d define os folds da validação temporal)

Aceites vs rejeitados, colunas equivalentes:
    amount      loan_amnt (aceites) / amount_requested (rejeitados). Limitação: o loan_amnt
                pode refletir reduções feitas pela LendingClub (dicionário de dados)
    risk_score  ponto médio de fico_range_low e fico_range_high (aceites) / risk_score
                (rejeitados). Nos rejeitados, o risk score passa de FICO para VantageScore a
                2013-11-06 (rejected.VANTAGE_START), por isso só é comparável antes disso
    date        issue_d (aceites, mês de financiamento) / application_date (rejeitados, dia
                do pedido)
    same_scale_period
                True nos meses completos antes da mudança de escala (até 2013-10). Usa-se o
                mesmo período de calendário nos dois grupos; os 5 dias de novembro de 2013
                ficam de fora porque o issue_d dos aceites só tem o mês
"""

import pandas as pd

from lending import config
from lending.rejected import VANTAGE_START

# Primeiro mês que deixa de entrar na comparação do risk score (o da mudança de escala)
SAME_SCALE_END = VANTAGE_START.to_period("M").to_timestamp()   # 2013-11-01

DEFAULT_COLUMNS = ["dti", "purpose", config.CLASS_COL]


class HypothesisDataExporter:
    """
    Prepara e grava em CSV as tabelas usadas pelos testes de hipóteses em R

    Attributes:
        out_dir (Path): Pasta de destino (dentro de data/, fora do git)

    Methods:
        default_table(cls_train): Tabela das hipóteses sobre o default
        accepted_vs_rejected(accepted_dev, rejected_dev): Tabela aceites vs rejeitados
        pipeline_table(cls_train, columns): Tabela do pipeline em R
        export(df, name): Grava uma tabela em CSV
    """

    def __init__(self, out_dir=config.HYPOTHESIS_DIR):
        """
        Parameters:
            out_dir (str | Path): Pasta de destino
        """
        self.out_dir = out_dir

    @staticmethod
    def default_table(cls_train):
        """
        Parameters:
            cls_train (pd.DataFrame): Treino da classificação (classificacao_treino.pkl)

        Returns:
            pd.DataFrame: dti, purpose e target, uma linha por empréstimo
        """
        return cls_train[DEFAULT_COLUMNS].reset_index(drop=True)

    @staticmethod
    def accepted_vs_rejected(accepted_dev, rejected_dev):
        """
        Junta aceites e rejeitados com colunas de nome e unidade iguais

        Parameters:
            accepted_dev (pd.DataFrame): Aceites do desenvolvimento (aceites_dev.pkl)
            rejected_dev (pd.DataFrame): Rejeitados do desenvolvimento (rejeitados_dev.pkl)

        Returns:
            pd.DataFrame: group ("aceite"/"rejeitado"), amount, risk_score, date, same_scale_period
        """
        accepted = pd.DataFrame({
            "group": "aceite",
            "amount": accepted_dev["loan_amnt"],
            # O FICO dos aceites é um intervalo de 4 pontos; o dos rejeitados é um valor só
            "risk_score": (accepted_dev["fico_range_low"] + accepted_dev["fico_range_high"]) / 2,
            "date": accepted_dev["issue_d"],
        })
        rejected = pd.DataFrame({
            "group": "rejeitado",
            "amount": rejected_dev["amount_requested"],
            "risk_score": rejected_dev["risk_score"],
            "date": rejected_dev["application_date"],
        })
        table = pd.concat([accepted, rejected], ignore_index=True)
        table["same_scale_period"] = table["date"] < SAME_SCALE_END
        return table

    @staticmethod
    def pipeline_table(cls_train, columns):
        """
        Parameters:
            cls_train (pd.DataFrame): Treino da classificação (classificacao_treino.pkl)
            columns (list[str]): Colunas numéricas de entrada do pipeline

        Returns:
            pd.DataFrame: issue_d, as colunas indicadas e target, uma linha por empréstimo
        """
        return cls_train[["issue_d", *columns, config.CLASS_COL]].reset_index(drop=True)

    def export(self, df, name):
        """
        Parameters:
            df (pd.DataFrame): Tabela a gravar
            name (str): Nome do ficheiro, sem extensão

        Returns:
            Path: Caminho do CSV escrito
        """
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{name}.csv"
        # Datas no formato ISO (AAAA-MM-DD), que o R lê sem ambiguidade
        df.to_csv(path, index=False, date_format="%Y-%m-%d")
        return path
