"""
Limpeza do ficheiro accepted como um só objeto, para aplicar ao teste as regras do desenvolvimento

Motivo:
    As regras de limpeza foram decididas no desenvolvimento (notebook 01, secções 4 a 8).
    O teste final tem de receber exatamente as mesmas regras, sem as rever: se fossem
    ajustadas olhando para o teste, o teste deixava de ser uma avaliação independente.
    Esta classe junta as regras pela ordem em que foram aplicadas, para que o mesmo código
    limpe os dois conjuntos.

Passos, por esta ordem (secção do notebook entre parênteses):
    1. Conversão de tipos com o SchemaParser (4)
    2. Zeros que significam "nunca aconteceu" -> valor em falta (6.1)
    3. Exclusão dos empréstimos com condições modificadas, registada no log (8.1)
    4. Códigos de teto 9 999 999 -> valor em falta (8.2)

Nenhum passo aprende nada com os dados: são regras fixas, aplicadas linha a linha.
"""

import pandas as pd

from lending.cleaning import (GRADES_ABOVE_MODIFIED_RATE, INSTALLMENT_TOLERANCE, MODIFIED_RATE,
                              find_modified_terms)
from lending.quality import recode_cap_codes, recode_disguised_missing
from lending.schema import SchemaParser


class AcceptedCleaner:
    """
    Aplica ao accepted, em texto, as regras de limpeza fixadas no desenvolvimento

    Attributes:
        parser (SchemaParser): Conversão de tipos
        tolerance (float): Diferença máxima aceite na prestação, em USD

    Methods:
        exclude_modified(df, log): Retira os empréstimos com condições modificadas
        clean(raw, log): Aplica os 4 passos e devolve o conjunto limpo e as correções feitas
    """

    def __init__(self, parser=None, tolerance=INSTALLMENT_TOLERANCE):
        self.parser = SchemaParser() if parser is None else parser
        self.tolerance = tolerance

    def exclude_modified(self, df, log):
        """
        Retira os empréstimos cuja taxa ou prestação não são as originais (cleaning.py)

        Parameters:
            df (pd.DataFrame): Dados já convertidos
            log (ExclusionLog): Registo do conjunto (é acrescentado)

        Returns:
            pd.DataFrame: As linhas que ficam
        """
        modified = find_modified_terms(df, tolerance=self.tolerance)["modified"]
        return log.apply(
            df, modified,
            step="Condições modificadas depois da concessão",
            kind="remoção",
            rule=(f"int_rate = {MODIFIED_RATE:.2f} num grau {GRADES_ABOVE_MODIFIED_RATE[0]} a "
                  f"{GRADES_ABOVE_MODIFIED_RATE[-1]}, ou |installment - fórmula de amortização| "
                  f"> {self.tolerance:g} USD"),
            reason="Taxa ou prestação alteradas depois da concessão; o enunciado pede os "
                   "valores originais",
        )

    def clean(self, raw, log):
        """
        Aplica os 4 passos da limpeza

        Parameters:
            raw (pd.DataFrame): Dados em texto, só com linhas que são empréstimos
            log (ExclusionLog): Registo do conjunto (é acrescentado no passo 3)

        Returns:
            tuple[pd.DataFrame, pd.Series]: Conjunto limpo e número de valores corrigidos
                em cada coluna (passos 2 e 4)
        """
        df = self.parser.parse(raw)
        df, disguised = recode_disguised_missing(df)
        df = self.exclude_modified(df, log)
        df, capped = recode_cap_codes(df)
        return df, pd.concat([disguised, capped])
