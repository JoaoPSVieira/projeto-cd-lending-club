"""
Ficheiro rejected: schema, nomes das colunas e limpeza

Motivo:
    O rejected tem só 9 colunas, com nomes e formatos diferentes do accepted (ex.:
    "Debt-To-Income Ratio" = "38.64%"). Para comparar aceites e rejeitados (testes de
    hipóteses, clustering de estados), as colunas passam a ter o nome da coluna equivalente
    do accepted e a mesma unidade.

Decisões de desenho:
1 - O mesmo SchemaParser do accepted, com outro schema:
    A conversão de cada tipo é a mesma (ex.: "< 1 year" -> 0 em emp_length), por isso os
    dois ficheiros ficam com valores comparáveis.

2 - Primeiro converte, depois muda o nome:
    O SchemaParser confirma que as colunas são exatamente as do ficheiro original.

Colunas (original -> nome usado):
    Amount Requested      -> amount_requested   USD (montante pedido; no accepted, loan_amnt)
    Application Date      -> application_date   data do pedido (usada na partição)
    Loan Title            -> title              texto livre escrito pelo requerente
    Risk_Score            -> risk_score         pontos
    Debt-To-Income Ratio  -> dti                % (escala 0-100, como no accepted)
    Zip Code              -> zip_code           3 primeiros dígitos ("123xx")
    State                 -> addr_state         sigla do estado
    Employment Length     -> emp_length         anos (0 a 10)
    Policy Code           -> policy_code        código

Limpeza (regras fixas, decididas no desenvolvimento; docs/02_preprocessamento/rejeitados.md):
    Exclusões (registadas no log), sobre os valores originais:
        policy_code = 2         produtos novos não públicos (os aceites são todos 1)
        linhas iguais           o mesmo pedido submetido mais de uma vez
    Correções (valor -> em falta):
        risk_score = 0          valor em falta (enunciado)
        dti em DTI_CODES        códigos (-1 = sem dados; 9999, 99999, 199998 = valores de teto)
        dti > DTI_MAX           artefactos de introdução (ex.: 4042.11%)
        amount_requested = 0    pedido sem montante
    Coluna nova:
        risk_score_scale        "FICO" até 2013-11-05, "VantageScore" a partir de 2013-11-06
"""

import pandas as pd

from lending.schema import SchemaParser

REJECTED_SCHEMA = {
    "Amount Requested": "number",
    "Application Date": "date",
    "Loan Title": "text",
    "Risk_Score": "number",
    "Debt-To-Income Ratio": "percent",
    "Zip Code": "category",
    "State": "category",
    "Employment Length": "emp_length",
    "Policy Code": "number",
}

REJECTED_NAMES = {
    "Amount Requested": "amount_requested",
    "Application Date": "application_date",
    "Loan Title": "title",
    "Risk_Score": "risk_score",
    "Debt-To-Income Ratio": "dti",
    "Zip Code": "zip_code",
    "State": "addr_state",
    "Employment Length": "emp_length",
    "Policy Code": "policy_code",
}

# Tipo e unidade de cada coluna, já com os nomes novos (para o SchemaReport)
REJECTED_SCHEMA_RENAMED = {REJECTED_NAMES[col]: kind for col, kind in REJECTED_SCHEMA.items()}
REJECTED_UNITS = {
    "amount_requested": "USD",
    "application_date": "data (dia)",
    "title": "texto",
    "risk_score": "pontos",
    "dti": "%",
    "zip_code": "categoria",
    "addr_state": "categoria",
    "emp_length": "anos",
    "policy_code": "código",
}


# Códigos que não são um DTI: -1 (sem dados) e valores redondos repetidos (teto do campo)
DTI_CODES = [-1, 9_999, 99_999, 199_998]

# DTI máximo plausível (%): acima disto as dívidas mensais valeriam mais de 10 vezes o
# rendimento, o que indica um rendimento mal introduzido e não um rácio real
DTI_MAX = 1000

# Código de política dos produtos novos não públicos (dicionário de dados)
NEW_PRODUCTS_POLICY_CODE = 2

# Primeiro dia com risk_score na escala VantageScore (501-990); antes, escala FICO (até 850)
VANTAGE_START = pd.Timestamp("2013-11-06")


class RejectedCleaner:
    """
    Converte e limpa o rejected em texto, com os nomes das colunas do accepted

    Attributes:
        parser (SchemaParser): Conversão de tipos com REJECTED_SCHEMA

    Methods:
        parse(raw): Tipos convertidos e colunas renomeadas
        summary(raw, parsed): Tabela de controlo da conversão, com os nomes novos
        clean(raw, log): Conversão, exclusões, correções e escala do risk_score
    """

    def __init__(self):
        self.parser = SchemaParser(REJECTED_SCHEMA)

    def parse(self, raw):
        """
        Parameters:
            raw (pd.DataFrame): Rejected em texto, com as 9 colunas originais

        Returns:
            pd.DataFrame: Mesmo índice, tipos convertidos, colunas com os nomes de REJECTED_NAMES
        """
        return self.parser.parse(raw).rename(columns=REJECTED_NAMES)

    def summary(self, raw, parsed):
        """
        Tabela de controlo da conversão (SchemaParser.summary), uma linha por coluna

        Parameters:
            raw (pd.DataFrame): Dados antes da conversão (nomes originais)
            parsed (pd.DataFrame): Resultado de parse(raw) (nomes novos)

        Returns:
            pd.DataFrame: Índice com os nomes novos
        """
        original = {new: old for old, new in REJECTED_NAMES.items()}
        return (self.parser.summary(raw, parsed.rename(columns=original))
                .rename(index=REJECTED_NAMES))

    def clean(self, raw, log):
        """
        Aplica, por ordem, a conversão, as exclusões e as correções da limpeza

        As exclusões usam os valores originais (convertidos): duas linhas só são o mesmo
        pedido se forem iguais antes das correções, que juntam valores diferentes em NaN.

        Parameters:
            raw (pd.DataFrame): Rejected em texto, com as 9 colunas originais
            log (ExclusionLog): Registo do conjunto (é acrescentado)

        Returns:
            tuple[pd.DataFrame, pd.Series]: Conjunto limpo (com risk_score_scale) e número
                de valores corrigidos em cada regra
        """
        df = self.parse(raw)

        df = log.apply(df, df["policy_code"] == NEW_PRODUCTS_POLICY_CODE,
                       step="Produtos novos não públicos", kind="remoção",
                       rule=f"policy_code = {NEW_PRODUCTS_POLICY_CODE}",
                       reason="Os aceites são todos policy_code 1: comparar só pedidos dos "
                              "mesmos produtos")
        df = log.apply(df, df.duplicated(keep="first"),
                       step="Pedidos repetidos", kind="remoção",
                       rule="linha igual a uma anterior em todas as colunas (fica a primeira)",
                       reason="O mesmo pedido submetido mais de uma vez no mesmo dia; contaria "
                              "como várias rejeições")

        dti_code = df["dti"].isin(DTI_CODES)
        corrections = {
            "risk_score = 0": df["risk_score"] == 0,
            f"dti = {', '.join(f'{c:g}' for c in DTI_CODES)}": dti_code,
            f"dti > {DTI_MAX} (sem os códigos)": (df["dti"] > DTI_MAX) & ~dti_code,
            "amount_requested = 0": df["amount_requested"] == 0,
        }
        df = df.copy()
        for rule, mask in corrections.items():
            column = rule.split()[0]
            df.loc[mask, column] = float("nan")

        df["risk_score_scale"] = (df["application_date"] >= VANTAGE_START).map(
            {False: "FICO", True: "VantageScore"})
        return df, pd.Series({rule: int(mask.sum()) for rule, mask in corrections.items()},
                             name="corrigidos")
