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
"""

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


class RejectedCleaner:
    """
    Converte o rejected em texto para os tipos do schema, com os nomes do accepted

    Attributes:
        parser (SchemaParser): Conversão de tipos com REJECTED_SCHEMA

    Methods:
        parse(raw): Tipos convertidos e colunas renomeadas
        summary(raw, parsed): Tabela de controlo da conversão, com os nomes novos
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
