"""
Parsing determinístico: converter texto em valores com uma regra fixa

Porquê "determinístico":
    O resultado de cada valor depende só desse valor e de uma regra escrita à mão,
    nunca dos outros dados (médias, categorias vistas, etc.). Por isso não aprende nada
    com o conjunto de teste e pode ser aplicado antes do split. O enunciado distingue
    explicitamente este tipo de conversão das transformações aprendidas.

Decisões de desenho:
1 - Os meses são lidos com uma tabela fixa (Jan..Dec):
    O "%b" do strptime depende da língua do sistema (locale). Com a tabela, o resultado
    é igual em qualquer computador.

2 - Valores com formato inesperado dão erro em vez de passarem a NaN:
    Um NaN silencioso esconderia problemas no ficheiro. Só o valor em falta original
    continua em falta.
"""

import pandas as pd

# Abreviaturas inglesas usadas pela LendingClub (ex.: "Dec-2015")
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

MONTH_YEAR_PATTERN = r"(?P<month>[A-Z][a-z]{2})-(?P<year>\d{4})"


def parse_month_year(values):
    """
    Converte texto "Mon-AAAA" (ex.: "Dec-2015") no primeiro dia desse mês

    Parameters:
        values (pd.Series): Texto no formato "Mon-AAAA"; valores em falta são permitidos

    Returns:
        pd.Series: Datas (datetime64) com o mesmo índice; NaT onde o valor estava em falta

    Raises:
        ValueError: Se algum valor preenchido não tiver o formato "Mon-AAAA"
            ou tiver um mês desconhecido
    """
    parts = values.str.extract(MONTH_YEAR_PATTERN)

    # Preenchido no original, mas não reconhecido pelo padrão ou pela tabela de meses
    month = parts["month"].map(MONTHS)
    invalid = values.notna() & month.isna()
    if invalid.any():
        examples = values[invalid].unique()[:5].tolist()
        raise ValueError(f"{invalid.sum()} valores fora do formato 'Mon-AAAA', "
                         f"por exemplo: {examples}")

    return pd.to_datetime(
        pd.DataFrame({"year": parts["year"].astype("float"), "month": month, "day": 1})
    ).rename(values.name)
