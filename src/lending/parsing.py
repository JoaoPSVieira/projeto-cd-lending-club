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
    continua em falta (exceção: texto só com espaços, que é um campo vazio).

3 - Cada função recebe e devolve uma pd.Series com o mesmo índice:
    Assim podem ser aplicadas coluna a coluna pelo SchemaParser.
"""

import pandas as pd

# Abreviaturas inglesas usadas pela LendingClub (ex.: "Dec-2015")
MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

MONTH_YEAR_PATTERN = r"(?P<month>[A-Z][a-z]{2})-(?P<year>\d{4})"

# Número decimal, com sinal e notação científica opcionais (ex.: "10000.0", "-5e-10")
NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

# Prazo do empréstimo, com os espaços que vêm no ficheiro (ex.: " 36 months")
TERM_PATTERN = r"\s*(\d+) months\s*"

# Anos de emprego: os 11 valores do ficheiro, por ordem.
# "< 1 year" -> 0 e "10+ years" -> 10 (o ficheiro não distingue mais de 10 anos)
EMP_LENGTH_YEARS = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
                    "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
                    "8 years": 8, "9 years": 9, "10+ years": 10}

# Indicadores sim/não (o ficheiro usa "y"/"n" e "Y"/"N")
FLAG_VALUES = {"Y": True, "N": False, "y": True, "n": False}

# Data completa do ficheiro rejected (ex.: "2007-05-26")
ISO_DATE_PATTERN = r"\d{4}-\d{2}-\d{2}"

# Percentagem em texto, com o símbolo no fim (ex.: "38.64%", "-1%")
PERCENT_PATTERN = rf"(?P<number>{NUMBER_PATTERN})%"


def parse_month_year(values):
    """
    Converte texto "Mon-AAAA" (ex.: "Dec-2015") no primeiro dia desse mês

    Parameters:
        values (pd.Series): Texto no formato "Mon-AAAA"; valores em falta são permitidos

    Returns:
        pd.Series: Datas (datetime64[ns]) com o mesmo índice; NaT onde o valor estava em falta

    Raises:
        ValueError: Se algum valor preenchido não tiver o formato "Mon-AAAA"
            ou tiver um mês desconhecido
    """
    parts = values.str.extract(MONTH_YEAR_PATTERN)

    # Preenchido no original, mas não reconhecido pelo padrão ou pela tabela de meses
    month = parts["month"].map(MONTHS)
    _raise_if_invalid(values, values.notna() & month.isna(), "'Mon-AAAA'")

    # Unidade fixa: sem ela, uma coluna toda vazia ficaria com uma unidade diferente das outras
    return pd.to_datetime(
        pd.DataFrame({"year": parts["year"].astype("float"), "month": month, "day": 1})
    ).astype("datetime64[ns]").rename(values.name)


def parse_iso_date(values):
    """
    Converte texto "AAAA-MM-DD" (ex.: "2007-05-26") em data

    Parameters:
        values (pd.Series): Texto no formato "AAAA-MM-DD"; valores em falta são permitidos

    Returns:
        pd.Series: Datas (datetime64[ns]) com o mesmo índice; NaT onde o valor estava em falta

    Raises:
        ValueError: Se algum valor preenchido não tiver o formato ou não for uma data válida
    """
    valid = values.isna() | values.str.fullmatch(ISO_DATE_PATTERN)
    _raise_if_invalid(values, ~valid, "'AAAA-MM-DD'")
    # Formato fixo: uma data impossível (ex.: 2015-02-30) dá erro em vez de NaT
    return pd.to_datetime(values, format="%Y-%m-%d").astype("datetime64[ns]")


def parse_percent(values):
    """
    Converte uma percentagem em texto (ex.: "38.64%") no número, na escala 0-100 (38.64)

    A escala 0-100 é a mesma das percentagens do accepted (int_rate, dti, revol_util).

    Parameters:
        values (pd.Series): Texto "<número>%"; valores em falta são permitidos

    Returns:
        pd.Series: float64 com o mesmo índice; NaN onde o valor estava em falta

    Raises:
        ValueError: Se algum valor preenchido não terminar em "%" depois de um número
    """
    number = values.str.extract(PERCENT_PATTERN, expand=False)
    _raise_if_invalid(values, values.notna() & number.isna(), "'<número>%'")
    return number.astype("float64").rename(values.name)


def parse_number(values):
    """
    Converte texto numérico (ex.: "10000.0", "12.99", "1e-10") em float

    Parameters:
        values (pd.Series): Texto numérico; valores em falta são permitidos

    Returns:
        pd.Series: float64 com o mesmo índice; NaN onde o valor estava em falta

    Raises:
        ValueError: Se algum valor preenchido não for um número
    """
    # Valores em falta são aceites; o str.fullmatch devolve False neles
    valid = values.isna() | values.str.fullmatch(NUMBER_PATTERN)
    _raise_if_invalid(values, ~valid, "numérico")
    return values.astype("float64")


def parse_id(values):
    """
    Converte o identificador do empréstimo em inteiro

    Raises:
        ValueError: Se houver ids em falta ou que não sejam inteiros
    """
    if values.isna().any():
        raise ValueError(f"{values.name}: {values.isna().sum()} identificadores em falta.")
    valid = values.str.fullmatch(r"\d+").astype(bool)
    _raise_if_invalid(values, ~valid, "inteiro")
    return values.astype("int64")


def parse_term(values):
    """
    Converte o prazo (ex.: " 36 months") no número de meses (36.0)

    Raises:
        ValueError: Se algum valor preenchido não tiver o formato "<n> months"
    """
    months = values.str.extract(TERM_PATTERN, expand=False)
    _raise_if_invalid(values, values.notna() & months.isna(), "'<n> months'")
    return months.astype("float64").rename(values.name)


def parse_emp_length(values):
    """
    Converte os anos de emprego (ex.: "< 1 year", "10+ years") num número de 0 a 10

    Raises:
        ValueError: Se aparecer um valor que não esteja em EMP_LENGTH_YEARS
    """
    years = values.map(EMP_LENGTH_YEARS)
    _raise_if_invalid(values, values.notna() & years.isna(), "de anos de emprego")
    return years.astype("float64")


def parse_flag(values):
    """
    Converte indicadores "Y"/"N" (ou "y"/"n") em booleano

    Returns:
        pd.Series: dtype "boolean" (aceita valores em falta)

    Raises:
        ValueError: Se aparecer um valor que não esteja em FLAG_VALUES
    """
    flags = values.map(FLAG_VALUES)
    _raise_if_invalid(values, values.notna() & flags.isna(), "Y/N")
    return flags.astype("boolean")


def clean_text(values):
    """
    Tira os espaços no início e no fim; texto só com espaços passa a valor em falta

    Não muda maiúsculas nem o conteúdo: uniformizar categorias (ex.: "RN" e
    "Registered Nurse") é uma decisão de limpeza, não de parsing.

    Returns:
        pd.Series: Texto com o mesmo dtype e índice
    """
    stripped = values.str.strip()
    return stripped.mask(stripped == "")


def _raise_if_invalid(values, invalid, expected):
    """Dá erro com exemplos se a máscara invalid tiver algum True"""
    if invalid.any():
        examples = values[invalid].unique()[:5].tolist()
        raise ValueError(f"{values.name}: {invalid.sum()} valores fora do formato "
                         f"{expected}, por exemplo: {examples}")
