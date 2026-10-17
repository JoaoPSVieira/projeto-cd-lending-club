"""
Regras de limpeza do ficheiro accepted

Linhas que não são empréstimos:
    O CSV do Kaggle é a junção de várias exportações da LendingClub. Cada exportação
    termina com duas linhas de totais ("Total amount funded in policy code 1: ...") e uma
    delas tem um separador ("Loans that do not meet the credit policy"). Nestas linhas o
    texto está na coluna id e todas as outras colunas estão vazias.

    A regra é estrutural (não depende de nenhuma estatística dos dados), por isso é
    aplicada ao ficheiro inteiro, antes da partição.

Empréstimos com condições modificadas depois da concessão:
    O enunciado pede para usar os valores originais de int_rate e installment, "not later
    modifications". Alguns empréstimos têm a taxa ou a prestação alterada depois da
    concessão (ex.: renegociação). Detetam-se por duas regras fixas, dentro de cada linha:
    - int_rate = 6,00% num grau B a G: abaixo da taxa mínima de qualquer desses graus
      (no desenvolvimento, sem estes casos, o mínimo de B é 8,67% e o de G 16,59%);
    - a prestação não corresponde à fórmula de amortização de funded_amnt, int_rate e term.
"""

import pandas as pd

# Taxa que aparece em empréstimos modificados, e graus em que não é uma taxa da tabela
MODIFIED_RATE = 6.0
GRADES_ABOVE_MODIFIED_RATE = ["B", "C", "D", "E", "F", "G"]

# Diferença máxima (USD) entre a prestação registada e a da fórmula de amortização
INSTALLMENT_TOLERANCE = 1.0


def amortization_installment(principal, annual_rate_pct, months):
    """
    Prestação mensal de um empréstimo com taxa fixa (fórmula de amortização)

    A = P * r * (1 + r)^n / ((1 + r)^n - 1), com r = taxa anual / 12

    Parameters:
        principal (pd.Series): Montante (P)
        annual_rate_pct (pd.Series): Taxa anual em % (ex.: 12.99)
        months (pd.Series): Número de prestações (n)

    Returns:
        pd.Series: Prestação mensal (A)
    """
    r = annual_rate_pct / 1200
    growth = (1 + r) ** months
    return principal * r * growth / (growth - 1)


def find_modified_terms(df, tolerance=INSTALLMENT_TOLERANCE):
    """
    Marca os empréstimos cuja taxa ou prestação não são as originais

    Parameters:
        df (pd.DataFrame): Dados já convertidos (int_rate, grade, installment,
            funded_amnt, term)
        tolerance (float): Diferença máxima aceite na prestação, em USD

    Returns:
        pd.DataFrame: Três colunas booleanas, com o índice de df:
            rate_below_grade (6,00% num grau B a G), installment_mismatch (prestação
            diferente da fórmula) e modified (qualquer uma das duas)
    """
    expected = amortization_installment(df["funded_amnt"], df["int_rate"], df["term"])
    rate_below_grade = ((df["int_rate"] == MODIFIED_RATE)
                        & df["grade"].isin(GRADES_ABOVE_MODIFIED_RATE))
    installment_mismatch = (df["installment"] - expected).abs() > tolerance

    return pd.DataFrame({
        "rate_below_grade": rate_below_grade,
        "installment_mismatch": installment_mismatch,
        "modified": rate_below_grade | installment_mismatch,
    })


def find_non_loan_rows(df, id_column="id"):
    """
    Marca as linhas que não descrevem um empréstimo

    Uma linha é marcada quando o id não é um número inteiro e todas as outras colunas
    estão vazias.

    Parameters:
        df (pd.DataFrame): Dados do accepted, ainda em texto
        id_column (str): Coluna do identificador do empréstimo

    Returns:
        pd.Series: Máscara booleana, True nas linhas que não são empréstimos

    Raises:
        ValueError: Se houver um id não numérico numa linha com outros dados preenchidos
            (não é um rodapé, por isso tem de ser analisado à parte)
    """
    numeric_id = df[id_column].str.fullmatch(r"\d+").fillna(False).astype(bool)
    others_empty = df.drop(columns=id_column).isna().all(axis=1)

    unexpected = ~numeric_id & ~others_empty
    if unexpected.any():
        raise ValueError(
            f"{unexpected.sum()} linhas com id não numérico mas com dados preenchidos, "
            f"por exemplo: {df.loc[unexpected, id_column].head(3).tolist()}"
        )

    return ~numeric_id
