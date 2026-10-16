"""
Regras de limpeza do ficheiro accepted

Linhas que não são empréstimos:
    O CSV do Kaggle é a junção de várias exportações da LendingClub. Cada exportação
    termina com duas linhas de totais ("Total amount funded in policy code 1: ...") e uma
    delas tem um separador ("Loans that do not meet the credit policy"). Nestas linhas o
    texto está na coluna id e todas as outras colunas estão vazias.

    A regra é estrutural (não depende de nenhuma estatística dos dados), por isso é
    aplicada ao ficheiro inteiro, antes da partição.
"""


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
