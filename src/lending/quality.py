"""
Qualidade dos dados: valores em falta

Porquê olhar para o "porquê" e não só para o "quanto":
    Um valor em falta pode querer dizer coisas muito diferentes, e cada uma pede um
    tratamento diferente:
    - "sem evento": mths_since_last_delinq vazio = o mutuário nunca teve um atraso.
      Imputar a mediana inventava um atraso que não existiu.
    - "sem denominador": revol_util vazio quando não há limite de crédito rotativo; o
      rácio não está definido.
    - "só alguns empréstimos": hardship_*, settlement_*, campos do 2.º proponente. Só
      existem para alguns casos (o enunciado pede para não pôr tudo a zero).
    - "não registado": o valor existia mas não foi guardado (ex.: emp_length).
    Além disso, há colunas que a LendingClub só começou a registar mais tarde: estão
    vazias nos primeiros anos por razões estruturais, não por causa do mutuário.

Decisões de desenho:
1 - O significado do valor em falta é declarado à mão (MISSING_MEANING), a partir do
    dicionário de dados e de verificações com outras colunas (ex.: mths_since_last_record
    vazio exatamente quando pub_rec = 0). As colunas não declaradas são "não registado".

2 - O ano a partir do qual cada coluna é registada é calculado a partir dos dados do
    desenvolvimento: o primeiro ano com pelo menos 1% dos valores preenchidos. Um limiar
    mais alto (ex.: 50%) confundiria "a coluna ainda não existia" com "o evento é raro"
    (mths_since_last_record está vazia em ~87% dos casos porque poucos mutuários têm
    registos públicos, e não porque não fosse registada).

3 - Códigos disfarçados: em 2007-2008, "nunca aconteceu" foi guardado como 0 em
    mths_since_last_record e mths_since_last_delinq. recode_disguised_missing corrige-o
    com uma regra determinística dentro de cada linha.
"""

import pandas as pd

# Significado do valor em falta (as colunas não indicadas são "não registado")
_MISSING_MEANING = {
    "sem evento": """
        mths_since_last_delinq mths_since_last_record mths_since_last_major_derog
        mths_since_recent_bc_dlq mths_since_recent_revol_delinq mths_since_recent_inq
        mths_since_recent_bc mths_since_rcnt_il mo_sin_old_il_acct
        sec_app_mths_since_last_major_derog
    """,
    "sem denominador": "revol_util bc_util bc_open_to_buy percent_bc_gt_75 il_util all_util",
    "só alguns empréstimos": """
        next_pymnt_d hardship_type hardship_reason hardship_status deferral_term
        hardship_amount hardship_start_date hardship_end_date payment_plan_start_date
        hardship_length hardship_dpd hardship_loan_status
        orig_projected_additional_accrued_interest hardship_payoff_balance_amount
        hardship_last_payment_amount debt_settlement_flag_date settlement_status
        settlement_date settlement_amount settlement_percentage settlement_term
    """,
    "só pedidos conjuntos": """
        annual_inc_joint dti_joint verification_status_joint revol_bal_joint
        sec_app_fico_range_low sec_app_fico_range_high sec_app_earliest_cr_line
        sec_app_inq_last_6mths sec_app_mort_acc sec_app_open_acc sec_app_revol_util
        sec_app_open_act_il sec_app_num_rev_accts sec_app_chargeoff_within_12_mths
        sec_app_collections_12_mths_ex_med
    """,
}
MISSING_MEANING = {col: meaning for meaning, cols in _MISSING_MEANING.items()
                   for col in cols.split()}
DEFAULT_MEANING = "não registado"

# Pares (meses desde o último evento, contagem de eventos) em que 0 meses com
# contagem 0 é impossível: é o código antigo de "nunca aconteceu"
DISGUISED_NEVER_PAIRS = [
    ("mths_since_last_record", "pub_rec"),
    ("mths_since_last_delinq", "delinq_2yrs"),
]


def recode_disguised_missing(df, pairs=None):
    """
    Passa a valor em falta ("sem evento") os 0 que são o código antigo de "nunca aconteceu"

    Regra, para cada par (meses, contagem): meses == 0 e contagem == 0 -> meses = NaN.
    "0 meses desde o último registo público" é impossível para quem tem 0 registos
    públicos; o mesmo para atrasos (um atraso há 0 meses contaria em delinq_2yrs).
    É uma regra fixa dentro de cada linha, por isso não aprende nada com os dados.

    Parameters:
        df (pd.DataFrame): Dados já convertidos
        pairs (list[tuple[str, str]] | None): Por omissão, DISGUISED_NEVER_PAIRS

    Returns:
        tuple[pd.DataFrame, pd.Series]: Cópia corrigida e número de valores corrigidos
            em cada coluna
    """
    pairs = DISGUISED_NEVER_PAIRS if pairs is None else pairs
    fixed = df.copy()
    counts = {}
    for months_col, count_col in pairs:
        disguised = (df[months_col] == 0) & (df[count_col] == 0)
        fixed.loc[disguised, months_col] = float("nan")
        counts[months_col] = int(disguised.sum())
    return fixed, pd.Series(counts, name="corrigidos")


class MissingnessReport:
    """
    Quanto falta em cada coluna, desde quando é registada e o que quer dizer faltar

    Attributes:
        meaning (dict[str, str]): Coluna -> significado do valor em falta
        threshold (float): Fração máxima de valores em falta para um ano contar como
            "registado" (por omissão 0.99, ou seja, pelo menos 1% preenchido)

    Methods:
        by_year(df, dates): Percentagem em falta de cada coluna em cada ano
        summary(df, dates): Uma linha por coluna com contagens, ano de início e significado
        identical_patterns(df): Grupos de colunas com valores em falta exatamente nas mesmas linhas
    """

    def __init__(self, meaning=None, threshold=0.99):
        """
        Parameters:
            meaning (dict[str, str] | None): Por omissão, MISSING_MEANING
            threshold (float): Ver Attributes
        """
        self.meaning = MISSING_MEANING if meaning is None else meaning
        self.threshold = threshold

    def by_year(self, df, dates):
        """
        Parameters:
            df (pd.DataFrame): Dados
            dates (pd.Series): Data de cada linha (ex.: issue_d)

        Returns:
            pd.DataFrame: Colunas de df nas linhas, anos nas colunas, % em falta (0-100)
        """
        return 100 * df.isna().groupby(dates.dt.year.rename("ano")).mean().T

    def summary(self, df, dates):
        """
        Uma linha por coluna, pela ordem de df

        Parameters:
            df (pd.DataFrame): Dados
            dates (pd.Series): Data de cada linha (ex.: issue_d)

        Returns:
            pd.DataFrame: missing, pct_missing, registered_from (primeiro ano com menos de
                threshold em falta; vazio se nunca), status ("completa", "vazia" ou
                "parcial") e meaning (só para colunas com valores em falta)
        """
        missing = df.isna().sum()
        per_year = self.by_year(df, dates) / 100
        registered = per_year.lt(self.threshold)

        status = pd.Series("parcial", index=df.columns)
        status[missing == 0] = "completa"
        status[missing == len(df)] = "vazia"

        meaning = pd.Series([self.meaning.get(c, DEFAULT_MEANING) for c in df.columns],
                            index=df.columns)
        meaning[status == "completa"] = ""

        return pd.DataFrame({
            "missing": missing,
            "pct_missing": (100 * missing / len(df)).round(2),
            "registered_from": registered.idxmax(axis=1).where(registered.any(axis=1)),
            "status": status,
            "meaning": meaning,
        }).rename_axis("column")

    def identical_patterns(self, df):
        """
        Grupos de colunas (com valores em falta, mas não vazias) em que os valores em falta
        estão exatamente nas mesmas linhas. Indica colunas que vêm da mesma fonte.

        Returns:
            list[list[str]]: Grupos com pelo menos 2 colunas, do maior para o menor número
                de valores em falta
        """
        missing = df.isna()
        partial = missing.columns[missing.any() & ~missing.all()]

        groups = {}
        for col in partial:
            key = missing[col].to_numpy().tobytes()
            groups.setdefault(key, []).append(col)

        found = [cols for cols in groups.values() if len(cols) > 1]
        return sorted(found, key=lambda cols: -missing[cols[0]].sum())
