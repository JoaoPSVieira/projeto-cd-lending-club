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

Duplicados e mutuários repetidos (DuplicateReport):
    - Empréstimo duplicado: o mesmo empréstimo em mais do que uma linha. Inflaciona os
      dados e, se as cópias caírem em partições diferentes, o modelo é avaliado com
      casos que já viu.
    - Mutuário repetido: a mesma pessoa com vários empréstimos. Não é um erro, mas as
      linhas deixam de ser independentes. O member_id está vazio em todo o ficheiro, por
      isso só é possível uma estimativa com uma chave aproximada.
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


# Chave aproximada do mutuário: atributos pessoais que mudam pouco entre dois pedidos.
# earliest_cr_line é o mês da primeira linha de crédito (fixo para cada pessoa); zip_code e
# addr_state dão a localização; annual_inc, emp_title e home_ownership o perfil declarado.
REPEAT_BORROWER_KEY = ["earliest_cr_line", "zip_code", "addr_state", "annual_inc",
                       "emp_title", "home_ownership"]


class DuplicateReport:
    """
    Empréstimos duplicados e estimativa de mutuários repetidos

    Attributes:
        id_column (str): Identificador do empréstimo
        url_column (str): Endereço do empréstimo (termina em "loan_id=<id>")
        borrower_id_column (str): Identificador do mutuário (member_id)
        borrower_key (list[str]): Colunas da chave aproximada do mutuário
        date_column (str): Data de concessão, para medir o intervalo entre empréstimos

    Methods:
        checks(df): Tabela com as verificações de duplicados
        repeat_borrowers(df): Linhas cuja chave aproximada aparece mais do que uma vez
        repeat_summary(df): Números da estimativa de mutuários repetidos
    """

    def __init__(self, id_column="id", url_column="url", borrower_id_column="member_id",
                 borrower_key=None, date_column="issue_d"):
        self.id_column = id_column
        self.url_column = url_column
        self.borrower_id_column = borrower_id_column
        self.borrower_key = REPEAT_BORROWER_KEY if borrower_key is None else borrower_key
        self.date_column = date_column

    def checks(self, df):
        """
        Verificações de duplicados, uma por linha

        Returns:
            pd.DataFrame: check, rule e rows (número de linhas que falham a verificação,
                exceto na última, que conta os identificadores de mutuário preenchidos)
        """
        url_id = df[self.url_column].str.extract(r"loan_id=(\d+)$", expand=False)
        url_mismatch = url_id.isna() | (url_id.astype("float") != df[self.id_column])
        others = df.drop(columns=[self.id_column, self.url_column])

        rows = [
            ("id repetido", f"{self.id_column} aparece em mais do que uma linha",
             df[self.id_column].duplicated(keep=False).sum()),
            ("url de outro empréstimo", f"o loan_id do {self.url_column} difere do {self.id_column}",
             url_mismatch.sum()),
            ("linha repetida", f"todas as colunas iguais, exceto {self.id_column} e {self.url_column}",
             others.duplicated(keep=False).sum()),
            ("identificador do mutuário", f"{self.borrower_id_column} preenchido",
             df[self.borrower_id_column].notna().sum()),
        ]
        return pd.DataFrame(rows, columns=["check", "rule", "rows"]).set_index("check")

    def repeat_borrowers(self, df):
        """
        Empréstimos cuja chave aproximada do mutuário aparece mais do que uma vez

        Só entram linhas com todas as colunas da chave preenchidas.

        Returns:
            pd.DataFrame: id, data, colunas da chave, group (número do grupo) e group_size,
                ordenado por grupo e data
        """
        keyed = df.dropna(subset=self.borrower_key)
        size = keyed.groupby(self.borrower_key)[self.id_column].transform("size")
        repeated = keyed.loc[size > 1, [self.id_column, self.date_column, *self.borrower_key]]
        return (repeated
                .assign(group=repeated.groupby(self.borrower_key).ngroup(),
                        group_size=size[size > 1])
                .sort_values(["group", self.date_column]))

    def repeat_summary(self, df):
        """
        Números da estimativa de mutuários repetidos

        Returns:
            pd.Series: empréstimos com chave completa, empréstimos em grupos repetidos (e %),
                grupos, intervalo mediano em meses entre o 1.º e o último empréstimo do grupo,
                e grupos com todos os empréstimos no mesmo mês
        """
        keyed = df[self.borrower_key].notna().all(axis=1).sum()
        repeated = self.repeat_borrowers(df)
        dates = repeated.groupby("group")[self.date_column]
        gap_months = (dates.max() - dates.min()).dt.days / 30.44

        return pd.Series({
            "empréstimos com chave completa": int(keyed),
            "empréstimos em grupos repetidos": len(repeated),
            "% dos empréstimos com chave": round(100 * len(repeated) / keyed, 2),
            "grupos (possíveis mutuários)": repeated["group"].nunique(),
            "intervalo mediano no grupo (meses)": round(gap_months.median(), 1),
            "grupos com todos os empréstimos no mesmo mês": int((gap_months == 0).sum()),
        }, dtype="object")  # object: as contagens não passam a decimais
