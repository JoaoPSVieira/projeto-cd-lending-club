"""
Schema do ficheiro accepted: o tipo de cada uma das 151 colunas e a sua conversão

Motivo:
    O tipo de cada coluna é declarado aqui, em vez de ser adivinhado pelo pandas.
    Assim a conversão é igual no desenvolvimento e no teste, e uma coluna com um formato
    inesperado dá erro em vez de mudar de tipo sem ninguém reparar.

Como foi decidido o tipo:
    - Colunas preenchidas no desenvolvimento: pelo formato dos valores e pelo significado no dicionário de dados.
    - Colunas vazias no desenvolvimento: pelo significado no dicionário. Se no teste tiverem outro formato, o parser dá erro.

Tipos ("kind"):
    id          identificador do empréstimo -> int64
    number      quantias, taxas, contagens, meses, pontuações -> float64
    term        " 36 months" -> 36.0 (meses)
    emp_length  "< 1 year".."10+ years" -> 0..10 (anos)
    month_year  "Dec-2015" -> 2015-12-01
    date        "2015-12-24" -> 2015-12-24 (só no rejected)
    percent     "38.64%" -> 38.64 (só no rejected; no accepted as % já vêm sem símbolo)
    flag        "Y"/"N" -> True/False
    category    poucas categorias fixas (ex.: grade, purpose) -> texto sem espaços nas pontas
    text        texto livre ou identificadores sem uso (ex.: emp_title, url) -> idem

Unidades:
    ACCEPTED_UNITS dá a unidade de cada coluna (USD, %, meses, pontos FICO, ...). O
    SchemaReport descreve cada coluna e escreve a tabela do schema em Markdown.
"""

from datetime import date
from pathlib import Path

import pandas as pd

from lending.parsing import (clean_text, parse_emp_length, parse_flag, parse_id,
                             parse_iso_date, parse_month_year, parse_number, parse_percent,
                             parse_term)

_NUMBER = """
    loan_amnt funded_amnt funded_amnt_inv int_rate installment annual_inc dti
    delinq_2yrs fico_range_low fico_range_high inq_last_6mths mths_since_last_delinq
    mths_since_last_record open_acc pub_rec revol_bal revol_util total_acc
    out_prncp out_prncp_inv total_pymnt total_pymnt_inv total_rec_prncp total_rec_int
    total_rec_late_fee recoveries collection_recovery_fee last_pymnt_amnt
    last_fico_range_high last_fico_range_low collections_12_mths_ex_med
    mths_since_last_major_derog policy_code annual_inc_joint dti_joint acc_now_delinq
    tot_coll_amt tot_cur_bal open_acc_6m open_act_il open_il_12m open_il_24m
    mths_since_rcnt_il total_bal_il il_util open_rv_12m open_rv_24m max_bal_bc all_util
    total_rev_hi_lim inq_fi total_cu_tl inq_last_12m acc_open_past_24mths avg_cur_bal
    bc_open_to_buy bc_util chargeoff_within_12_mths delinq_amnt mo_sin_old_il_acct
    mo_sin_old_rev_tl_op mo_sin_rcnt_rev_tl_op mo_sin_rcnt_tl mort_acc
    mths_since_recent_bc mths_since_recent_bc_dlq mths_since_recent_inq
    mths_since_recent_revol_delinq num_accts_ever_120_pd num_actv_bc_tl num_actv_rev_tl
    num_bc_sats num_bc_tl num_il_tl num_op_rev_tl num_rev_accts num_rev_tl_bal_gt_0
    num_sats num_tl_120dpd_2m num_tl_30dpd num_tl_90g_dpd_24m num_tl_op_past_12m
    pct_tl_nvr_dlq percent_bc_gt_75 pub_rec_bankruptcies tax_liens tot_hi_cred_lim
    total_bal_ex_mort total_bc_limit total_il_high_credit_limit revol_bal_joint
    sec_app_fico_range_low sec_app_fico_range_high sec_app_inq_last_6mths
    sec_app_mort_acc sec_app_open_acc sec_app_revol_util sec_app_open_act_il
    sec_app_num_rev_accts sec_app_chargeoff_within_12_mths
    sec_app_collections_12_mths_ex_med sec_app_mths_since_last_major_derog
    deferral_term hardship_amount hardship_length hardship_dpd
    orig_projected_additional_accrued_interest hardship_payoff_balance_amount
    hardship_last_payment_amount settlement_amount settlement_percentage settlement_term
"""

_MONTH_YEAR = """
    issue_d earliest_cr_line last_pymnt_d next_pymnt_d last_credit_pull_d
    sec_app_earliest_cr_line hardship_start_date hardship_end_date
    payment_plan_start_date debt_settlement_flag_date settlement_date
"""

_FLAG = "pymnt_plan hardship_flag debt_settlement_flag"

_CATEGORY = """
    grade sub_grade home_ownership verification_status loan_status purpose zip_code
    addr_state initial_list_status application_type verification_status_joint
    hardship_type hardship_reason hardship_status hardship_loan_status
    disbursement_method settlement_status
"""

_TEXT = "member_id emp_title url desc title"

# Tipo de cada coluna, agrupado por tipo (o parser segue a ordem das colunas do DataFrame)
ACCEPTED_SCHEMA = {
    "id": "id", "member_id": "text", "term": "term", "emp_length": "emp_length",
    **{c: "number" for c in _NUMBER.split()},
    **{c: "month_year" for c in _MONTH_YEAR.split()},
    **{c: "flag" for c in _FLAG.split()},
    **{c: "category" for c in _CATEGORY.split()},
    **{c: "text" for c in _TEXT.split()},
}

# -------------
# - Unidades -
# -------------
# Os tipos que não são "number" têm uma unidade única
KIND_UNITS = {
    "id": "identificador",
    "term": "meses",
    "emp_length": "anos",
    "month_year": "data (mês)",
    "date": "data (dia)",
    "percent": "%",
    "flag": "booleano",
    "category": "categoria",
    "text": "texto",
}

# Colunas "number", agrupadas por unidade (dicionário de dados + intervalo observado)
_NUMBER_UNITS = {
    "USD": """
        loan_amnt funded_amnt funded_amnt_inv installment revol_bal out_prncp out_prncp_inv
        total_pymnt total_pymnt_inv total_rec_prncp total_rec_int total_rec_late_fee
        recoveries collection_recovery_fee last_pymnt_amnt tot_coll_amt tot_cur_bal
        total_bal_il max_bal_bc total_rev_hi_lim avg_cur_bal bc_open_to_buy delinq_amnt
        tot_hi_cred_lim total_bal_ex_mort total_bc_limit total_il_high_credit_limit
        revol_bal_joint hardship_amount orig_projected_additional_accrued_interest
        hardship_payoff_balance_amount hardship_last_payment_amount settlement_amount
    """,
    "USD/ano": "annual_inc annual_inc_joint",
    # Percentagem de 0 a 100 (ex.: 12.99 = 12,99%), não fração de 0 a 1
    "%": """
        int_rate dti revol_util dti_joint il_util all_util bc_util pct_tl_nvr_dlq
        percent_bc_gt_75 sec_app_revol_util settlement_percentage
    """,
    "pontos FICO": """
        fico_range_low fico_range_high last_fico_range_high last_fico_range_low
        sec_app_fico_range_low sec_app_fico_range_high
    """,
    "meses": """
        mths_since_last_delinq mths_since_last_record mths_since_last_major_derog
        mths_since_rcnt_il mo_sin_old_il_acct mo_sin_old_rev_tl_op mo_sin_rcnt_rev_tl_op
        mo_sin_rcnt_tl mths_since_recent_bc mths_since_recent_bc_dlq mths_since_recent_inq
        mths_since_recent_revol_delinq sec_app_mths_since_last_major_derog deferral_term
        hardship_length settlement_term
    """,
    "dias": "hardship_dpd",
    "contagem": """
        delinq_2yrs inq_last_6mths open_acc pub_rec total_acc collections_12_mths_ex_med
        acc_now_delinq open_acc_6m open_act_il open_il_12m open_il_24m open_rv_12m
        open_rv_24m inq_fi total_cu_tl inq_last_12m acc_open_past_24mths
        chargeoff_within_12_mths mort_acc num_accts_ever_120_pd num_actv_bc_tl
        num_actv_rev_tl num_bc_sats num_bc_tl num_il_tl num_op_rev_tl num_rev_accts
        num_rev_tl_bal_gt_0 num_sats num_tl_120dpd_2m num_tl_30dpd num_tl_90g_dpd_24m
        num_tl_op_past_12m pub_rec_bankruptcies tax_liens sec_app_inq_last_6mths
        sec_app_mort_acc sec_app_open_acc sec_app_open_act_il sec_app_num_rev_accts
        sec_app_chargeoff_within_12_mths sec_app_collections_12_mths_ex_med
    """,
    "código": "policy_code",
}

# Unidade de cada coluna
ACCEPTED_UNITS = {
    **{col: KIND_UNITS[kind] for col, kind in ACCEPTED_SCHEMA.items() if kind != "number"},
    **{col: unit for unit, cols in _NUMBER_UNITS.items() for col in cols.split()},
}

# Função de conversão de cada tipo
PARSERS = {
    "id": parse_id,
    "number": parse_number,
    "term": parse_term,
    "emp_length": parse_emp_length,
    "month_year": parse_month_year,
    "date": parse_iso_date,
    "percent": parse_percent,
    "flag": parse_flag,
    "category": clean_text,
    "text": clean_text,
}


class SchemaParser:
    """
    Converte um DataFrame de texto para os tipos declarados no schema

    Não aprende nada com os dados: o mesmo objeto converte o desenvolvimento e o teste
    exatamente da mesma forma.

    Attributes:
        schema (dict[str, str]): Coluna -> tipo (ver PARSERS)

    Methods:
        parse(df): Devolve um DataFrame novo com as colunas convertidas
        summary(raw, parsed): Tabela de controlo, coluna a coluna
    """

    def __init__(self, schema=None):
        """
        Parameters:
            schema (dict[str, str] | None): Coluna -> tipo. Por omissão, ACCEPTED_SCHEMA.

        Raises:
            ValueError: Se algum tipo não existir em PARSERS
        """
        self.schema = ACCEPTED_SCHEMA if schema is None else schema
        unknown = set(self.schema.values()) - set(PARSERS)
        if unknown:
            raise ValueError(f"Tipos desconhecidos no schema: {sorted(unknown)}")

    def parse(self, df):
        """
        Converte todas as colunas de df

        Parameters:
            df (pd.DataFrame): Dados em texto, com exatamente as colunas do schema

        Returns:
            pd.DataFrame: Dados convertidos, com o mesmo índice e a mesma ordem de colunas

        Raises:
            ValueError: Se as colunas não forem as do schema, se algum valor não tiver o
                formato esperado, ou se a conversão perder valores que não eram vazios
        """
        self._check_columns(df)

        parsed = pd.DataFrame(
            {col: PARSERS[self.schema[col]](df[col]) for col in df.columns},
            index=df.index,
        )
        self._check_no_values_lost(df, parsed)
        return parsed

    def summary(self, raw, parsed):
        """
        Tabela de controlo da conversão, uma linha por coluna

        Parameters:
            raw (pd.DataFrame): Dados antes da conversão
            parsed (pd.DataFrame): Resultado de parse(raw)

        Returns:
            pd.DataFrame: kind, dtype, preenchidos antes e depois, e quantos textos só
                com espaços passaram a valor em falta
        """
        before = raw.notna().sum()
        after = parsed.notna().sum()
        return pd.DataFrame({
            "kind": pd.Series(self.schema),
            "dtype": parsed.dtypes.astype(str),
            "filled_before": before,
            "filled_after": after,
            "blank_to_missing": before - after,
        }).loc[raw.columns]


    # Métodos Privados
    def _check_columns(self, df):
        """Confirma que df tem exatamente as colunas do schema"""
        missing = [c for c in self.schema if c not in df.columns]
        extra = [c for c in df.columns if c not in self.schema]
        if missing or extra:
            raise ValueError(f"Colunas diferentes do schema. Em falta: {missing}; "
                             f"a mais: {extra}")

    def _check_no_values_lost(self, raw, parsed):
        """
        Confirma que só o texto vazio passou a valor em falta

        Nos tipos que não são texto, o número de valores preenchidos tem de ser igual
        antes e depois.
        """
        for col in raw.columns:
            before, after = raw[col].notna().sum(), parsed[col].notna().sum()
            if self.schema[col] in ("category", "text"):
                expected = before - (raw[col].str.strip() == "").sum()
            else:
                expected = before
            if after != expected:
                raise ValueError(f"{col}: {before} preenchidos antes, {after} depois "
                                 f"(esperado {expected}).")


class SchemaReport:
    """
    Descreve as colunas de um DataFrame já convertido: tipo, unidade e valores

    Serve para documentar o schema e confirmar as unidades com os valores observados
    (ex.: int_rate entre 5 e 26 confirma que está em %, e não em fração de 0 a 1).

    Attributes:
        schema (dict[str, str]): Coluna -> tipo
        units (dict[str, str]): Coluna -> unidade

    Methods:
        describe(df): Tabela com uma linha por coluna
        write_markdown(description, path, dataset, n_rows): Escreve a tabela em Markdown
    """

    # Número de categorias mais frequentes mostradas como exemplo
    N_EXAMPLES = 3

    def __init__(self, schema=None, units=None):
        """
        Parameters:
            schema (dict[str, str] | None): Por omissão, ACCEPTED_SCHEMA
            units (dict[str, str] | None): Por omissão, ACCEPTED_UNITS

        Raises:
            ValueError: Se alguma coluna do schema não tiver unidade, ou vice-versa
        """
        self.schema = ACCEPTED_SCHEMA if schema is None else schema
        self.units = ACCEPTED_UNITS if units is None else units

        without_unit = [c for c in self.schema if c not in self.units]
        not_in_schema = [c for c in self.units if c not in self.schema]
        if without_unit or not_in_schema:
            raise ValueError(f"Colunas sem unidade: {without_unit}; "
                             f"unidades de colunas fora do schema: {not_in_schema}")

    def describe(self, df):
        """
        Uma linha por coluna, pela ordem de df

        Parameters:
            df (pd.DataFrame): Dados já convertidos pelo SchemaParser

        Returns:
            pd.DataFrame: kind, unit, dtype, min, max (números e datas), distinct e
                examples (as categorias mais frequentes)
        """
        rows = []
        for col in df.columns:
            values = df[col].dropna()
            kind = self.schema[col]
            row = {"kind": kind, "unit": self.units[col], "dtype": str(df[col].dtype),
                   "min": "", "max": "", "distinct": values.nunique(), "examples": ""}

            if len(values) and kind in ("number", "term", "emp_length", "month_year", "date",
                                        "percent"):
                row["min"], row["max"] = _format_value(values.min()), _format_value(values.max())
            if kind in ("category", "flag"):
                top = values.value_counts().head(self.N_EXAMPLES).index
                row["examples"] = ", ".join(str(v) for v in top)
            rows.append(row)

        return pd.DataFrame(rows, index=pd.Index(df.columns, name="column"))

    def write_markdown(self, description, path, dataset, n_rows,
                       notebook="01_preprocessamento.ipynb"):
        """
        Escreve a tabela do schema num ficheiro Markdown

        Parameters:
            description (pd.DataFrame): Resultado de describe()
            path (str | Path): Ficheiro de destino
            dataset (str): Nome do conjunto descrito (ex.: "accepted, desenvolvimento")
            n_rows (int): Número de linhas do conjunto descrito
            notebook (str): Notebook que gera o ficheiro (indicado no topo)

        Returns:
            Path: Caminho do ficheiro escrito
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        units = description["unit"].value_counts()
        lines = [
            f"# Schema: {dataset}",
            "",
            f"_Gerado automaticamente por `notebooks/{notebook}` - não editar à mão._",
            "",
            f"- Conjunto descrito: {dataset}, {n_rows:,} linhas x {len(description)} colunas",
            f"- Gerado em: {date.today().isoformat()}",
            "- Tipos e unidades: `src/lending/schema.py`. Significado de cada coluna: "
            "`docs/01_formulacao_problema/elegibilidade_colunas.md`.",
            "- Mín./máx.: valores observados neste conjunto (vazio se a coluna não tiver valores).",
            "- Exemplos: as categorias mais frequentes.",
            "",
            "## Colunas por unidade",
            "",
            "| Unidade | Colunas |",
            "|---|---:|",
            *[f"| {unit} | {n} |" for unit, n in units.items()],
            "",
            "## Colunas",
            "",
            "| Coluna | Tipo | Unidade | dtype | Mín. | Máx. | Distintos | Exemplos |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
        for col, r in description.iterrows():
            examples = r["examples"].replace("|", "\\|")
            lines.append(f"| `{col}` | {r['kind']} | {r['unit']} | {r['dtype']} | "
                         f"{r['min']} | {r['max']} | {r['distinct']:,} | {examples} |")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _format_value(value):
    """Mín./máx. legível: datas em AAAA-MM, inteiros sem casas decimais, resto com 2"""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m")
    if float(value).is_integer():
        return f"{int(value):,}"
    if abs(value) < 0.01:
        return f"{value:.1e}"
    return f"{value:,.2f}"
