# Descrição dos ficheiros de origem, schema e unidades

Terceiro passo do pré-processamento. Responde ao primeiro ponto do *Pre-processing* do
enunciado: *"Describe the source files, schema, units"*. Os valores em falta, os duplicados
e as restantes verificações de qualidade são tratados no passo seguinte.

Documentos relacionados:
- `docs/dados_origem.md`: tamanhos dos ficheiros e versões usadas na conversão.
- `docs/02_preprocessamento/schema_accepted.md`: tabela das 151 colunas (gerada pelo notebook).
- `docs/01_formulacao_problema/elegibilidade_colunas.md`: significado de cada coluna e uso
  em cada tarefa.

---

## 1. Ficheiros de origem

Fonte: Kaggle, *Lending Club Loan Data* (wordsforthewise/lending-club), descarregado a
2026-10-14.

| Ficheiro | Unidade de análise | Linhas | Colunas | Período |
|---|---|---:|---:|---|
| `accepted_2007_to_2018Q4.csv.gz` | Um empréstimo financiado | 2 260 701 (2 260 668 empréstimos) | 151 | `issue_d` de 2007-06 a 2018-12 |
| `rejected_2007_to_2018Q4.csv.gz` | Um pedido recusado | 27 648 741 | 9 | `Application Date` de 2007 a 2018 |

### 1.1 Estrutura do ficheiro *accepted*

- **Junção de 16 exportações** da LendingClub, identificadas pelas linhas de totais que
  fecham cada uma:

  | Ordem no ficheiro | `issue_d` | Empréstimos |
  |---:|---|---:|
  | 1 | 2015-01 a 2015-12 | 421 095 |
  | 2–7, 9, 10, 12, 14–16 | 12 exportações trimestrais de 2016 a 2018, fora de ordem | 1 373 228 |
  | 8 | 2014-01 a 2014-12 | 235 629 |
  | 11 | 2007-06 a 2011-12 | 42 535 |
  | 13 | 2012-01 a 2013-12 | 188 181 |

  Cada exportação termina com duas linhas de totais, e a de 2007–2011 tem ainda um separador
  antes dos empréstimos "Does not meet the credit policy". Ao todo, **33 linhas não são
  empréstimos** e foram removidas (`particao_temporal.md`).
- **A ordem das linhas não é cronológica:** o ficheiro começa pelos empréstimos de 2015 e
  mistura os trimestres. Por isso a partição usa cortes de calendário e nunca a posição das
  linhas.
- **Um empréstimo por linha:** no desenvolvimento, os 466 345 `id` são todos diferentes.
- **Fotografia do estado de cada empréstimo:** as colunas de pagamento e o `loan_status`
  descrevem o empréstimo na data em que o ficheiro foi extraído (ver secção 4).

### 1.2 Ficheiro *rejected*

Nove colunas: `Amount Requested`, `Application Date`, `Loan Title`, `Risk_Score`,
`Debt-To-Income Ratio`, `Zip Code`, `State`, `Employment Length`, `Policy Code`. É uma
"fotografia" muito mais pequena do pedido, porque estes requerentes nunca chegaram a ser
mutuários. A conversão de tipos, a partição por `Application Date` e as verificações
indicadas no enunciado (`Risk_Score` = 0 como valor em falta, DTI em texto com "%",
artefactos como 4042.11%) são feitas num passo próprio, descrito em `rejeitados.md`.

---

## 2. Schema do *accepted*

Cada coluna tem um tipo declarado em `ACCEPTED_SCHEMA` (`src/lending/schema.py`) e é
convertida pelo `SchemaParser` (`conversao_tipos.md`).

| Tipo | Resultado | Colunas |
|---|---|---:|
| `number` | `float64` | 112 |
| `category` | texto | 17 |
| `month_year` | `datetime64[ns]` (dia 1 do mês) | 11 |
| `text` | texto | 5 |
| `flag` | `boolean` | 3 |
| `id` | `int64` | 1 |
| `term` | `float64` (meses) | 1 |
| `emp_length` | `float64` (anos, 0–10) | 1 |
| **Total** | | **151** |

---

## 3. Unidades

Cada coluna tem uma unidade em `ACCEPTED_UNITS`. O `SchemaReport` confirma que todas as
colunas do schema têm unidade.

| Unidade | Colunas | Exemplos |
|---|---:|---|
| contagem | 42 | `open_acc`, `delinq_2yrs`, `inq_last_6mths`, `num_actv_rev_tl` |
| USD | 33 | `loan_amnt`, `installment`, `revol_bal`, `tot_cur_bal` |
| meses | 17 | `term`, `mths_since_last_delinq`, `mo_sin_old_rev_tl_op` |
| categoria | 17 | `grade`, `purpose`, `home_ownership`, `addr_state` |
| % (escala 0–100) | 11 | `int_rate`, `dti`, `revol_util`, `bc_util` |
| data (mês) | 11 | `issue_d`, `earliest_cr_line`, `last_pymnt_d` |
| pontos FICO | 6 | `fico_range_low`, `fico_range_high` |
| texto | 5 | `emp_title`, `title`, `desc` |
| booleano | 3 | `pymnt_plan`, `hardship_flag`, `debt_settlement_flag` |
| USD/ano | 2 | `annual_inc`, `annual_inc_joint` |
| identificador, anos, código, dias | 1 cada | `id`, `emp_length`, `policy_code`, `hardship_dpd` |

### 3.1 Convenções

- **Percentagens na escala 0–100:** `int_rate` = 12.99 significa 12,99%. Confirmado pelos
  intervalos observados (`int_rate` entre 5,42 e 26,06; `dti` entre 0 e 39,99).
- **Datas ao mês:** as datas do ficheiro só têm mês e ano. São guardadas como o dia 1 desse
  mês.
- **FICO em intervalo:** o *score* vem como um intervalo, entre 610 e 850. No
  desenvolvimento, `fico_range_high` = `fico_range_low` + 4 em 466 311 empréstimos e + 5
  em 34.
- **Montantes em dólares (USD) nominais**, sem correção da inflação.

### 3.2 Confirmação das unidades com os valores observados (desenvolvimento)

| Coluna | Unidade | Mín. | Máx. | Leitura |
|---|---|---:|---:|---|
| `loan_amnt` | USD | 500 | 35 000 | Coerente com o limite de 35 000 USD da LendingClub neste período |
| `int_rate` | % | 5,42 | 26,06 | Percentagem, não fração |
| `dti` | % | 0 | 39,99 | Percentagem; o máximo indica um limite de 40% |
| `annual_inc` | USD/ano | 1 896 | 7 500 000 | Rendimento declarado; extremos a analisar na qualidade |
| `fico_range_low` | pontos FICO | 610 | 845 | Escala FICO (300–850) |
| `term` | meses | 36 | 60 | Só dois prazos |
| `emp_length` | anos | 0 | 10 | 0 = menos de 1 ano; 10 = 10 ou mais |
| `earliest_cr_line` | data | 1944-01 | 2011-11 | Em todos os empréstimos é anterior ao respetivo `issue_d` |

As restantes 143 colunas estão em `schema_accepted.md`.

---

## 4. Data de extração do ficheiro

Na Fase 1 assumiu-se como cutoff de observação **31/12/2018**, a partir do nome do ficheiro
(`2018Q4`). As colunas de datas mostram que o ficheiro foi atualizado **depois** dessa data:

| Coluna | Última data | Empréstimos do desenvolvimento com data depois de 2018-12 |
|---|---|---:|
| `last_pymnt_d` | 2019-03 | 16 157 |
| `next_pymnt_d` | 2019-04 | 12 537 |
| `last_credit_pull_d` | 2019-03 | 171 975 |
| `debt_settlement_flag_date` | 2019-03 | 515 |
| `settlement_date` | 2019-03 | 160 |

O último pagamento registado é de março de 2019 e o próximo pagamento previsto é de abril
de 2019. A fotografia (o `loan_status` e as colunas de pagamento) corresponde, portanto, ao
**fim do 1.º trimestre de 2019**, e não ao fim de 2018. O "2018Q4" do nome refere-se ao
último trimestre de **concessão** (`issue_d` vai até 2018-12), não à data da extração.

**Impacto nas decisões da Fase 1:** nenhuma regra muda. O cutoff de observação comum
continua a ser 31/12/2018, agora justificado como uma escolha conservadora, anterior à
extração: todas as regras usam "o prazo terminou até ao cutoff", e como o ficheiro tem
informação até março de 2019, tudo o que terminou até dezembro de 2018 está observado. O
texto de `docs/01_formulacao_problema/desenho_split.md` e da secção 1 do relatório foi
corrigido em conformidade.

---

## 5. Observações para a análise de qualidade

Os intervalos observados levantam questões que são tratadas no passo seguinte:

- **Percentagens acima de 100:** `revol_util` até 892,3%, `bc_util` até 339,6%. Uma
  utilização acima de 100% é possível (saldo acima do limite), mas valores tão altos podem
  ser erros.
- **Valores que parecem códigos:** `tot_hi_cred_lim` e `total_rev_hi_lim` têm máximo
  exatamente 9 999 999, que parece um valor de "teto" e não um limite real.
  `last_fico_range_low/high` têm mínimo 0, fora da escala FICO, provavelmente um código de
  "sem informação" (pós-concessão, excluídas das tarefas).
- **Extremos:** `annual_inc` até 7,5 milhões, `tot_coll_amt` até 9,15 milhões.
- **Colunas vazias ou constantes:** 31 colunas totalmente vazias no desenvolvimento;
  `policy_code`, `application_type` e `disbursement_method` com um único valor.
- **Prazos de 60 meses:** 128 349 no desenvolvimento, todos a partir de 2010 (2010: 3 381;
  2011: 7 620; 2012: 9 897; 2013: 34 392; 2014: 73 059). Confirma-se a verificação pedida
  em `desenho_split.md`: não existem empréstimos de 60 meses antes de 2010.
