# Valores em falta

Primeira parte da análise de qualidade do conjunto de desenvolvimento (466 345 empréstimos,
2007-06 a 2014-12). Responde a dois pedidos do enunciado: descrever os *"missing values"* e
*"distinguish non-applicability from missing information"*. O teste final não foi usado.

Código: `src/lending/quality.py` (`MISSING_MEANING`, `MissingnessReport`,
`recode_disguised_missing`). Notebook: `notebooks/01_preprocessamento.ipynb`, secção 6.

---

## 1. Porque não chega contar

Um valor em falta pode ter significados muito diferentes, e cada um pede um tratamento
diferente. Imputar a mediana de `mths_since_last_delinq` a quem nunca teve um atraso
inventava um atraso; pôr a zero os campos do 2.º proponente num pedido individual inventava
um proponente sem rendimento. Por isso cada coluna tem um **significado do valor em falta**,
declarado em `MISSING_MEANING` a partir do dicionário de dados e de verificações com outras
colunas:

| Significado | O que quer dizer | Colunas | Exemplo |
|---|---|---:|---|
| **sem evento** | O evento nunca aconteceu | 10 | `mths_since_last_record` vazio ⇔ `pub_rec` = 0 |
| **sem denominador** | O rácio não está definido | 6 | `revol_util` sem limite de crédito rotativo |
| **só alguns empréstimos** | Só existe para quem teve um plano de *hardship*, um acordo, ou um empréstimo ainda ativo | 21 | `hardship_*`, `settlement_*`, `next_pymnt_d` |
| **só pedidos conjuntos** | Só existe em pedidos conjuntos | 15 | `dti_joint`, `sec_app_*` |
| **não registado** | O valor existia mas não foi guardado (todas as outras) | 59 | `emp_length` |

Verificações que suportam esta classificação (no desenvolvimento):
- `mths_since_last_record` está vazio em todos os 404 942 empréstimos com `pub_rec` = 0 e
  preenchido em todos os 61 374 com `pub_rec` > 0 (depois da correção da secção 2).
- Os 6 campos `settlement_*` estão preenchidos exatamente nos 6 449 empréstimos com
  `debt_settlement_flag` = Y.
- `next_pymnt_d` está preenchido em todos os empréstimos ainda ativos (`Current`, atrasos,
  *grace period*) e vazio em todos os `Fully Paid` e `Charged Off`. A exceção são os 2 749
  "Does not meet the credit policy", que já terminaram mas têm a data preenchida: outra
  diferença de registo desse bloco antigo.
- Os 13 campos `hardship_*` estão preenchidos nas mesmas 583 linhas. `hardship_flag` = Y só em
  10 delas porque, segundo o dicionário, indica se o mutuário está **atualmente** num plano.

---

## 2. Códigos disfarçados: "nunca aconteceu" guardado como 0

Em 2007–2008, `mths_since_last_record` e `mths_since_last_delinq` quase não têm valores em
falta, mas têm muitos zeros (86,9% e 57,2% dos empréstimos de 2007). Esses zeros aparecem
só em mutuários **sem** o evento:

| Coluna | Contagem associada | Zeros com contagem = 0 (2007 / 2008 / 2009) | A partir de 2010 |
|---|---|---|---|
| `mths_since_last_record` | `pub_rec` | 524 / 748 / 3 | 0 |
| `mths_since_last_delinq` | `delinq_2yrs` | 344 / 474 / 2 | 0 |

"0 meses desde o último registo público" é impossível para quem tem 0 registos públicos, e
um atraso há 0 meses contaria em `delinq_2yrs` (atrasos nos últimos 2 anos). É uma mudança
de codificação da LendingClub: até 2008 "nunca aconteceu" era 0, depois passou a vazio. É o
mesmo tipo de problema que o enunciado aponta no `Risk_Score` = 0 dos rejeitados.

**Regra:** meses = 0 **e** contagem = 0 -> meses passa a valor em falta ("sem evento"). É
determinística e usa só a própria linha, por isso aplica-se igual ao teste.

**Resultado:** 1 275 valores corrigidos em `mths_since_last_record` e 820 em
`mths_since_last_delinq`. Os zeros que restam têm contagem > 0 e são verdadeiros (atraso
no próprio mês).

---

## 3. Resumo

| Estado | Colunas | Significado |
|---|---:|---|
| Completas (sem valores em falta) | 40 | - |
| Parciais | 80 | 47 não registado, 8 sem evento, 4 sem denominador, 21 só alguns empréstimos |
| Vazias no desenvolvimento | 31 | 15 só pedidos conjuntos, 12 não registado, 2 sem evento, 2 sem denominador |
| **Total** | **151** | |

Colunas parciais com mais interesse para as tarefas:

| Coluna | Em falta | % | Significado / observação |
|---|---:|---:|---|
| `emp_title` | 27 599 | 5,92 | Não registado; 20 832 em simultâneo com `emp_length` |
| `emp_length` | 21 018 | 4,51 | Não registado; nenhum em falta em 2007–2009 (possível mudança de codificação, não verificável) |
| `pub_rec_bankruptcies` | 1 365 | 0,29 | Não registado; sobretudo em 2007–2008 |
| `revol_util` | 340 | 0,07 | Sem denominador (274 têm `revol_bal` = 0) ou não registado (66) |
| `annual_inc` | 4 | 0,00 | Todos em empréstimos "Does not meet" de 2007 |
| 8 colunas básicas do *bureau* (`open_acc`, `delinq_2yrs`, …) | 29 | 0,01 | As mesmas 29 linhas, todas "Does not meet" de 2007 |
| `mths_since_last_delinq` | 251 200 | 53,87 | Sem evento |
| `mths_since_last_record` | 404 971 | 86,84 | Sem evento |

A tabela completa (151 colunas) está no notebook, secção 6.2.

---

## 4. Colunas registadas só a partir de 2012

**37 colunas** do *bureau* estão vazias em **todos** os empréstimos de 2007 a 2011, aparecem
durante 2012 e ficam praticamente completas a partir de 2013. A razão é estrutural: a
LendingClub ainda não as guardava. "Registada a partir de" = primeiro ano com pelo menos 1%
dos valores preenchidos.

| Grupo (falta nas mesmas linhas) | Colunas | Em falta em 2012 |
|---|---:|---:|
| `tot_cur_bal`, `tot_hi_cred_lim`, `total_rev_hi_lim`, `num_*_tl`, `num_rev_accts`, … | 17 | 52% |
| `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op` | 2 | 52% |
| `num_sats`, `num_bc_sats` | 2 | 30% |
| `acc_open_past_24mths`, `mort_acc`, `total_bal_ex_mort`, `total_bc_limit` | 4 | 14% |
| Outras (`bc_util`, `bc_open_to_buy`, `avg_cur_bal`, `pct_tl_nvr_dlq`, `mths_since_recent_*`, …) | 12 | 15%–90% |

### 4.1 Consequência para a classificação

O treino da classificação é 2007-06 a 2011-12 (`desenho_split.md`). Estas 37 colunas estão
**100% vazias nesse treino**: o classificador não pode aprender nada com elas, mesmo que no
teste (2015) estejam preenchidas. Entre elas estão **8 dos 27 *inputs* sugeridos pelo
enunciado**:

`mort_acc`, `tot_cur_bal`, `tot_hi_cred_lim`, `bc_util`, `bc_open_to_buy`,
`mo_sin_old_rev_tl_op`, `num_actv_rev_tl`, `pct_tl_nvr_dlq`.

Os outros 19 *inputs* sugeridos têm no máximo 6,2% em falta no treino da classificação
(`emp_title`).

**Decisão:** estas 37 colunas **não são usadas na classificação**. O pipeline retira-as
automaticamente, porque estão vazias no treino (`DropEmptyColumns`, `pipeline.md`).

### 4.2 Consequência para a regressão

No desenho inicial, o treino da regressão era todo o desenvolvimento (2007–2014). Com esse
período, as 37 colunas faltariam em cerca de 15% das linhas (os anos de 2007–2011 e parte de
2012). Isto **não é um problema de mutuários individuais, mas do período**: um indicador de
"em falta" nestas colunas seria, na prática, um indicador do ano de concessão, e a imputação
inventaria valores para cinco anos inteiros.

**Decisão: a regressão treina só com 2013–2014** (`config.REGRESSION_PARTITIONS`), os anos em
que as 37 colunas estão completas (no máximo 1% em falta). Os 95 846 empréstimos de 2007-06 a
2012-12 ficam fora do treino da regressão, num passo de tipo "partição" da tabela de
exclusões; continuam no conjunto limpo e são usados na EDA geral e no clustering. A validação
cruzada temporal passa a ser por trimestre (6 *folds*), porque por ano só haveria um. Decisão
e alternativas em `coortes.md`, secção 4.2, e em `desenho_split.md`, secção 3.

---

## 5. Colunas vazias no desenvolvimento

31 colunas não têm nenhum valor em 2007–2014: `member_id`, os 4 campos de pedido conjunto e
os 12 `sec_app_*` (não há nenhum pedido conjunto no desenvolvimento; `application_type` é
sempre `Individual`), e 14 agregados do *bureau* introduzidos depois de 2014 (`open_acc_6m`,
`il_util`, `all_util`, `inq_fi`, …). Sem nenhum valor no treino, não podem ser usadas por
nenhum modelo.

---

## 6. Tratamento previsto

Este passo só **analisa** (e corrige os códigos disfarçados). A imputação e os indicadores
são transformações aprendidas, por isso são feitos no pipeline, com *fit* só no treino:

| Significado | Tratamento previsto |
|---|---|
| sem evento | Indicador "sem evento" + valor de preenchimento fixo; nunca a mediana |
| sem denominador | Indicador "não aplicável" |
| só alguns empréstimos | Nenhum: são colunas pós-concessão, excluídas de todas as tarefas |
| só pedidos conjuntos | Vazias no desenvolvimento; não podem ser usadas |
| não registado | Imputação (ex.: mediana ou moda) aprendida no treino, com indicador de "em falta" |
| registadas só a partir de 2012 | Decidido na seleção de *features* (secção 4) |
