# Conversão de tipos (parsing determinístico)

Segundo passo do pré-processamento do ficheiro *accepted*. Converte as 151 colunas, até
aqui todas em texto, para o seu tipo. Aplicado ao conjunto de desenvolvimento; o teste final
recebe exatamente as mesmas regras no fim, sem as rever.

Código: `src/lending/parsing.py` (funções de conversão), `src/lending/schema.py`
(`ACCEPTED_SCHEMA` e `SchemaParser`). Notebook: `notebooks/01_preprocessamento.ipynb`,
secção 4.

---

## 1. Porque é feito antes de qualquer *fit*

O enunciado distingue o *"deterministic parsing, such as removing a percent sign or
converting a date string using a fixed format"* das transformações aprendidas
(imputação, normalização, *encoding*). Todas as conversões deste passo são
determinísticas: cada valor é convertido só a partir de si próprio, com uma regra fixa
escrita no código. Nada é estimado a partir dos dados, por isso não há fuga de informação
e o mesmo `SchemaParser` serve para o desenvolvimento e para o teste.

---

## 2. Regras por tipo

O tipo de cada coluna está declarado em `ACCEPTED_SCHEMA`, em vez de ser adivinhado pelo
pandas. Foi decidido pelo formato dos valores, verificado em **todas** as linhas do
desenvolvimento, e pelo significado no dicionário de dados. As colunas vazias no
desenvolvimento (pedidos conjuntos, `sec_app_*`, etc.) têm o tipo do dicionário; se no
teste aparecerem noutro formato, o parser dá erro.

| Tipo | Formato no ficheiro | Regra | Resultado | Colunas |
|---|---|---|---|---:|
| `id` | `"9006963"` | só dígitos, sem valores em falta | `int64` | 1 |
| `number` | `"10000.0"`, `"12.99"`, `"-5e-10"` | número decimal, com sinal e notação científica opcionais | `float64` | 112 |
| `term` | `" 36 months"`, `" 60 months"` | número antes de "months", ignorando os espaços | `float64` (meses) | 1 |
| `emp_length` | `"< 1 year"` … `"10+ years"` | tabela fixa com os 11 valores | `float64` (0–10 anos) | 1 |
| `month_year` | `"Dec-2015"` | tabela fixa dos 12 meses em inglês, dia 1 | `datetime64[ns]` | 11 |
| `flag` | `"Y"`/`"N"`, `"y"`/`"n"` | tabela fixa | `boolean` | 3 |
| `category` | `"MORTGAGE"`, `"B3"` | tira os espaços das pontas | texto | 17 |
| `text` | texto livre | tira os espaços das pontas | texto | 5 |

**Colunas de cada tipo** (exceto `number`, cuja lista completa está em `schema.py`):

- `id`: `id`
- `term`: `term`
- `emp_length`: `emp_length`
- `month_year`: `issue_d`, `earliest_cr_line`, `last_pymnt_d`, `next_pymnt_d`,
  `last_credit_pull_d`, `sec_app_earliest_cr_line`, `hardship_start_date`,
  `hardship_end_date`, `payment_plan_start_date`, `debt_settlement_flag_date`,
  `settlement_date`
- `flag`: `pymnt_plan`, `hardship_flag`, `debt_settlement_flag`
- `category`: `grade`, `sub_grade`, `home_ownership`, `verification_status`, `loan_status`,
  `purpose`, `zip_code`, `addr_state`, `initial_list_status`, `application_type`,
  `verification_status_joint`, `hardship_type`, `hardship_reason`, `hardship_status`,
  `hardship_loan_status`, `disbursement_method`, `settlement_status`
- `text`: `member_id`, `emp_title`, `url`, `desc`, `title`

---

## 3. Decisões

| Decisão | Alternativa | Justificação |
|---|---|---|
| Schema declarado à mão | Deixar o pandas inferir os tipos | A inferência pode dar tipos diferentes no desenvolvimento e no teste (ex.: uma coluna vazia). Com o schema, uma mudança de formato dá erro |
| Formato inesperado dá erro | `errors="coerce"` (passa a NaN) | Um NaN silencioso misturava "valor em falta" com "valor mal lido" |
| Meses com tabela fixa | `format="%b-%Y"` | O `%b` depende da língua do sistema; a tabela dá o mesmo resultado em qualquer computador |
| `emp_length` em anos (0–10) | Manter como categoria ordenada | Há exatamente 11 valores e cada um passa a um número diferente, por isso não se perde informação. "< 1 year" passa a 0 e "10+ years" a 10, porque o ficheiro não distingue mais de 10 anos. A forma final para os modelos é decidida no pipeline |
| `term` em meses | Manter `"36 months"` | É um número com unidade; o texto só acrescenta espaços e a palavra "months" |
| Todos os números em `float64` | Inteiros para as contagens | `float64` aceita valores em falta (NaN) sem tipos especiais e é o que o scikit-learn espera |
| Texto só com espaços passa a valor em falta | Manter `"   "` | É um campo vazio escrito de outra forma |
| Não mudar maiúsculas nem juntar categorias | Uniformizar já (ex.: `"RN"` e `"Registered Nurse"`) | Isso é uma decisão de limpeza, que depende dos dados, e não de parsing |

---

## 4. Verificações

Feitas pelo `SchemaParser` em cada conversão (dão erro se falharem):

- As colunas do DataFrame são exatamente as 151 do schema.
- Nenhum valor preenchido se perde: nos tipos que não são texto, os preenchidos antes e
  depois são iguais; nos de texto, a diferença é exatamente o número de valores só com
  espaços.

Resultado no desenvolvimento (466 345 linhas):

| Verificação | Resultado |
|---|---|
| Valores fora do formato | 0 em todas as colunas |
| Texto só com espaços -> em falta | 235, todos em `desc` |
| `term` | 36 meses: 337 996; 60 meses: 128 349 |
| `emp_length` | 0–10 anos, 21 018 em falta (já vazios no ficheiro) |
| `issue_d` | 2007-06 a 2014-12 |
| `earliest_cr_line` | 1944-01 a 2011-11 |
| Memória | 850 MB -> 667 MB |

---

## 5. Observações para os passos seguintes

Encontradas ao verificar os formatos; não alteram a conversão, mas são tratadas na análise
de qualidade:

- **Colunas totalmente vazias no desenvolvimento:** `member_id`, os campos de pedido
  conjunto (`annual_inc_joint`, `dti_joint`, `verification_status_joint`,
  `revol_bal_joint`), todos os `sec_app_*` e vários agregados de *bureau* introduzidos
  mais tarde (ex.: `open_acc_6m`, `il_util`, `all_util`, `inq_fi`). No desenvolvimento,
  `application_type` é sempre `Individual`.
- **Colunas constantes:** `policy_code` (sempre 1), `disbursement_method` (sempre `Cash`).
- **`total_rec_late_fee`** tem 83 valores minúsculos em notação científica (ex.: `1e-10`),
  2 deles negativos: resíduos de arredondamento. É uma coluna pós-concessão, excluída de
  todas as tarefas.
- **`hardship_flag`** é `Y` em apenas 10 linhas, mas `hardship_type` está preenchido em
  583. Também é pós-concessão e excluída.
