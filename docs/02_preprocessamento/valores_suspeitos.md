# Valores suspeitos e consistência entre colunas

Terceira parte da análise de qualidade do conjunto de desenvolvimento. Responde aos
*"quality checks"* do enunciado, que dá como exemplo um artefacto de introdução de dados
(DTI de 4042.11% nos rejeitados; os rejeitados são tratados num passo próprio).

Código: `src/lending/quality.py` (`QualityChecks`, `recode_cap_codes`),
`src/lending/cleaning.py` (`find_modified_terms`, `amortization_installment`). Notebook:
`notebooks/01_preprocessamento.ipynb`, secção 8.

---

## 1. Critério de decisão

| Situação | Decisão | Porquê |
|---|---|---|
| Valor que é claramente um código (ex.: 9 999 999) | Passa a valor em falta, com uma regra fixa | Não é um valor real; a regra é determinística e aplica-se igual ao teste |
| Linha cujas condições não são as originais | Excluída, com uma regra fixa, e registada no log | O enunciado pede os valores originais |
| Valor extremo mas possível | Fica | Cortar num percentil ou transformar é uma transformação aprendida: vai para o pipeline, com *fit* só no treino |
| Pequena incoerência entre colunas do *bureau* | Fica, documentada | Sem saber qual das colunas está certa, qualquer correção seria arbitrária |

---

## 2. Verificações

Número de empréstimos do desenvolvimento (466 345) que falham cada regra. As comparações
com valores em falta não contam.

### 2.1 Intervalo (uma coluna)

| Verificação | Regra | Linhas | Decisão |
|---|---|---:|---|
| `revol_util` acima de 100% | `revol_util` > 100 | 1 292 | Fica (secção 5) |
| `bc_util` acima de 100% | `bc_util` > 100 | 8 082 | Fica (secção 5) |
| Limite de crédito no teto | `tot_hi_cred_lim` ou `total_rev_hi_lim` = 9 999 999 | 3 | **Passa a valor em falta** (secção 4) |
| Rendimento acima de 1 milhão | `annual_inc` > 1 000 000 | 59 | Fica (secção 5) |
| Rendimento abaixo de 4 000 | `annual_inc` < 4 000 | 6 | Fica |
| DTI igual a 0 | `dti` = 0 | 363 | Fica (possível: sem outras dívidas) |
| FICO fora do intervalo de 4 pontos | `fico_range_high` − `fico_range_low` ≠ 4, exceto 845–850 | 0 | - |

### 2.2 Consistência (várias colunas)

| Verificação | Regra | Linhas | Decisão |
|---|---|---:|---|
| Financiado acima do pedido | `funded_amnt` > `loan_amnt` | 0 | - |
| Investidores acima do financiado | `funded_amnt_inv` > `funded_amnt` | 0 | - |
| Grau diferente do subgrau | `grade` ≠ 1.ª letra de `sub_grade` | 0 | - |
| Taxa de 6% num grau B a G | `int_rate` = 6,00 e `grade` em B…G | 139 | **Excluídos** (secção 3) |
| Prestação diferente da fórmula | \|`installment` − fórmula de amortização\| > 1 USD | 203 | **Excluídos** (secção 3) |
| Crédito mais antigo depois da concessão | `earliest_cr_line` ≥ `issue_d` | 0 | - |
| Contas abertas acima do total | `open_acc` > `total_acc` | 2 | Fica, documentada |
| Registos públicos incoerentes | `pub_rec_bankruptcies` + `tax_liens` > `pub_rec` | 0 | - |
| Atraso recente sem data recente | `delinq_2yrs` > 0 e `mths_since_last_delinq` > 24 | 272 | Fica, documentada |
| Data recente sem atraso recente | `delinq_2yrs` = 0 e `mths_since_last_delinq` < 24 | 40 | Fica, documentada |
| Saldo sem hipoteca acima do saldo total | `total_bal_ex_mort` > `tot_cur_bal` | 419 | Fica, documentada |
| Saldo rotativo sem utilização | `revol_bal` > 0 e `revol_util` = 0 | 372 | Fica, documentada |
| Utilização sem saldo rotativo | `revol_bal` = 0 e `revol_util` > 0 | 83 | Fica, documentada |
| Pagamento antes da concessão | `last_pymnt_d` < `issue_d` | 0 | - |
| Ativo depois do fim do prazo | estado ativo e `issue_d` + `term` < 2019-03 | 88 | Fica (secção 5) |

As regras com 0 falhas confirmam a coerência do ficheiro nos pontos mais importantes: o
financiamento nunca passa do pedido, o grau bate sempre com o subgrau e as datas estão na
ordem certa. Os 34 empréstimos com um intervalo FICO de 5 pontos são todos do escalão
845–850 (o FICO acaba em 850), por isso não são erros.

---

## 3. Empréstimos com condições modificadas (excluídos)

O enunciado pede para usar os valores originais de `int_rate` e `installment` (*"Use
original values, not later modifications"*). Duas regras mostram empréstimos cuja taxa ou
prestação foram alteradas depois da concessão:

**Regra 1: `int_rate` = 6,00% num grau B a G.** Sem estes casos, a taxa mínima de cada grau
no desenvolvimento é:

| Grau | A | B | C | D | E | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|
| Taxa mínima (sem os 6,00%) | 5,42 | 8,67 | 10,28 | 11,86 | 13,43 | 15,01 | 16,59 |
| Empréstimos com 6,00% | 6 | 32 | 38 | 37 | 26 | 4 | 2 |

Nos graus B a G, 6,00% fica muito abaixo de qualquer taxa da tabela (ex.: um D4 de
dezembro de 2014 com 6,00%, quando os D4 desse mês tinham 17,14%). No grau A, 6% é uma
taxa normal e não é marcada.

**Regra 2: prestação diferente da fórmula de amortização.** A prestação de um empréstimo
de taxa fixa é A = P·r·(1+r)ⁿ / ((1+r)ⁿ − 1), com P = `funded_amnt`, r = `int_rate`/1200
e n = `term`. Em 99,9% dos empréstimos a diferença para `installment` é inferior a 0,05 USD
(arredondamento). Em 203 é superior a 1 USD (até 496 USD): a prestação foi alterada.

**Resultado:** 221 empréstimos marcados (139 pela regra 1, 203 pela regra 2; 121 por ambas),
0,05% do desenvolvimento: 196 `Fully Paid`, 14 `Current`, 9 `Charged Off`, 1 em atraso e 1
"Does not meet". São **excluídos** de todas as tarefas: a taxa (alvo da regressão) e a
prestação (*input* da classificação) não são as originais, e não é possível recuperá-las.
A exclusão fica no registo de exclusões:

| N.º | Passo | Tipo | Antes | Removidas | Depois |
|---:|---|---|---:|---:|---:|
| 4 | Condições modificadas depois da concessão | remoção | 466 345 | 221 | 466 124 |

As duas regras são fixas e usam só a própria linha, por isso aplicam-se ao teste sem
alterações.

---

## 4. Limites de crédito no teto (passam a valor em falta)

`tot_hi_cred_lim` (3 linhas) e `total_rev_hi_lim` (2 das mesmas linhas) têm exatamente
9 999 999, o maior valor que cabe em 7 dígitos. Numa das linhas o limite total é 9 999 999,
mas o limite rotativo é 63 400 USD e o saldo total 268 743 USD: não é um limite real.
**Regra:** 9 999 999 exato -> valor em falta. 5 valores corrigidos.

---

## 5. Valores extremos que ficam

- **`revol_util` e `bc_util` acima de 100%:** uma utilização acima do limite é possível, e os
  valores são coerentes com as outras colunas. O caso mais extremo (892%) tem um saldo de
  2 677 USD com um limite de apenas 300 USD.
- **Rendimentos extremos:** 59 acima de 1 milhão, até 7,5 milhões. Alguns são pouco
  plausíveis (ex.: 7,4 milhões com o cargo "Registered Nurse"), mas o DTI foi calculado com
  esse rendimento e não há forma de saber o valor certo. A forte assimetria do rendimento é
  tratada no pipeline (ex.: logaritmo ou corte aprendido no treino).
- **Empréstimos ativos depois do fim do prazo (88):** empréstimos de 60 meses de 2013–2014
  ainda em pagamento na data de extração (prolongamentos). A classificação só usa
  empréstimos de 36 meses.

---

## 6. Alterações feitas neste passo

| Alteração | Linhas / valores |
|---|---:|
| Exclusão dos empréstimos com condições modificadas | 221 linhas |
| 9 999 999 -> valor em falta em `tot_hi_cred_lim` / `total_rev_hi_lim` | 3 / 2 valores |

O conjunto de desenvolvimento passa de 466 345 para **466 124** empréstimos. O conjunto limpo
e o registo de exclusões completo são guardados no fim do pré-processamento.
