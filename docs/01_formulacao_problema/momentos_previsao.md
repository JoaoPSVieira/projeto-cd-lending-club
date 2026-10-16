# Momentos de previsão e informação disponível

Uma variável só pode ser usada como preditor se o seu valor já for conhecido no
**momento de previsão** da tarefa. Este documento descreve a linha temporal de um
empréstimo na LendingClub e indica que informação existe em cada momento. É a base da
tabela de elegibilidade das colunas.

---

## Linha temporal de um empréstimo

```text
 T0 Pedido        T1 Verificação   T2 Pricing        T3 Listagem /    T4 Concessão     T5 Pagamentos    T6 Extração
 submetido        do pedido        (taxa fixada)     financiamento    (issue_d)        mensais          dos dados (2018Q4)
    │                 │                 │                 │                 │                 │                 │
────●─────────────────●─────────────────●─────────────────●─────────────────●─────────────────●─────────────────●──►
    ▲                                                                       ▲
    │                                                                       │
 REGRESSÃO                                                            CLASSIFICAÇÃO
 (antes do pricing)                                                   (depois do pricing,
                                                                       antes do 1º pagamento)
```

| Momento | O que acontece | Informação que passa a existir (exemplos) |
|---|---|---|
| **T0 - Pedido submetido** | O requerente preenche o pedido; é consultado o relatório de crédito (*bureau*) | `loan_amnt`, `term`, `purpose`, `title`, `emp_title`, `emp_length`, `home_ownership`, `annual_inc`, `dti`, `addr_state`, `zip_code`, `application_type`, campos do 2.º proponente (`*_joint`, `sec_app_*`), histórico de crédito (`fico_range_*`, `earliest_cr_line`, `delinq_2yrs`, `inq_last_6mths`, `open_acc`, `revol_util`, `tot_cur_bal`, `mo_sin_*`, `num_*`, ...) |
| **T1 - Verificação** | A LendingClub pode verificar o rendimento declarado | `verification_status`, `verification_status_joint` |
| **T2 - Pricing** | É atribuído o grau de risco e a taxa | `grade`, `sub_grade`, `int_rate`, `installment` |
| **T3 - Listagem e financiamento** | O empréstimo é listado e financiado por investidores | `initial_list_status`, `funded_amnt`, `funded_amnt_inv` |
| **T4 - Concessão** | O dinheiro é entregue; o empréstimo começa | `issue_d` (mês de financiamento), `disbursement_method` |
| **T5 - Pagamentos** | O mutuário paga (ou não) as prestações | `total_pymnt*`, `total_rec_*`, `out_prncp*`, `last_pymnt_*`, `next_pymnt_d`, `recoveries`, `collection_recovery_fee`, `hardship_*`, `settlement_*`, `debt_settlement_flag*` |
| **T6 - Extração** | Fotografia do estado do empréstimo na data de extração (≈ fim de março de 2019) | `loan_status`, `last_credit_pull_d`, `last_fico_range_*` |

> **Pressuposto:** os campos de crédito (*bureau*) refletem o relatório obtido no pedido
> (T0). O ficheiro é uma fotografia histórica, por isso não é possível confirmar o momento
> exato em que cada campo foi registado. Esta limitação é reconhecida no relatório.

---

## Regressão - momento T0 (antes do pricing)

**Informação disponível:** apenas a de **T0**, ou seja, as características do pedido e o
perfil de crédito do requerente.

| Decisão | Justificação |
|---|---|
| `grade`, `sub_grade`, `int_rate`, `installment` **excluídos** | Surgem em T2. `int_rate` é o alvo; `grade`/`sub_grade` determinam a taxa quase diretamente; `installment` é calculada a partir da taxa |
| `funded_amnt`, `funded_amnt_inv`, `initial_list_status` **excluídos** | Surgem depois do pricing (T3) |
| `issue_d` e `disbursement_method` **excluídos como preditores** | Surgem em T4. `issue_d` é o mês de financiamento, não a data do pedido; serve apenas para a divisão temporal |
| `verification_status*` **excluídos** | Não é possível garantir que a verificação estava concluída antes do pricing |
| Tudo o que surge em T5–T6 **excluído** | Informação posterior à concessão |
| `loan_amnt`, `term`, `purpose`, `title` **permitidos** | Assume-se que os valores registados são os do pedido; limitação: podem ter sido ajustados depois |
| Antiguidade do crédito | **Não** calcular como `issue_d − earliest_cr_line`, porque `issue_d` é uma data futura em relação ao pedido. Usar variáveis de antiguidade já calculadas pelo *bureau* (ex.: `mo_sin_old_rev_tl_op`) ou outra referência anterior ao pricing |

---

## Classificação - momento T4 (concessão)

**Informação disponível:** tudo o que existe **até T4**, ou seja, o pedido, a verificação,
o pricing, o financiamento e a data de concessão. Nenhum pagamento foi ainda observado.

| Decisão | Justificação |
|---|---|
| `grade`, `sub_grade`, `int_rate`, `installment` **permitidos** | Já estão fixados na concessão; usam-se os valores originais |
| `verification_status*` **permitidos** | A verificação está concluída antes de o empréstimo ser concedido |
| `funded_amnt`, `funded_amnt_inv`, `initial_list_status` | Candidatos (conhecidos em T3, antes da concessão); decisão final na tabela de elegibilidade |
| `issue_d` | Permitido como preditor quando justificado (ex.: ano ou mês de concessão); usado também na divisão temporal |
| Antiguidade do crédito | Calculada relativamente a `issue_d` (`issue_d − earliest_cr_line`), não à data de hoje nem a um *credit pull* posterior |
| Tudo o que surge em T5–T6 **excluído** | Descreve o que aconteceu depois da concessão; `loan_status` é usado **apenas** para construir o alvo |
| Variáveis derivadas de campos excluídos **excluídas** | Ex.: `has_recovery`, contagem de campos `hardship_*` preenchidos |

---

## Clustering - sem momento de previsão

Não há previsão, mas para os grupos serem interpretáveis como **perfis no momento do
pedido** usam-se apenas variáveis de T0 (perfil de crédito e indicadores de risco). Para os
estados, os agregados (taxa de aprovação, montantes, *risk scores*) são calculados
combinando aceites e rejeitados, usando apenas o período de desenvolvimento, para não
tocar no período de teste final.

---

## Questões em aberto (resolvidas nos documentos seguintes)

- Decisão final, coluna a coluna, para `funded_amnt`, `funded_amnt_inv`,
  `initial_list_status` e `disbursement_method` -> tabela de elegibilidade.
- Campos do 2.º proponente: distinguir "não aplicável" (pedido individual) de "em falta" ->
  tabela de elegibilidade.
- Data a partir da qual os desfechos são observáveis e escolha do período de teste ->
  desenho do split.
