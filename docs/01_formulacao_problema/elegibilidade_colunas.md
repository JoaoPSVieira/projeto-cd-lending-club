# Tabela de elegibilidade das colunas

Para cada uma das 151 colunas de `accepted_2007_to_2018Q4.csv`, indica-se se pode ser
usada como **preditor** em cada tarefa, e porquê. O critério é o descrito em
[`momentos_previsao.md`](momentos_previsao.md): uma coluna é elegível se o seu valor já é
conhecido no momento de previsão da tarefa (regressão em **T0**, classificação em **T4**;
clustering usa apenas o perfil em **T0**).

As decisões foram tomadas a partir do enunciado e do dicionário de dados
(`docs/LCDataDictionary.csv`), **sem consultar os valores dos dados**.

**Legenda**

| Símbolo | Significado |
|---|---|
| ✅ | Elegível |
| ❌ | Não elegível |
| ⚠️ | Elegível só com condição (indicada na justificação) |
| 🎯 | Alvo da tarefa |

> Elegível **não** significa que a coluna vai ser usada: indica apenas que usá-la não
> constitui fuga de informação. A seleção final é feita na engenharia de features e na
> modelação, com base nos dados de treino.

---

## Organização em grupos

| Grupo | Conteúdo | N.º de colunas |
|---|---|---|
| 1 | Empréstimo: pedido e condições | 13 |
| 2 | Mutuário | 8 |
| 3 | Histórico de crédito (*bureau*) | 69 |
| 4 | Pedido conjunto e 2.º proponente | 17 |
| 5 | Desempenho pós-concessão | 38 |
| 6 | Identificadores, administrativos e alvo | 6 |
| | **Total** | **151** |

---

## Grupo 1 - Empréstimo: pedido e condições

| Coluna | Descrição | Momento | Regressão | Classificação | Clustering | Justificação |
|---|---|---|---|---|---|---|
| `loan_amnt` | Montante pedido | T0 | ✅ | ✅ | ✅ | Indicado no pedido. Limitação: o dicionário indica que reflete eventuais reduções feitas pelo departamento de crédito. No clustering de estados é a base do "montante pedido" |
| `term` | Prazo (36 ou 60 meses) | T0 | ✅ | ✅ | ❌ | Escolhido no pedido. Não descreve o perfil de crédito do mutuário |
| `purpose` | Finalidade (categoria) | T0 | ✅ | ✅ | ❌ | Escolhida no pedido. Não descreve o perfil de crédito |
| `title` | Título do empréstimo (texto livre) | T0 | ✅ | ✅ | ❌ | Fornecido no pedido. Provavelmente redundante com `purpose`; a utilidade avalia-se na engenharia de features |
| `int_rate` | Taxa de juro | T2 | 🎯 | ✅ | ❌ | Alvo da regressão. Na classificação já está fixada na concessão (valor original) |
| `grade` | Grau de risco LC (A–G) | T2 | ❌ | ✅ | ❌ | Determina a taxa quase deterministicamente: na regressão trivializaria a tarefa. Fixado antes da concessão |
| `sub_grade` | Subgrau LC (A1–G5) | T2 | ❌ | ✅ | ❌ | Igual a `grade` |
| `installment` | Prestação mensal | T2 | ❌ | ✅ | ❌ | Calculada a partir de `loan_amnt`, `term` e `int_rate` (fórmula de amortização): permitiria reconstruir a taxa. Fixada na concessão e não depende do desfecho |
| `funded_amnt` | Montante comprometido para o empréstimo | T3 | ❌ | ✅ | ❌ | Posterior ao pricing. Na concessão (T4) o financiamento já está concluído, por isso o valor é conhecido |
| `funded_amnt_inv` | Parte financiada por investidores | T3 | ❌ | ❌ | ❌ | Posterior ao pricing. Descreve a repartição do financiamento entre investidores e não o risco do mutuário; o dicionário ("*at that point in time*") não garante que seja o valor na concessão - excluído por precaução |
| `initial_list_status` | Modo de listagem inicial (W/F) | T3 | ❌ | ✅ | ❌ | Decidido na listagem, depois do pricing. Na concessão já é conhecido e "*initial*" garante que é o valor original |
| `disbursement_method` | Modo de desembolso | T4 | ❌ | ❌ | ❌ | Momento de definição incerto (pode ser escolhido ou alterado até ao desembolso). O enunciado manda excluir em caso de dúvida |
| `issue_d` | Mês de financiamento | T4 | ❌ | ⚠️ | ⚠️ | Mês de financiamento, não data do pedido: futuro em relação a T0. Usado na **divisão temporal** de todas as tarefas. Na classificação só como feature temporal justificada (ex: ano de concessão) e como referência para a antiguidade do crédito. No clustering de mutuários só como referência para a antiguidade do crédito (exemplo do enunciado); nunca entra sozinho no ajuste |

---

## Grupo 2 - Mutuário

| Coluna | Descrição | Momento | Regressão | Classificação | Clustering | Justificação |
|---|---|---|---|---|---|---|
| `emp_title` | Cargo / empregador (texto livre) | T0 | ✅ | ✅ | ❌ | Declarado no pedido. Texto livre com muitas categorias. Mudança de significado a 23/09/2013 (nome do empregador -> cargo), a ter em conta na limpeza |
| `emp_length` | Anos de emprego (0–10) | T0 | ✅ | ✅ | ✅ | Declarado no pedido; descreve a estabilidade do mutuário |
| `home_ownership` | Situação habitacional | T0 | ✅ | ✅ | ✅ | Declarada no pedido |
| `annual_inc` | Rendimento anual declarado | T0 | ✅ | ✅ | ✅ | Declarado no pedido (*self-reported*) |
| `dti` | Rácio dívida / rendimento | T0 | ✅ | ✅ | ✅ | Calculado **excluindo** a hipoteca e o empréstimo LC pedido, logo não depende da taxa nem da prestação. Indicador de risco referido no enunciado |
| `verification_status` | Rendimento verificado ou não | T1 | ❌ | ✅ | ❌ | Não é possível garantir que a verificação termina antes do pricing; na concessão já está concluída |
| `zip_code` | 3 primeiros dígitos do código postal | T0 | ⚠️ | ⚠️ | ❌ | Disponível no pedido (não é fuga de informação), mas a geografia pode funcionar como *proxy* de características sensíveis. Uso condicionado a uma análise de *fairness* separada |
| `addr_state` | Estado de residência | T0 | ⚠️ | ⚠️ | ⚠️ | Igual a `zip_code`. No clustering é a **chave de agregação** dos estados, não uma variável do clustering de mutuários |

---

## Grupo 3 - Histórico de crédito (*bureau*)

Pressuposto comum a todo o grupo: os valores refletem o relatório de crédito obtido **no
momento do pedido** (T0) e descrevem as **outras contas** do mutuário, não o empréstimo
LendingClub em análise. Por isso são elegíveis nas três tarefas, mesmo quando o nome refere
atrasos ou cobranças (o enunciado indica que não devem ser excluídos só por causa do nome).
Nas variáveis `mths_since_*`, um valor em falta pode significar "o evento nunca aconteceu"
(não aplicável), o que é tratado no pré-processamento.

### 3a - Pontuação e antiguidade do crédito

| Coluna | Descrição | Momento | Regressão | Classificação | Clustering | Justificação |
|---|---|---|---|---|---|---|
| `fico_range_low` | Limite inferior do intervalo FICO | T0 | ✅ | ✅ | ✅ | Pontuação de crédito; indicador de risco do enunciado. Dicionário refere "*at loan origination*" - assume-se o valor do pedido |
| `fico_range_high` | Limite superior do intervalo FICO | T0 | ✅ | ✅ | ✅ | Igual a `fico_range_low` |
| `earliest_cr_line` | Mês da linha de crédito mais antiga | T0 | ⚠️ | ✅ | ✅ | Data anterior ao pedido. Na regressão **não** pode ser convertida em antiguidade com `issue_d` (data futura); usar referência anterior ao pricing ou `mo_sin_old_*`. Na classificação, antiguidade = `issue_d − earliest_cr_line` |
| `mo_sin_old_il_acct` | Meses desde a conta a prestações mais antiga | T0 | ✅ | ✅ | ✅ | Antiguidade calculada pelo *bureau* |
| `mo_sin_old_rev_tl_op` | Meses desde a conta rotativa mais antiga | T0 | ✅ | ✅ | ✅ | Antiguidade calculada pelo *bureau*; alternativa elegível a `earliest_cr_line` na regressão |
| `mo_sin_rcnt_rev_tl_op` | Meses desde a conta rotativa mais recente | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `mo_sin_rcnt_tl` | Meses desde a conta mais recente | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `mths_since_rcnt_il` | Meses desde a conta a prestações mais recente | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `mths_since_recent_bc` | Meses desde o cartão bancário mais recente | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |

### 3b - Atrasos, cobranças e registos negativos

| Coluna | Descrição | Momento | Regressão | Classificação | Clustering | Justificação |
|---|---|---|---|---|---|---|
| `delinq_2yrs` | N.º de atrasos ≥30 dias nos últimos 2 anos | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `mths_since_last_delinq` | Meses desde o último atraso | T0 | ✅ | ✅ | ✅ | Histórico do *bureau*; falta = possivelmente nunca houve atraso |
| `mths_since_last_record` | Meses desde o último registo público | T0 | ✅ | ✅ | ✅ | Histórico do *bureau*; falta = possivelmente sem registos |
| `mths_since_last_major_derog` | Meses desde a última classificação ≥90 dias | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `mths_since_recent_bc_dlq` | Meses desde o último atraso em cartão bancário | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `mths_since_recent_revol_delinq` | Meses desde o último atraso em crédito rotativo | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `acc_now_delinq` | N.º de contas atualmente em atraso | T0 | ✅ | ✅ | ✅ | "Atualmente" refere-se às outras contas no momento do pedido, não a este empréstimo |
| `delinq_amnt` | Montante em atraso nessas contas | T0 | ✅ | ✅ | ✅ | Igual a `acc_now_delinq` |
| `num_tl_30dpd` | Contas com 30 dias de atraso (últimos 2 meses) | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_tl_120dpd_2m` | Contas com 120 dias de atraso (últimos 2 meses) | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_tl_90g_dpd_24m` | Contas com ≥90 dias de atraso (últimos 24 meses) | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_accts_ever_120_pd` | Contas que alguma vez tiveram ≥120 dias de atraso | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `pct_tl_nvr_dlq` | % de contas sem atrasos | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `collections_12_mths_ex_med` | Cobranças nos últimos 12 meses (exceto médicas) | T0 | ✅ | ✅ | ✅ | Histórico anterior de crédito; o enunciado permite-o explicitamente |
| `chargeoff_within_12_mths` | *Charge-offs* nos últimos 12 meses | T0 | ✅ | ✅ | ✅ | Refere-se a outras contas; o enunciado permite-o explicitamente |
| `tot_coll_amt` | Montante total alguma vez em cobrança | T0 | ✅ | ✅ | ✅ | Histórico anterior de crédito; o enunciado permite-o explicitamente |
| `pub_rec` | N.º de registos públicos negativos | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `pub_rec_bankruptcies` | N.º de falências registadas | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `tax_liens` | N.º de penhoras fiscais | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |

### 3c - Contas e consultas

| Coluna | Descrição | Momento | Regressão | Classificação | Clustering | Justificação |
|---|---|---|---|---|---|---|
| `open_acc` | N.º de linhas de crédito abertas | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `total_acc` | N.º total de linhas de crédito | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `open_acc_6m` | Contas abertas nos últimos 6 meses | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `open_act_il` | Contas a prestações ativas | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* (no dicionário aparece como `open_il_6m`) |
| `open_il_12m` | Contas a prestações abertas (12 meses) | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `open_il_24m` | Contas a prestações abertas (24 meses) | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `open_rv_12m` | Contas rotativas abertas (12 meses) | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `open_rv_24m` | Contas rotativas abertas (24 meses) | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `acc_open_past_24mths` | Contas abertas nos últimos 24 meses | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_tl_op_past_12m` | Contas abertas nos últimos 12 meses | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `mort_acc` | N.º de contas de hipoteca | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_actv_bc_tl` | Cartões bancários ativos | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_actv_rev_tl` | Contas rotativas ativas | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_bc_sats` | Cartões bancários em situação regular | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_bc_tl` | N.º de cartões bancários | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_il_tl` | N.º de contas a prestações | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_op_rev_tl` | Contas rotativas abertas | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_rev_accts` | N.º de contas rotativas | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_rev_tl_bal_gt_0` | Contas rotativas com saldo > 0 | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `num_sats` | Contas em situação regular | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `total_cu_tl` | N.º de contas de financiamento | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `inq_last_6mths` | Consultas de crédito (6 meses) | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `inq_last_12m` | Consultas de crédito (12 meses) | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `inq_fi` | Consultas de financiamento pessoal | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `mths_since_recent_inq` | Meses desde a consulta mais recente | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |

### 3d - Saldos, limites e utilização

| Coluna | Descrição | Momento | Regressão | Classificação | Clustering | Justificação |
|---|---|---|---|---|---|---|
| `revol_bal` | Saldo total em crédito rotativo | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `revol_util` | % utilizada do crédito rotativo | T0 | ✅ | ✅ | ✅ | Indicador de utilização referido no enunciado |
| `bc_util` | % utilizada dos cartões bancários | T0 | ✅ | ✅ | ✅ | Indicador de utilização |
| `il_util` | % utilizada das contas a prestações | T0 | ✅ | ✅ | ✅ | Indicador de utilização |
| `all_util` | % utilizada de todas as contas | T0 | ✅ | ✅ | ✅ | Indicador de utilização |
| `percent_bc_gt_75` | % de cartões acima de 75% do limite | T0 | ✅ | ✅ | ✅ | Indicador de utilização |
| `bc_open_to_buy` | Crédito disponível nos cartões bancários | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `max_bal_bc` | Saldo máximo em contas rotativas | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `avg_cur_bal` | Saldo médio de todas as contas | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `tot_cur_bal` | Saldo atual total | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `total_bal_ex_mort` | Saldo total excluindo hipoteca | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `total_bal_il` | Saldo total das contas a prestações | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `tot_hi_cred_lim` | Limite de crédito total | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `total_rev_hi_lim` | Limite total do crédito rotativo | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `total_bc_limit` | Limite total dos cartões bancários | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |
| `total_il_high_credit_limit` | Limite total das contas a prestações | T0 | ✅ | ✅ | ✅ | Histórico do *bureau* |

---

## Grupo 4 - Pedido conjunto e 2.º proponente

Estes campos **não são fuga de informação**: são preenchidos no pedido. Contudo, só se
aplicam a pedidos conjuntos (`application_type` = Joint). Num pedido individual, a
ausência de valor significa **"não aplicável"**, e não "em falta", e **não deve ser
substituída por zero** (um rendimento conjunto de 0 seria um valor falso). O tratamento é
definido no pré-processamento. Não são usados no clustering de mutuários, que descreve o
perfil do proponente principal.

| Coluna | Descrição | Momento | Regressão | Classificação | Clustering | Justificação |
|---|---|---|---|---|---|---|
| `application_type` | Pedido individual ou conjunto | T0 | ✅ | ✅ | ❌ | Escolhido no pedido; indica se os restantes campos do grupo se aplicam |
| `annual_inc_joint` | Rendimento conjunto declarado | T0 | ✅ | ✅ | ❌ | Declarado no pedido; não aplicável em pedidos individuais |
| `dti_joint` | Rácio dívida / rendimento conjunto | T0 | ✅ | ✅ | ❌ | Igual a `dti`, para os dois proponentes |
| `revol_bal_joint` | Saldo rotativo conjunto | T0 | ✅ | ✅ | ❌ | Histórico do *bureau* dos dois proponentes |
| `verification_status_joint` | Rendimento conjunto verificado ou não | T1 | ❌ | ✅ | ❌ | Igual a `verification_status` (no dicionário aparece como `verified_status_joint`) |
| `sec_app_fico_range_low` | FICO (limite inferior) do 2.º proponente | T0 | ✅ | ✅ | ❌ | *Bureau* do 2.º proponente no pedido |
| `sec_app_fico_range_high` | FICO (limite superior) do 2.º proponente | T0 | ✅ | ✅ | ❌ | Igual ao anterior |
| `sec_app_earliest_cr_line` | Linha de crédito mais antiga do 2.º proponente | T0 | ⚠️ | ✅ | ❌ | Mesma regra de `earliest_cr_line` |
| `sec_app_inq_last_6mths` | Consultas (6 meses) do 2.º proponente | T0 | ✅ | ✅ | ❌ | *Bureau* do 2.º proponente |
| `sec_app_mort_acc` | Contas de hipoteca do 2.º proponente | T0 | ✅ | ✅ | ❌ | *Bureau* do 2.º proponente |
| `sec_app_open_acc` | Linhas abertas do 2.º proponente | T0 | ✅ | ✅ | ❌ | *Bureau* do 2.º proponente |
| `sec_app_revol_util` | Utilização rotativa do 2.º proponente | T0 | ✅ | ✅ | ❌ | *Bureau* do 2.º proponente |
| `sec_app_open_act_il` | Contas a prestações ativas do 2.º proponente | T0 | ✅ | ✅ | ❌ | *Bureau* do 2.º proponente |
| `sec_app_num_rev_accts` | Contas rotativas do 2.º proponente | T0 | ✅ | ✅ | ❌ | *Bureau* do 2.º proponente |
| `sec_app_chargeoff_within_12_mths` | *Charge-offs* (12 meses) do 2.º proponente | T0 | ✅ | ✅ | ❌ | Histórico anterior de crédito (outras contas) |
| `sec_app_collections_12_mths_ex_med` | Cobranças (12 meses) do 2.º proponente | T0 | ✅ | ✅ | ❌ | Histórico anterior de crédito |
| `sec_app_mths_since_last_major_derog` | Meses desde a última classificação ≥90 dias do 2.º proponente | T0 | ✅ | ✅ | ❌ | Histórico anterior de crédito |

---

## Grupo 5 - Desempenho pós-concessão

Todas estas colunas descrevem o que aconteceu **depois da concessão** (T5–T6):
pagamentos, recuperações, renegociações e acordos. Excluídas em todas as tarefas, como
indica o enunciado. Também são proibidas variáveis **derivadas** destas colunas (flags,
rácios, contagens de valores preenchidos), mesmo depois de as colunas originais serem
removidas - por exemplo `has_recovery` ou uma contagem de campos `hardship_*` não nulos.

| Coluna | Descrição | Momento | Regressão | Classificação | Clustering | Justificação |
|---|---|---|---|---|---|---|
| `pymnt_plan` | Existe plano de pagamento | T5 | ❌ | ❌ | ❌ | Posto em prática durante o empréstimo; o enunciado considera-o administrativo |
| `out_prncp` | Capital em dívida | T5 | ❌ | ❌ | ❌ | Pagamentos posteriores à concessão |
| `out_prncp_inv` | Capital em dívida (parte dos investidores) | T5 | ❌ | ❌ | ❌ | Pagamentos posteriores |
| `total_pymnt` | Total pago até à data | T5 | ❌ | ❌ | ❌ | Pagamentos posteriores |
| `total_pymnt_inv` | Total pago (parte dos investidores) | T5 | ❌ | ❌ | ❌ | Pagamentos posteriores |
| `total_rec_prncp` | Capital recebido | T5 | ❌ | ❌ | ❌ | Pagamentos posteriores |
| `total_rec_int` | Juros recebidos | T5 | ❌ | ❌ | ❌ | Pagamentos posteriores |
| `total_rec_late_fee` | Penalizações por atraso recebidas | T5 | ❌ | ❌ | ❌ | Revela atrasos no próprio empréstimo |
| `recoveries` | Montante recuperado após *charge-off* | T5 | ❌ | ❌ | ❌ | Só existe se houver *charge-off*: codifica o alvo |
| `collection_recovery_fee` | Comissão de cobrança após *charge-off* | T5 | ❌ | ❌ | ❌ | Igual a `recoveries` |
| `last_pymnt_d` | Mês do último pagamento | T5 | ❌ | ❌ | ❌ | Pagamentos posteriores |
| `last_pymnt_amnt` | Valor do último pagamento | T5 | ❌ | ❌ | ❌ | Pagamentos posteriores |
| `next_pymnt_d` | Próximo pagamento agendado | T5 | ❌ | ❌ | ❌ | Só existe para empréstimos em curso: revela o estado |
| `last_credit_pull_d` | Última consulta de crédito pela LC | T6 | ❌ | ❌ | ❌ | Consulta posterior à concessão |
| `last_fico_range_high` | FICO mais recente (limite superior) | T6 | ❌ | ❌ | ❌ | FICO posterior, influenciado pelo desfecho do empréstimo |
| `last_fico_range_low` | FICO mais recente (limite inferior) | T6 | ❌ | ❌ | ❌ | Igual ao anterior |
| `hardship_flag` | Mutuário em plano de dificuldades | T5 | ❌ | ❌ | ❌ | Dificuldades durante o empréstimo |
| `hardship_type` | Tipo de plano de dificuldades | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_reason` | Motivo das dificuldades | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_status` | Estado do plano | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `deferral_term` | Meses de diferimento | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_amount` | Montante do plano | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_start_date` | Início do plano | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_end_date` | Fim do plano | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `payment_plan_start_date` | Início do plano de pagamento | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_length` | Duração do plano | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_dpd` | Dias de atraso no início do plano | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_loan_status` | Estado do empréstimo no início do plano | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `orig_projected_additional_accrued_interest` | Juros adicionais projetados | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_payoff_balance_amount` | Saldo a liquidar no plano | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `hardship_last_payment_amount` | Último pagamento no plano | T5 | ❌ | ❌ | ❌ | Grupo `hardship_*` |
| `debt_settlement_flag` | Mutuário em acordo de liquidação | T5 | ❌ | ❌ | ❌ | Acordo após incumprimento: codifica o alvo |
| `debt_settlement_flag_date` | Data da flag de acordo | T5 | ❌ | ❌ | ❌ | Grupo `settlement_*` |
| `settlement_status` | Estado do acordo | T5 | ❌ | ❌ | ❌ | Grupo `settlement_*` |
| `settlement_date` | Data do acordo | T5 | ❌ | ❌ | ❌ | Grupo `settlement_*` |
| `settlement_amount` | Montante do acordo | T5 | ❌ | ❌ | ❌ | Grupo `settlement_*` |
| `settlement_percentage` | % do saldo acordada | T5 | ❌ | ❌ | ❌ | Grupo `settlement_*` |
| `settlement_term` | Prazo do acordo | T5 | ❌ | ❌ | ❌ | Grupo `settlement_*` |

---

## Grupo 6 - Identificadores, administrativos e alvo

| Coluna | Descrição | Momento | Regressão | Classificação | Clustering | Justificação |
|---|---|---|---|---|---|---|
| `id` | Identificador do empréstimo | - | ❌ | ❌ | ❌ | Identificador sem significado preditivo. Usado na deteção de duplicados |
| `member_id` | Identificador do mutuário | - | ❌ | ❌ | ❌ | Identificador. Seria útil para detetar mutuários repetidos; a sua disponibilidade é verificada na limpeza |
| `url` | Endereço da página do empréstimo | - | ❌ | ❌ | ❌ | Administrativo (enunciado) |
| `desc` | Descrição escrita pelo mutuário | T0 | ❌ | ❌ | ❌ | Não é fuga de informação, mas o enunciado classifica-o como administrativo e provavelmente irrelevante |
| `policy_code` | Código de política (1 = público, 2 = novos produtos) | - | ❌ | ❌ | ❌ | Administrativo (enunciado) |
| `loan_status` | Estado do empréstimo à data da extração | T6 | ❌ | 🎯 | ❌ | Usado **apenas** para construir o alvo da classificação e definir a coorte (exclusão dos `Current`). Nunca é preditor |

---

## Resumo

| Tarefa | ✅ Elegível | ⚠️ Condicional | ❌ Não elegível | 🎯 Alvo | Total |
|---|---|---|---|---|---|
| Regressão | 92 | 4 | 54 | 1 | 151 |
| Classificação | 102 | 3 | 45 | 1 | 151 |
| Clustering (mutuários) | 74 | 2 | 75 | - | 151 |

**Condicionais:**
- Regressão: `zip_code`, `addr_state` (*fairness*), `earliest_cr_line`, `sec_app_earliest_cr_line` (antiguidade sem `issue_d`).
- Classificação: `zip_code`, `addr_state` (*fairness*), `issue_d` (feature temporal justificada).
- Clustering: `addr_state` (chave de agregação dos estados), `issue_d` (só como referência da antiguidade do crédito; revisto na Parte 2).

## Regras transversais

1. **Derivadas herdam a elegibilidade das colunas de origem.** Uma feature calculada a
   partir de uma coluna ❌ é também ❌ (ex: `installment / annual_inc` na regressão).
2. **Não aplicável ≠ em falta.** Campos do 2.º proponente e `mths_since_*` vazios podem
   significar que o caso não se aplica; não se imputam automaticamente com zero.
3. **Pressuposto da fotografia histórica.** Os campos do *bureau* e do pedido são assumidos
   como os do momento do pedido. O dataset não permite confirmar quando cada valor foi
   registado; é uma limitação reconhecida.
4. **Geografia como questão de *fairness*.** `zip_code` e `addr_state` são tratados à parte,
   porque a localização pode funcionar como *proxy* de características protegidas.
5. **Ficheiro de rejeitados.** As suas 9 colunas são usadas apenas no clustering de
   estados e nos testes de hipóteses (aceites vs. rejeitados), não nas tarefas
   supervisionadas.
