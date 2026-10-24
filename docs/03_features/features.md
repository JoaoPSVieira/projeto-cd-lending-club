# Features novas (Fase 3)

O enunciado pede pelo menos 10 features novas e para cada uma as colunas de origem, a fórmula, o motivo, o tratamento
dos valores em falta e a elegibilidade em cada momento de previsão. No total são 10 features

Código: `src/lending/feature_engineering.py` (`FEATURES`, `FeatureBuilder`).
Verificação: `notebooks/03_features.ipynb`.

---

## Regras comuns

1. **Herança da elegibilidade.** Uma feature só existe a partir do momento em que existem todas as colunas de onde vem.
   Tudo o que usa `installment` (T2, *pricing*) ou `issue_d` (T4, concessão) fica fora da regressão (T0). O código
   verifica-o, `check_leakage` usa as listas de colunas proibidas de `features.py` e dá erro se uma feature de uma
   tarefa usar uma coluna proibida nessa tarefa.
2. **Dentro do pipeline.** O `FeatureBuilder` é um transformer do scikit-learn e vem antes do `build_preprocessor`. Os
   cálculos são linha a linha mas assim são aplicados sempre da mesma forma no treino, na validação e no teste.
3. **Valores em falta.** Uma divisão por zero (ou por um valor negativo), uma data impossível ou uma coluna de origem em
   falta dão **NaN**, nunca infinito nem 0. Depois, o pipeline trata esses NaN como os de qualquer coluna. Imputação
   pela mediana do treino de cada *fold* + indicador de "em falta". O *clipping* aos percentis 0,5–99,5 do treino limita
   os rácios extremos.

Momentos (`docs/01_formulacao_problema/momentos_previsao.md`): T0 pedido, T2 *pricing*, T4 concessão. A regressão prevê
em T0, a classificação em T4 e o clustering usa o perfil em T0.

---

## Resumo

| Feature                 | Autor  | Fórmula                                       | Momento | Regressão | Classificação | Clustering |
|-------------------------|--------|-----------------------------------------------|---------|-----------|---------------|------------|
| `loan_to_income`        | João   | `loan_amnt / annual_inc`                      | T0      | ✅        | ✅            | ✅         |
| `installment_to_income` | João   | `100 × installment / (annual_inc / 12)`       | T2      | ❌        | ✅            | ❌         |
| `dti_with_loan`         | João   | `dti + 100 × installment / (annual_inc / 12)` | T2      | ❌        | ✅            | ❌         |
| `credit_history_years`  | João   | `(issue_d − earliest_cr_line)` em meses / 12  | T4      | ❌        | ✅            | ❌         |
| `issue_year`            | João   | ano de `issue_d`                              | T4      | ❌        | ⚠️ ✅         | ❌         |
| `revol_to_income`       | Rodrigo | `revol_bal / annual_inc`                     | T0      | ✅        | ✅            | ✅         |
| `open_acc_pct`          | Rodrigo | `100 × open_acc / total_acc`                 | T0      | ✅        | ✅            | ✅         |
| `ever_delinquent`       | Rodrigo | 1 se `delinq_2yrs > 0` ou `mths_since_last_delinq` preenchido | T0 | ✅ | ✅       | ✅         |
| `has_public_record`     | Rodrigo | 1 se `pub_rec > 0`                           | T0      | ✅        | ✅            | ✅         |
| `total_credit_util`     | Rodrigo | `100 × tot_cur_bal / tot_hi_cred_lim`        | T0      | ✅        | ❌            | ❌         |

## Fichas

### 1. `loan_to_income` - montante / rendimento (João)

- **Fontes:** `loan_amnt`, `annual_inc`
- **Fórmula:** `loan_amnt / annual_inc`
- **Motivo:** o mesmo montante é um esforço diferente para rendimentos diferentes. Na EDA, o montante e o rendimento
  estão correlacionados (ρ = 0,50), mas nenhum dos dois, sozinho, mede o esforço do pedido.
- **Valores em falta:** NaN se `annual_inc` faltar ou for ≤ 0; imputação pela mediana + indicador.
- **Elegibilidade:** T0 (ambas as colunas existem no pedido) -> regressão, classificação e clustering.

### 2. `installment_to_income` - prestação / rendimento mensal (João)

- **Fontes:** `installment`, `annual_inc`
- **Fórmula:** `100 × installment / (annual_inc / 12)` (% do rendimento mensal, como o `dti`)
- **Motivo:** mede o peso da prestação no orçamento mensal. Na EDA, a prestação em valor absoluto não tem relação com o
  *default*; em proporção do rendimento, passa a medir o esforço.
- **Valores em falta:** NaN se `annual_inc` faltar ou for ≤ 0; imputação + indicador.
- **Elegibilidade:** T2 (`installment` é calculada a partir da taxa) -> só classificação. Na regressão é proibida pelo
  enunciado.

### 3. `dti_with_loan` - DTI com o novo empréstimo (João)

- **Fontes:** `dti`, `installment`, `annual_inc`
- **Fórmula:** `dti + 100 × installment / (annual_inc / 12)`
- **Motivo:** o `dti` da LendingClub não inclui o empréstimo pedido. Somar a nova prestação dá o endividamento do
  mutuário depois de o receber, que é o que determina a capacidade de pagar.
- **Valores em falta:** NaN se `dti`, `installment` ou `annual_inc` faltarem; imputação + indicador.
- **Elegibilidade:** T2 -> só classificação.
- **Nota:** é uma combinação linear de `dti` e `installment_to_income`. Nos modelos lineares e no kNN usa-se só uma das
  duas; nas árvores podem ficar ambas. A escolha é feita na validação cruzada.

### 4. `credit_history_years` - antiguidade do crédito (João)

- **Fontes:** `issue_d`, `earliest_cr_line`
- **Fórmula:** `(issue_d − earliest_cr_line)` em meses de calendário / 12
- **Motivo:** um historial de crédito curto dá menos informação sobre o comportamento do mutuário. O enunciado pede que,
  no classificador, a antiguidade seja calculada em relação a `issue_d` (e não à data de hoje ou a um *credit pull*
  posterior).
- **Valores em falta:** NaN se alguma das datas faltar ou se a diferença for negativa (data impossível); imputação +
  indicador.
- **Elegibilidade:** T4 -> só classificação. Na regressão, `issue_d` é uma data futura em relação ao pedido; aí a
  antiguidade vem de `mo_sin_old_rev_tl_op`, calculada pelo *bureau*.

### 5. `issue_year` - ano de concessão (João)

- **Fontes:** `issue_d`
- **Fórmula:** ano de `issue_d`
- **Motivo:** capta mudanças de política e de contexto económico. Na EDA, a população de 2013–2014 é mais arriscada do
  que a de 2007–2011 (FICO mediano de 710 -> 685).
- **Valores em falta:** nenhum (`issue_d` existe sempre).
- **Elegibilidade:** T4 -> só classificação, como condicional (⚠️): o enunciado permite datas de concessão no
  classificador "quando justificado".
- **Nota:** o teste (2015) está fora dos anos de treino. Como o pipeline limita cada variável aos percentis do treino,
  um ano posterior é tratado como o último ano do treino (a tendência não é extrapolada). A validação temporal, em que
  cada *fold* valida um ano que o treino não tem, mostra se a feature ajuda; a decisão de a manter é tomada aí.

### 6. `revol_to_income` - saldo rotativo / rendimento (Rodrigo)

- **Fontes:** `revol_bal`, `annual_inc`
- **Fórmula:** `revol_bal / annual_inc`
- **Motivo:** o `revol_util` mede a parte do limite dos cartões que está usada, mas não diz se essa dívida é grande para
  o rendimento do mutuário. No treino da classificação, o quintil mais alto tem 14,9% de *default*, contra cerca de
  11,3% nos três primeiros. A correlação com o `revol_util` (ρ = 0,34) e com o `dti` (ρ = 0,40) é moderada, por isso
  acrescenta informação.
- **Valores em falta:** NaN se `revol_bal` faltar ou se `annual_inc` faltar ou for ≤ 0; imputação + indicador.
- **Elegibilidade:** T0 (ambas as colunas existem no pedido) -> regressão, classificação e clustering.

### 7. `open_acc_pct` - % de contas abertas (Rodrigo)

- **Fontes:** `open_acc`, `total_acc`
- **Fórmula:** `100 × open_acc / total_acc`
- **Motivo:** o mesmo número de contas abertas significa coisas diferentes para quem teve 5 ou 50 contas ao longo do
  historial; uma percentagem alta indica crédito recente ou acumulado. No treino da classificação, o quintil mais alto
  tem 15,5% de *default*, contra 10,5% a 12,1% nos outros (relação não linear). A correlação com o `open_acc` é só
  ρ = 0,26.
- **Valores em falta:** NaN se alguma das colunas faltar, se `total_acc` for ≤ 0 ou se `open_acc` for maior do que
  `total_acc` (caso impossível: 2 empréstimos de 2007, já assinalados em `valores_suspeitos.md`); imputação + indicador.
- **Elegibilidade:** T0 -> regressão, classificação e clustering.

### 8. `ever_delinquent` - já teve atrasos (Rodrigo)

- **Fontes:** `delinq_2yrs`, `mths_since_last_delinq`
- **Fórmula:** 1 se `delinq_2yrs > 0` ou `mths_since_last_delinq` preenchido; senão 0
- **Motivo:** o `delinq_2yrs` só conta os atrasos dos últimos 2 anos, e o `mths_since_last_delinq` vazio significa
  "nunca houve atraso" (`valores_em_falta.md`, os zeros disfarçados de 2007–2008 já foram corrigidos). Juntar as duas dá
  o historial completo num valor simples e resolve o problema de um vazio que quer dizer ausência de evento, que um
  modelo trataria como valor desconhecido. 48% dos empréstimos da regressão têm 1. No treino da classificação, o
  *default* é 13,5% com atraso e 11,7% sem.
- **Valores em falta:** NaN só se as duas colunas faltarem (29 empréstimos no treino da classificação), porque aí não se
  sabe se houve atraso; `mths_since_last_delinq` vazio sozinho dá 0. Os NaN são imputados + indicador.
- **Elegibilidade:** T0 (histórico do *bureau*) -> regressão, classificação e clustering.

### 9. `has_public_record` - tem registos públicos (Rodrigo)

- **Fontes:** `pub_rec`
- **Fórmula:** 1 se `pub_rec > 0`; senão 0
- **Motivo:** quase todos os mutuários têm 0 registos públicos negativos (falências, penhoras) e a contagem tem uma cauda
  longa, por isso a diferença que importa é entre nenhum e pelo menos um. É a feature nova do Rodrigo com mais efeito no
  *default*: no treino da classificação, 19,9% com registo contra 11,9% sem. Na regressão, a taxa média é 14,70%
  contra 13,93%.
- **Valores em falta:** NaN se `pub_rec` faltar (29 empréstimos no treino da classificação); imputação + indicador.
- **Elegibilidade:** T0 -> regressão, classificação e clustering.

### 10. `total_credit_util` - utilização total do crédito (Rodrigo)

- **Fontes:** `tot_cur_bal`, `tot_hi_cred_lim`
- **Fórmula:** `100 × tot_cur_bal / tot_hi_cred_lim`
- **Motivo:** mede quanto de todo o crédito (cartões, empréstimos e hipotecas) ainda está em dívida, enquanto o
  `revol_util` só olha para o crédito rotativo. Na regressão, é a feature nova com mais relação com a taxa a seguir ao
  `loan_to_income` (ρ = 0,11 com o `int_rate`, contra 0,20), e a correlação com o `revol_util` é só ρ = 0,39.
- **Valores em falta:** NaN se alguma das colunas faltar ou se `tot_hi_cred_lim` for ≤ 0 (15 empréstimos na regressão);
  imputação + indicador. Os valores acima de 100% (saldo acima do limite, 5 536 empréstimos) ficam; o
  *clipping* aos percentis do treino limita os extremos.
- **Elegibilidade:** T0 -> **só regressão**. As duas colunas são das 37 do *bureau* que só existem a partir de 2012
  (`valores_em_falta.md`, secção 4): estão vazias em todo o treino da classificação (2007–2011) e em 2007–2012 no
  clustering.