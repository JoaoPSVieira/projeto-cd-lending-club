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
|                         | Soares |                                               |         |           |               |            |
|                         | Soares |                                               |         |           |               |            |
|                         | Soares |                                               |         |           |               |            |
|                         | Soares |                                               |         |           |               |            |
|                         | Soares |                                               |         |           |               |            |

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