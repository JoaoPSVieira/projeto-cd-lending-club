# Classificação: risco de *default* na concessão (Fase 4 e 5)

Bloco 5 do plano da Parte 2. Prevê, no momento da concessão, se um empréstimo de 36 meses
fica em *Good standing* ou entra em *Default*, compara os 8 modelos do catálogo na mesma
validação cruzada temporal, interpreta o modelo escolhido e define o modelo final para o teste
de 2015 e para o *deployment*.

| Ficheiro | Conteúdo |
|---|---|
| `notebooks/04c_classificacao.ipynb` | O bloco inteiro, com os resultados gravados (11 secções) |
| `src/lending/model_catalog.py` | `classification_specs()` (os 8 modelos e as grelhas) e `final_classification_spec()` (o modelo final) |
| `src/lending/experiment.py`, `evaluation.py`, `baselines.py` | Runner, métricas, limiar, bootstrap e baselines (bloco 1, João) |
| `src/lending/knn.py`, `boosting.py`, `deep.py` | kNN de raiz (João), LightGBM com paragem antecipada (João), MLP em Keras (bloco 3) |
| `reports/results/classification_*.csv` | Resultados (secção 9 deste documento) |
| `reports/figures/mod_cls_*.png` | Figuras do relatório |
| `data/processed/oof/classification/` | Previsões fora do fold de cada configuração (fora do git) |

---

## 1. Requisitos do enunciado

| Requisito | Onde |
|---|---|
| *"Apply your kNN implementation to ... classification, evaluate it, and discuss the results"* | 04c, secção 5 (kNN com pesos das classes) |
| *"Two supervised models from scikit-learn (other than ensemble models)"* | 04c, secção 3: regressão logística L2 e árvore de decisão |
| *"One bagging and one boosting ensemble model"* | 04c, secção 4: Random Forest e LightGBM |
| *"A deep learning model (TensorFlow or PyTorch) ... justify the architecture"* | 04c, secção 6; `docs/04_modelos/mlp.md` |
| *"Use the same eligible evaluation cohort and split definitions for all models within a task"* | Todos os modelos passam pelo mesmo `CrossValidationRunner`: mesma coorte, mesmos 3 folds, mesmo pré-processamento |
| *"Report training sample sizes, tuning procedures, and any computational limitations"* | Secção 2 (folds e tamanhos), tabelas de cada grelha (configuração, limiar, tempos), secção 8 deste documento |
| *"Prepare one comparison table for each task, including baselines"* | 04c, secção 7; `classification_comparison.csv` |
| *"Confusion matrix, precision, recall, F1 (macro and micro averages), AUC, and accuracy"* | 04c, secção 7: tabela final, matrizes de confusão (3 folds juntos e cada fold) |
| *"Present the mean and the variation of the metric in the cross-validation"* | Todas as tabelas: média ± desvio nos 3 folds |
| *"... and the mean in the test dataset"* | Bloco 8 (teste de 2015), com o modelo final definido aqui |
| *"Do not select anything by looking at held-out results"* | Configurações, limiar e modelo final escolhidos só com as previsões fora do fold de 2009–2011; o teste não foi usado |
| *"Resampling only within training folds; never rebalance the test"* | Não há reamostragem: o desequilíbrio é tratado com pesos das classes |
| *"`issue_d` ... permitted in the classifier if justified"* | 04c, secção 10: testado com e sem `issue_year`; sai do modelo final |
| *"Discuss temporal change ... and fairness"* | Deriva do limiar (secções 2, 5 e 7), `issue_year` (secção 10), limitações (secção 8 deste documento); fairness no teste (bloco 8) |

---

## 2. Coorte e validação

- **Coorte:** 31 522 empréstimos de 36 meses emitidos em 2007–2011, com o desfecho observado em
  T4 (Parte 1). 12,3% de *Default*. A classe *At risk* não existe no treino, por isso a
  classificação é binária; no teste, a tabela é 3 × 2 (as três classes reais contra as duas
  previstas).
- **Janela expansível anual:** cada fold treina com todos os anos anteriores e valida no ano
  seguinte, para nunca aprender com o futuro.

  | Fold | Treino | Validação | *Default* no treino | *Default* na validação |
  |---|---|---|---|---|
  | 1 | 2007–2008 (2 991) | 2009 (5 280) | 21,9% | 13,7% |
  | 2 | 2007–2009 (8 271) | 2010 (9 153) | 16,6% | 10,9% |
  | 3 | 2007–2010 (17 424) | 2011 (14 098) | 13,6% | 10,6% |

- **Previsões fora do fold:** cada empréstimo de 2009–2011 recebe uma previsão de um modelo
  que não o viu (28 531 no total). São guardadas e servem para o limiar, as matrizes de
  confusão, o *bootstrap* e as figuras.
- **Desequilíbrio:** `class_weight="balanced"` em todos os modelos supervisionados (o kNN pondera
  os votos pela classe), sem reamostragem.
- **Limiar:** para cada configuração, o limiar entre 0,05 e 0,95 que maximiza o F1 macro nas
  previsões fora do fold dos 3 folds juntos. Os resultados com o limiar de 0,5 ficam no CSV do
  runner, para comparar.
- **Escolha da configuração:** o F1 macro médio nos 3 folds. A AUC é a segunda métrica.

---

## 3. Modelos

As grelhas estão em `docs/04_modelos/runner.md` (secção 3 do relatório). A configuração
escolhida de cada modelo:

| Modelo | Configuração escolhida | Notas |
|---|---|---|
| Classe maioritária | — | Baseline: prevê sempre *Good standing* |
| Taxa do grau | — | Baseline: a taxa de *default* do grau no treino |
| kNN (de raiz) | k = 101, `distance` | Votos ponderados pela classe; sem `dti_with_loan` |
| Regressão logística L2 | C = 0,1 | Sem `dti_with_loan` |
| Árvore de decisão | profundidade 6, folha ≥ 50 | |
| Random Forest | `max_features` 0,3, folha ≥ 50 | 300 árvores, `balanced_subsample` |
| LightGBM | taxa 0,1, 63 folhas | Paragem antecipada nos 15% mais recentes do treino |
| MLP (Keras) | (256, 128, 64), taxa 0,001 | Paragem antecipada pela AUC nos 15% mais recentes |

---

## 4. Resultados

**Tabela de comparação** (configuração escolhida; média ± desvio nos 3 folds):

| # | Modelo | Limiar | F1 macro | AUC | Accuracy | Recall *Default* |
|---|---|---|---|---|---|---|
| 1 | MLP | 0,56 | 0,573 ± 0,008 | 0,666 ± 0,025 | 0,819 ± 0,023 | 0,257 ± 0,005 |
| 2 | Regressão logística | 0,56 | 0,570 ± 0,010 | 0,669 ± 0,021 | 0,814 ± 0,028 | 0,259 ± 0,020 |
| 3 | Random Forest | 0,52 | 0,564 ± 0,021 | 0,662 ± 0,036 | 0,796 ± 0,067 | 0,283 ± 0,073 |
| 4 | LightGBM | 0,52 | 0,561 ± 0,008 | 0,644 ± 0,031 | 0,801 ± 0,051 | 0,265 ± 0,059 |
| 5 | kNN | 0,52 | 0,555 ± 0,011 | 0,642 ± 0,028 | 0,792 ± 0,026 | 0,266 ± 0,010 |
| 6 | Árvore | 0,62 | 0,541 ± 0,011 | 0,613 ± 0,007 | 0,816 ± 0,051 | 0,187 ± 0,094 |
| 7 | Taxa do grau | 0,21 | 0,513 ± 0,045 | 0,630 ± 0,020 | 0,733 ± 0,174 | 0,344 ± 0,298 |
| 8 | Classe maioritária | — | 0,469 ± 0,005 | 0,500 | 0,882 ± 0,017 | 0 |

O F1 *micro* é igual à *accuracy* (classificação binária) e a tabela completa, com a precisão e
o recall *macro* e *micro*, está em `classification_comparison.csv`.

**Bootstrap emparelhado** (1 000 reamostragens das previsões fora do fold; diferença e IC 95%):

| A − B | F1 macro | AUC |
|---|---|---|
| MLP − logística | +0,003 [−0,003; 0,010] | −0,003 [−0,008; 0,002] |
| MLP − Random Forest | +0,009 [0,003; 0,016] | +0,003 [−0,002; 0,009] |
| Logística − Random Forest | +0,006 [−0,001; 0,013] | +0,006 [0,001; 0,012] |
| MLP − LightGBM | +0,012 [0,004; 0,020] | +0,022 [0,013; 0,030] |

A MLP e a logística não se distinguem; a Random Forest fica logo atrás; os outros ficam
claramente abaixo. A tabela completa está em `classification_bootstrap.csv`.

**Leituras principais:**

- **Limiar contra 0,5:** com pesos das classes, o limiar escolhido fica entre 0,52 e 0,62 e
  ganha 0,003 a 0,015 de F1 macro. Sem pesos (taxa do grau), com 0,5 não se prevê nenhum
  *Default*.
- **Otimismo do limiar:** escolhido só com os folds anteriores, o F1 macro do fold 3 muda no
  máximo 0,002.
- **Matrizes de confusão:** no fold 3 (2011, 10,6% de *Default*), a logística marca 10,9% dos
  empréstimos, com precisão de *Default* 0,23 (2,2 vezes a taxa base) e recall 0,24. Todos os
  modelos supervisionados ficam na mesma zona: com a informação da concessão, a maioria dos
  *Default* parece-se com empréstimos bons.
- **Deriva no tempo:** a taxa de *default* do treino desce de 21,9% para 13,6%. Os modelos sem
  pesos (taxa do grau, kNN sem pesos) seguem-na, e a percentagem marcada com um limiar fixo
  cai de ano para ano; com pesos, fica estável.
- **Mais dados ajudam os modelos flexíveis:** a MLP e os *ensembles* ficam atrás da logística
  nos folds pequenos e aproximam-se (ou passam-na) no fold 3.

---

## 5. Interpretação

Do modelo escolhido (regressão logística), 04c secções 8 e 9. Os *odds ratios* são de *Default*,
por 1 desvio-padrão, com as outras colunas fixas:

- **Aumentam o risco:** taxa de juro, pedidos de crédito nos últimos 6 meses, registos
  públicos, utilização do crédito rotativo, dívida em relação ao rendimento, finalidade
  *small business* (2,5 vezes as *odds* de *debt consolidation*).
- **Diminuem o risco:** FICO, rendimento, finalidade *credit card*.
- **Sem efeito, dado o resto:** DTI, montante do empréstimo, verificação do rendimento.
- **Sub-grau:** o efeito condicional chega a cerca de 4 (A1 → F/G), contra 13–28 observado: o
  grau resume as outras características.
- **Importância por permutação** (fora do fold, nos três primeiros modelos): rendimento, FICO,
  finalidade, pedidos de crédito e taxa de juro, pela mesma ordem nos três.
- **Dependência parcial:** a MLP desenha as mesmas curvas quase lineares da logística, o que
  explica o empate.

Cuidados de leitura, explicados no notebook: o sinal do scikit-learn (o OR de *Default* é
e^(−β)), as colunas quase iguais (lidas em grupo), as categorias (contra a mais frequente) e a
armadilha da permutação com colunas ligadas (`open_acc` e `total_acc`).

---

## 6. Modelo final

`final_classification_spec()`: **regressão logística L2, C = 0,1, classes ponderadas, sem
`dti_with_loan` nem `issue_year`.**

- **Porquê a logística:** empata com a MLP no *bootstrap* emparelhado e é a mais simples
  (parcimónia), interpretável (*odds ratios*), rápida (0,1 s) e determinística, sem
  TensorFlow no *deployment*. A MLP fica como alternativa.
- **Porquê sem `issue_year`:** é a data de financiamento, não uma característica do pedido.
  O clipping prende-o ao último ano do treino (no teste seria sempre 2011), e sem ele o F1
  macro é o mesmo (0,570) e a AUC perde 0,001.
- **Na validação cruzada:** F1 macro 0,570 ± 0,008, AUC 0,668 ± 0,020, recall de *Default*
  0,257 ± 0,015; marca 13,7%, 12,6% e 12,0% dos empréstimos nos três folds.
- **Limiar fixo: 0,57**, escolhido nas previsões fora do fold desta configuração. Nunca é
  mudado com o teste.
- **111 colunas**, treinado em 31 522 empréstimos (2007–2011).
- **Ficha:** `reports/results/classification_final_model.csv`. *Odds ratios* finais com
  intervalos: `classification_final_odds_ratios.csv`.

**Odds ratios do modelo final** (de *Default*, por 1 DP, IC 95%; 04c, secção 11):

| Aumentam o risco | OR | Diminuem o risco | OR |
|---|---|---|---|
| Pedidos de crédito (6 meses, +1,5) | 1,28 [1,24; 1,33] | FICO (+37 pontos) | 0,81 [0,73; 0,88] |
| Taxa de juro (+3,3 p.p.) | 1,24 [1,10; 1,40] | Rendimento (× 1,76) | 0,84 [0,77; 0,90] |
| Registos públicos | 1,23 [1,09; 1,36] | *credit card* (vs *debt consolidation*) | 0,74 [0,64; 0,84] |
| *small business* (vs *debt consolidation*) | 2,51 [2,15; 2,88] | | |

Sem efeito claro: DTI, montante do empréstimo, verificação do rendimento.

**Como usar (teste de 2015 e *deployment*):**

```python
spec = final_classification_spec()
final_model = runner.refit(data, spec, cid=spec.configs()[0][0])   # treino 2007-2011
p_default = positive_proba(final_model, X_novos)                    # colunas originais
previsto = np.where(p_default >= 0.57, "Default", "Good standing")
```

As probabilidades estão na escala dos pesos *balanced*, como se o treino tivesse 50% de cada
classe (a média prevista é cerca de 45%). Para as ler como probabilidades reais, corrige-se
com odds × 0,123 / 0,877 (04c, secção 9).

---

## 7. Alterações ao código feitas neste bloco

| Commit | Alteração |
|---|---|
| `5eb0d86` | `knn.py`: `class_weight` no `KNNClassifier` (votos ponderados pela classe); o catálogo usa `"balanced"`; 6 testes novos |
| `6d88601` | `deep.py`: filtro no logger do TensorFlow, que no Jupyter repunha o nível em INFO |
| `7de0c39` | `model_catalog.py`: `final_classification_spec()`, com teste |

---

## 8. Limitações

- **Mudança no tempo:** a taxa de *default* desceu de ano para ano. Os pesos das classes
  estabilizam o limiar, mas a percentagem marcada no teste pode mudar se a população mudar.
- **Folds pequenos:** o fold 1 só tem 2 991 empréstimos de treino; os desvios entre folds são
  grandes e assentam em 3 valores. O *bootstrap* emparelhado mede a incerteza da validação, não
  a do treino.
- **Limiar:** escolhido nas mesmas previsões em que se mede o F1 (otimismo medido: ≤ 0,01).
- **Viés de seleção:** só há desfechos dos empréstimos aceites. O modelo descreve o risco na
  população que a LendingClub aprovou, não na dos pedidos recusados.
- **Seguimento:** o desfecho é o observado em T4, como definido na Parte 1.
- **Interpretação:** os *odds ratios* são associações condicionais, encolhidas pela L2, não
  causas. A importância por permutação subestima colunas que se substituem (grau e taxa).
- **Fairness:** o modelo não usa atributos protegidos, mas a finalidade e a situação da casa
  podem ser *proxies* socioeconómicos. A análise faz-se no teste (bloco 8).
- **Custo:** o notebook corre em cerca de 25 minutos com os resultados já no CSV (a MLP é
  treinada mais do que uma vez, e há três *bootstraps*).

---

## 9. Resultados gravados (`reports/results/`)

| Ficheiro | Conteúdo |
|---|---|
| `classification_cv.csv` | Uma linha por modelo × configuração × fold, com as métricas no limiar de 0,5 e os tempos (o runner retoma a partir dele) |
| `classification_comparison.csv` | Tabela de comparação final (secção 7 do 04c) |
| `classification_bootstrap.csv` | Diferenças do *bootstrap* emparelhado |
| `classification_odds_ratios.csv` | *Odds ratios* da logística escolhida (com `issue_year`), secção 8 |
| `classification_permutation_importance.csv` | Queda da AUC por coluna, modelo e fold, secção 9 |
| `classification_final_model.csv` | Ficha do modelo final: parâmetros, colunas tiradas, limiar, métricas |
| `classification_final_odds_ratios.csv` | *Odds ratios* do modelo final, com intervalos |

**Para correr de novo:** o runner salta as combinações que já estão no `classification_cv.csv`,
mas lê as previsões fora do fold em `data/processed/oof/classification/` (fora do git). Para
refazer um modelo, apagam-se as suas linhas do CSV e a sua pasta de previsões.

---

## 10. Perguntas prováveis na defesa

| Pergunta | Resposta curta |
|---|---|
| Porquê uma janela expansível anual? | Treinar só com o passado e validar no ano seguinte, como no uso real; folds aleatórios deixavam o modelo aprender com o futuro |
| Porquê o F1 macro e não a *accuracy*? | Com 88% de bons pagadores, prever sempre *Good standing* dá *accuracy* 0,88 e não serve para nada; o F1 macro dá o mesmo peso às duas classes |
| Como trataram o desequilíbrio? | Pesos das classes em todos os modelos (no kNN, votos ponderados), sem reamostragem |
| Porque é que o limiar não é 0,5? | É o que maximiza o F1 macro nas previsões fora do fold; com pesos fica perto de 0,5 (0,57 no modelo final) |
| Qual é o melhor modelo? | A MLP e a logística empatam (o *bootstrap* emparelhado não as separa); escolhemos a logística por parcimónia |
| Porque tiraram o `issue_year`? | É a data de financiamento, não uma característica do pedido; no teste seria sempre 2011 (clipping) e sem ele o desempenho é o mesmo |
| Uma precisão de 0,23 não é baixa? | A taxa base é 10,6%: entre os marcados, há 2,2 vezes mais *Default*. Todos os modelos ficam na mesma zona |
| O DTI não importa? | Sozinho sim (EDA), mas com os rácios ao rendimento, o crédito rotativo, o FICO e a taxa, não acrescenta nada |
