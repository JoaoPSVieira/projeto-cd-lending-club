# Validação cruzada, métricas e catálogo de modelos

Bloco 1 do plano da Parte 2. É a infraestrutura comum a todos os modelos supervisionados.

| Ficheiro | Conteúdo |
|---|---|
| `src/lending/experiment.py` | `TaskSetup`, `ModelSpec`, `CrossValidationRunner` (validação cruzada temporal, resultados, resumo, modelo final) |
| `src/lending/evaluation.py` | Métricas de regressão e de classificação, escolha do limiar, matriz de confusão, bootstrap |
| `src/lending/baselines.py` | `TermMedianBaseline` (mediana por prazo) e `GradeRateBaseline` (taxa de *default* do grau) |
| `src/lending/boosting.py` | LightGBM com paragem antecipada nas linhas mais recentes |
| `src/lending/model_catalog.py` | Modelos e grelhas da secção 3: `regression_specs()` e `classification_specs()` |
| `tests/test_evaluation.py`, `tests/test_experiment.py`, `tests/test_boosting.py` | 40 testes (cerca de 5 s; o teste da grelha da MLP só corre com o `deep.py`) |

---

## 1. Requisitos do enunciado

| Requisito | Onde |
|---|---|
| *"Use the same eligible evaluation cohort and split definitions for all models within a task"* | Todos os modelos passam pelo mesmo `CrossValidationRunner`: mesma coorte, mesmos folds, mesmo pré-processamento |
| *"Report training sample sizes, tuning procedures, and any computational limitations"* | Cada linha de resultados tem `n_train`, a configuração e os tempos (`prep_seconds`, `fit_seconds`, `predict_seconds`) |
| *"Prepare one comparison table for each task, including baselines"* | `runner.comparison_table()`, com as baselines do catálogo |
| *"Regression: MAE, RMSE, and R2"* | `regression_metrics` |
| *"Classification: confusion matrix, precision, recall, F1 (macro and micro averages), AUC, and accuracy"* | `classification_metrics`, `confusion_table` |
| *"Present the mean and the variation of the metric in the cross-validation procedures and the mean in the test dataset"* | `runner.summary()` (média e desvio-padrão nos folds); no teste, `bootstrap_ci` |
| *"Two supervised models from scikit-learn (other than ensemble models)"* | Ridge / regressão logística e árvore de decisão |
| *"One bagging and one boosting ensemble model for each task"* | Random Forest e LightGBM |

---

## 2. Contrato com os modelos: dados ordenados por `issue_d`

**Os pickles não vêm ordenados por data.** A MLP e o LightGBM usam as últimas linhas do treino
de cada fold (15%) para a paragem antecipada, e essas linhas só são as mais recentes se os dados
estiverem ordenados. Por isso:

- o `runner.prepare(data)` ordena sempre por `issue_d` (ordenação estável) antes de criar os
  folds, seja qual for a ordem dos dados de entrada;
- os folds da `ExpandingWindowSplit` mantêm essa ordem: o treino de cada fold vai do empréstimo
  mais antigo ao mais recente;
- as amostras (`max_train_rows`) são tiradas ao acaso, mas as linhas escolhidas ficam pela
  ordem temporal.

O teste `test_models_receive_training_rows_sorted_by_date` baralha os dados e confirma que
todos os modelos recebem o treino ordenado.

---

## 3. Como usar (notebooks 04b e 04c)

```python
from lending import config
from lending.experiment import CrossValidationRunner
from lending.model_catalog import classification_specs
from lending.storage import PickleStore

data = PickleStore.load(config.CLS_TRAIN_PKL, expected_type=pd.DataFrame)
runner = CrossValidationRunner("classification")
runner.folds(data)                         # tabela dos 3 folds

specs = classification_specs()             # baselines + 6 famílias (kNN, linear, árvore, RF, LightGBM, MLP)
runner.run(data, specs)                    # retoma onde parou se for interrompido

runner.summary()                           # todas as configurações, a melhor primeiro
runner.comparison_table()                  # melhor configuração de cada modelo: "média ± desvio"
oof = runner.oof("LightGBM", config_id)    # previsões fora do fold (gráficos, limiar)

spec = next(s for s in specs if s.name == "LightGBM")
final = runner.refit(data, spec)           # melhor configuração treinada em todo o treino
threshold = runner.threshold("LightGBM", runner.best().set_index("model").loc["LightGBM", "config_id"])
```

Para correr só um modelo (por exemplo, a MLP numa noite), passa-se só esse `ModelSpec`. Os
resultados vão para o mesmo CSV.

---

## 4. Como o runner funciona

1. **Ordena** por `issue_d` e confirma colunas e alvo (na classificação, só `Good standing` e
   `Default`; um `At risk` no treino dá erro).
2. **Folds:** 6 trimestrais na regressão (validam 2013T3 a 2014T4) e 3 anuais na classificação
   (validam 2009, 2010 e 2011).
3. **Pré-processamento uma vez por fold.** O `FeatureBuilder` e o `build_preprocessor` são
   ajustados no treino do fold e reutilizados em todas as configurações desse fold. O resultado
   é o mesmo de um `Pipeline` completo por configuração, porque o pré-processamento não depende
   do modelo, e poupa cerca de 240 ajustes repetidos na regressão. O tempo fica em `prep_seconds`.
4. **Para cada configuração:**
    - tira as colunas de `drop_features`;
    - amostra o treino se o fold tiver mais do que `max_train_rows` linhas;
    - ajusta, prevê a validação do fold e mede tempos e métricas;
    - grava as previsões fora do fold e acrescenta uma linha ao CSV.
5. **Retoma:** as combinações (modelo, configuração, fold) que já estão no CSV são saltadas.

### Ficheiros

| Ficheiro | Conteúdo | Git |
|---|---|---|
| `reports/results/<tarefa>_cv.csv` | Uma linha por modelo × configuração × fold | Entra (exceção no `.gitignore`) |
| `data/processed/oof/<tarefa>/<modelo>/<config_id>_fold<k>.pkl` | Previsões fora do fold: `row`, `issue_d`, `y_true`, `y_pred` ou `p_default` | Fica fora |

Colunas do CSV: `task`, `model`, `config_id`, `params` (JSON), `fold`, `train_periods`,
`valid_period`, `n_valid`, `n_train`, `prep_seconds`, `fit_seconds`, `predict_seconds`,
`threshold` (só na classificação), as métricas e `timestamp`.

O `config_id` é um *hash* do modelo e dos parâmetros, por isso é o mesmo em todas as execuções.
**Se uma grelha mudar**, as configurações novas têm `config_id` novos e as antigas ficam no CSV.
Para começar do zero, apaga-se o CSV da tarefa e a pasta `oof` respetiva.

---

## 5. Classificação: limiar e métricas

- No CSV, as métricas com limiar usam **0,5**, para cada linha ser autónoma.
- O `summary()` escolhe para cada configuração o limiar (de 0,05 a 0,95, passo 0,01) que
  maximiza o **F1 macro nas previsões fora do fold de todos os folds juntos**, e recalcula as
  métricas de cada fold com esse limiar. A AUC não depende do limiar.
- O limiar fica fixo para o teste (`runner.threshold(...)`) e nunca é escolhido com o teste.
- Como o limiar é escolhido nas mesmas previsões em que se mede o F1, o F1 da validação
  cruzada fica ligeiramente otimista. É um único parâmetro escolhido em milhares de
  previsões, por isso o efeito é pequeno, mas deve ser dito no relatório.
- Em classificação binária, o F1, a precisão e o recall *micro* são iguais à *accuracy*. Não é
  um erro (`test_micro_metrics_equal_accuracy_in_binary_case`).
- `positive_proba` escolhe a coluna de `predict_proba` pelo nome da classe. No scikit-learn,
  `classes_` está por ordem alfabética e `Default` é a **primeira** coluna.

---

## 6. Catálogo de modelos (secção 3)

| Modelo | Regressão | Classificação | Grelha | Configurações |
|---|---|---|---|---:|
| Baselines | Mediana; mediana por prazo | Classe maioritária; taxa do grau | — | 1 cada |
| Linear | `Ridge` | `LogisticRegression` (classes ponderadas; sem `dti_with_loan`) | 6 valores de 10⁻³ a 10² | 6 |
| Árvore | `DecisionTreeRegressor` | `DecisionTreeClassifier` (classes ponderadas) | profundidade {4, 6, 8, 12} × folha {50, 200} | 8 |
| Random Forest | 300 árvores, `max_samples` 0,3 | Igual, `balanced_subsample` | `max_features` {√p, 0,3} × folha {10, 50} | 4 |
| LightGBM | `LGBMEarlyStoppingRegressor` | `LGBMEarlyStoppingClassifier` (classes ponderadas) | `learning_rate` {0,05; 0,1} × `num_leaves` {15, 31, 63} | 6 |
| MLP | `KerasMLPRegressor` | `KerasMLPClassifier` | 3 arquiteturas × 2 taxas de aprendizagem | 6 |
| kNN (`knn.py`) | 50 000 linhas por fold | Treino inteiro; sem `dti_with_loan` | k {5, 15, 31, 51, 101} × pesos {uniformes, distância} | 10 |

Total: 40 configurações por tarefa, mais as baselines, como na secção 3.

**LightGBM.** O `LGBMRegressor` só para cedo se receber um conjunto de validação no `fit`, e o
runner só chama `fit(X, y)`. O wrapper de `boosting.py` separa ele próprio as últimas 15%
linhas (as mais recentes) para a paragem antecipada, com a mesma regra da MLP: no máximo 500
árvores, paragem ao fim de 20 sem melhoria. O critério é l2 na regressão e AUC na
classificação, como na MLP de classificação. As importâncias das colunas obtêm-se com
`model.feature_importances()`.

**`dti_with_loan`.** É `dti + installment_to_income`, uma combinação linear. Sai da regressão
logística e do kNN (`drop_features`, que também tira o indicador de falta) e fica nas árvores,
como decidido na secção 3.

---

## 7. Avaliação final no teste (bloco 8)

```python
from lending.evaluation import (bootstrap_ci, classification_metrics, confusion_table,
                                labels_from_proba, positive_proba)

y_test = test[config.CLASS_COL]                         # três classes reais no teste
y_binary = y_test.replace({"At risk": "Default"})       # regra da Parte 1: At risk conta como Default
p = positive_proba(final, test[runner.columns])

classification_metrics(y_binary, p, threshold)
bootstrap_ci(lambda y, s: classification_metrics(y, s, threshold)["f1_macro"], y_binary, p)
confusion_table(y_test, labels_from_proba(p, threshold),
                row_labels=["Good standing", "At risk", "Default"])     # tabela 3 x 2
```

(Os nomes `test` e `config.CLASS_COL` são os da coorte de teste construída no bloco 8.)

`paired_bootstrap_diff` compara os dois melhores modelos nas mesmas linhas do teste.

---

## 8. Testes

| Ficheiro | O que se testa |
|---|---|
| `test_evaluation.py` | Métricas iguais às do scikit-learn; micro = accuracy no caso binário; coluna certa do `predict_proba`; F1 vetorizado igual ao do scikit-learn; limiar ótimo; tabela 3 × 2; bootstrap reprodutível; baselines |
| `test_experiment.py` | Treino entregue ordenado por data (dados baralhados); amostra ordenada e com o tamanho certo; uma linha por configuração e fold; retoma sem repetir; previsões fora do fold completas; `config_id` lido como texto; resumo e limiar; `ColumnDropper`; modelo final |
| `test_boosting.py` | O LightGBM aprende e para cedo; a paragem reage às últimas linhas; rótulos e pesos na classificação; `clone` e `pickle`; grelhas do catálogo iguais às da secção 3 |

Correr: `pytest -q`. Com a MLP do Rodrigo, são 63 testes em cerca de 1 minuto.

Verificação extra (fora do `pytest`): o runner correu com o pipeline real do projeto
(`FeatureBuilder` + `build_preprocessor`) e uma configuração de cada modelo, incluindo a MLP,
em dados **falsos** com as colunas dos pickles. As duas tarefas correram de início ao fim, com
o `refit` incluído. Os números dessa verificação não significam nada.
