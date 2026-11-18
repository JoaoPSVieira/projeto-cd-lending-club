# kNN de raiz em NumPy

Bloco 2 do plano da Parte 2.

| Ficheiro | Conteúdo |
|---|---|
| `src/lending/knn.py` | `NeighborSearch` (estrutura de dados), `KNNRegressor`, `KNNClassifier`, cache de vizinhos |
| `src/lending/model_catalog.py` | O kNN nas duas tarefas, com a grelha da secção 3 |
| `tests/test_knn.py` | Testes do kNN (correção, blocos, cache, erros, interface) |
| `notebooks/04a_knn.ipynb` | Explicação, verificação, *benchmark* contra a KD-tree e escolha de k no último fold |

---

## 1. Requisitos do enunciado

| Requisito | Como é cumprido |
|---|---|
| *"Implement the k-nearest neighbors (kNN) algorithm from scratch in Python, using NumPy arrays"* | `knn.py` só usa NumPy |
| *"... and a suitable data structure"* | `NeighborSearch`: matriz contígua com normas pré-calculadas, distâncias por blocos com produto de matrizes, seleção parcial (secção 3 deste documento) |
| *"Do not substitute a library kNN estimator"* | Nada de `sklearn.neighbors` no código; um teste lê os `import` do módulo e confirma-o. Do scikit-learn herda-se só a interface (`BaseEstimator`) |
| *"Explain and document your implementation"* | Este documento, as *docstrings* e o notebook 04a |
| *"Apply ... to regression and classification, evaluate it, and discuss the results"* | Está no catálogo das duas tarefas e é avaliado pelo runner nos mesmos folds dos outros modelos (04b, 04c) |

---

## 2. Algoritmo (T04, p. 34–38)

- **Treino:** guardar os dados de treino. Não há parâmetros a aprender.
- **Previsão:** para cada pedido, calcular a distância a todos os pontos de treino, escolher os
  k mais próximos e devolver:
    - na regressão, a média do alvo dos vizinhos;
    - na classificação, a proporção de cada classe entre eles. A proporção de *Default* é a
      "probabilidade" usada na AUC e no limiar; `predict` devolve a classe mais votada.
- **Pesos:** `"uniform"` (todos iguais) ou `"distance"` (1/d). Um vizinho a distância 0 tem um
  peso enorme e domina a previsão, como no scikit-learn.
- **Distância:** euclidiana, a indicada para variáveis numéricas (T04, p. 35). O pipeline
  padroniza todas as colunas, por isso nenhuma domina pela escala, que é o caso em que os
  slides sugerem a de Manhattan.

---

## 3. Estrutura de dados: `NeighborSearch`

| Passo | Como | Porquê |
|---|---|---|
| Guardar o treino | Matriz `float32`, contígua, e a norma ao quadrado de cada linha (‖x‖²), calculada uma vez | Metade da memória do `float64`; as normas não se repetem em cada pedido |
| Distâncias | Por blocos de 256 pedidos: ‖q − x‖² = ‖q‖² + ‖x‖² − 2 q·x, com q·x de todo o bloco num único produto de matrizes | O produto de matrizes usa BLAS (código otimizado); o bloco limita a memória |
| Escolher os k | `np.argpartition`: os k menores de cada linha sem a ordenar toda; depois ordenam-se só esses k | Custo linear por linha, em vez de n log n |
| Distância final | ‖q − x‖ recalculada diretamente para os k escolhidos | A fórmula do 2.º passo perde precisão em `float32` para pontos muito próximos, o que pesaria nos pesos 1/d |
| Empates | Ordenação pela distância e depois pelo índice | Resultado determinista |

**Custo:** o `fit` custa n × p (as normas). Cada `predict` custa m × n × p, no produto de
matrizes, mais m × n na seleção parcial (m pedidos, n pontos de treino, p colunas).

**Memória de um bloco:** cerca de 256 × n × 12 bytes (distâncias em `float32` e índices do
`argpartition` em `int64`). Com n = 50 000, são cerca de 150 MB.

**Tempos medidos** (na verificação deste bloco; variam com a máquina): 50 000 × 99 de treino e 4 000 pedidos
levam cerca de 1,5 s, ou seja, uns 25 s para os ~60 000 pedidos de validação de um fold da regressão.

### Porquê não uma KD-tree

Uma KD-tree divide o espaço em caixas e descarta, em cada pesquisa, as que não podem ter
vizinhos mais próximos. Com ~100 colunas, as distâncias ficam todas parecidas e quase nada é
descartado (maldição da dimensionalidade, T05, p. 4). O notebook 04a mede-o:

- em dados aleatórios, a KD-tree só ganha até cerca de 5 dimensões; a 100 dimensões foi mais de
  20 vezes mais lenta do que a força bruta (na verificação deste bloco; os tempos variam com a máquina);
- com as colunas reais do pipeline, a comparação repete-se no último fold.

A `KDTree` do scikit-learn só aparece nesse *benchmark*, nunca no modelo.

---

## 4. Vários k de uma vez

A grelha tem 5 valores de k × 2 pesos = 10 configurações por fold, todas com o mesmo treino e a
mesma validação. Para não calcular as mesmas distâncias 10 vezes (secção 3 do relatório):

- o catálogo cria todos os kNN com `search_k=101`: a pesquisa devolve sempre os 101 vizinhos e
  cada configuração usa os primeiros `n_neighbors`;
- a última pesquisa fica em `NEIGHBOR_CACHE`, identificada pelo conteúdo do treino e dos pedidos
  (um *hash* dos bytes). As outras 9 configurações do fold reutilizam-na;
- o resultado é o mesmo de calcular tudo de novo (`test_search_k_gives_the_same_predictions`).
  A cache tem uma só entrada e é substituída na pesquisa seguinte.

Com o runner, na regressão, isto dá 6 pesquisas em vez de 60 (uma por fold).

---

## 5. Uso nas duas tarefas

| | Regressão | Classificação |
|---|---|---|
| Treino | Amostra de 50 000 empréstimos do treino de cada fold (semente 42), pelo custo m × n | Treino inteiro do fold (até 17 424 empréstimos) |
| Colunas | Todas as do pipeline | Sem `dti_with_loan` (e o seu indicador de falta): é `dti + installment_to_income`, uma combinação linear |
| Grelha | k ∈ {5, 15, 31, 51, 101} × pesos {uniformes, distância} | Igual |

No teste de 2015, a regressão prevê ~420 000 empréstimos contra os 50 000 do treino: cerca de 3
minutos.

---

## 6. Testes (`tests/test_knn.py`)

| O que se testa | Testes |
|---|---|
| Vizinhos iguais a um cálculo direto, sem truques; distâncias ordenadas | `test_query_matches_direct_computation` |
| O tamanho do bloco não muda o resultado | `test_blocks_do_not_change_the_result` |
| k = n; cada ponto de treino é o seu vizinho a distância 0 | `test_query_with_k_equal_to_training_size`, `test_training_point_is_its_own_nearest_neighbour` |
| Regressão e classificação iguais às do scikit-learn (só como referência), com os dois pesos | `test_regressor_matches_sklearn`, `test_classifier_matches_sklearn` |
| `search_k` dá as mesmas previsões; a grelha calcula as distâncias uma vez; a cache não é usada com outros dados | `test_search_k_*`, `test_grid_computes_distances_once`, `test_cache_is_not_reused_for_other_data` |
| `Pipeline`, `clone`, `pickle`, colunas verificadas, parâmetros inválidos, NaN | `test_pipeline_clone_and_pickle` e seguintes |
| O módulo não importa `sklearn.neighbors` nem o SciPy | `test_knn_module_does_not_use_sklearn_neighbors` |

---

## 7. Perguntas prováveis na defesa

| Pergunta | Resposta curta |
|---|---|
| Qual é a estrutura de dados? | Matriz de treino em memória com as normas pré-calculadas; distâncias por blocos com um produto de matrizes; seleção parcial com `argpartition` |
| Porque não uma KD-tree? | Com ~100 colunas não poda quase nada e fica mais lenta do que a força bruta vetorizada; o notebook 04a mede-o |
| Porque é que o kNN da regressão usa só 50 000 empréstimos? | A previsão custa m × n; com o treino inteiro (até 370 000 empréstimos), demoraria até 7 vezes mais |
| Como se escolhe k? | Pelo erro de validação (T04, p. 38): no runner, a média do MAE ou do F1 macro nos folds |
| Porque é que o erro de treino com pesos pela distância é 0? | Cada ponto de treino é vizinho de si próprio a distância 0, e esse peso domina |
| Porque tirar o `dti_with_loan`? | É uma combinação linear de outras duas colunas: contaria duas vezes a mesma informação na distância |
| O que faz a cache? | Guarda os 101 vizinhos da última pesquisa; as configurações com o mesmo treino e os mesmos pedidos usam-nos em vez de recalcular |
