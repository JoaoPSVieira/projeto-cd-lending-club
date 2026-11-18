# Rede neuronal densa (MLP) em Keras (Fase 4)

O enunciado pede um modelo de *deep learning* (TensorFlow ou PyTorch) para a regressão e para a classificação, com a
arquitetura justificada. A secção 3 do relatório definiu uma MLP densa com ReLU, *dropout* e paragem antecipada, com
saída linear na regressão e sigmoide na classificação, e uma grelha de três arquiteturas com uma, duas e três camadas
escondidas. Este documento descreve a implementação,
as decisões tomadas e o que foi verificado.

Código: `src/lending/deep.py` (`KerasMLPRegressor`, `KerasMLPClassifier`).
Testes: `tests/test_deep.py` (23 testes; `python -m pytest tests/test_deep.py` na raiz do repositório, ~1 minuto).
Bibliotecas: TensorFlow 2.21 e Keras 3.15 (só CPU: a partir da versão 2.11 o TensorFlow não usa a GPU no Windows nativo).

---

## 1. Porque é que a rede tem a interface do scikit-learn

Todos os modelos de cada tarefa passam pelo mesmo pipeline e pelo mesmo runner da validação cruzada:

```
Pipeline([("features", FeatureBuilder(tarefa)),      # features novas
          ("pre", build_preprocessor(colunas)),     # log, clipping, imputação, padronização, one-hot
          ("model", modelo)])                       # kNN, Ridge, árvore, RF, LightGBM ou MLP
```

O Keras tem a sua própria forma de criar e treinar redes (`Sequential`, `compile`, `fit`). Para a MLP entrar no
pipeline, as duas classes seguem as regras de um estimador do scikit-learn:

| Regra | Para que serve |
|---|---|
| Hiperparâmetros só no `__init__`, guardados sem alterações | `get_params`/`set_params` funcionam, incluindo `set_params(model__learning_rate=...)` dentro do pipeline |
| A rede Keras só é criada dentro do `fit` | `clone` copia só os hiperparâmetros. Cada configuração e cada *fold* começam com uma rede nova e nunca reaproveitam pesos de outro *fold* |
| Atributos aprendidos terminam em `_` (`model_`, `history_`, ...) | o scikit-learn sabe se o modelo já foi treinado e dá `NotFittedError` se não foi |
| `fit(X, y)`, `predict(X)` e, na classificação, `predict_proba(X)` | o runner trata a MLP como trata qualquer outro modelo |

Assim, o pré-processamento continua a ser ajustado só no treino de cada *fold*, e a MLP é avaliada nos mesmos *folds* e
com as mesmas métricas dos outros modelos. Herdar do `BaseEstimator` só dá o `get_params`/`set_params`: a rede é
definida por nós.

---

## 2. Arquitetura

```
entrada (97 a 123 colunas)
  -> Dense(n1, ReLU) -> Dropout(0,2)
  -> Dense(n2, ReLU) -> Dropout(0,2)        (consoante a arquitetura)
  -> Dense(n3, ReLU) -> Dropout(0,2)        (consoante a arquitetura)
  -> Dense(1, linear)    na regressão:      taxa de juro (padronizada, ver secção 4)
     Dense(1, sigmoide)  na classificação:  probabilidade de uma das classes
```

| Escolha | Justificação |
|---|---|
| Camadas densas | Os dados são tabulares: uma linha por empréstimo, sem estrutura espacial (imagens) nem sequencial (texto, séries). Redes convolucionais ou recorrentes não se aplicam, e não há modelos pré-treinados adequados para *transfer learning*. |
| ReLU nas camadas escondidas | `max(0, x)` não satura para valores positivos, por isso o gradiente não se apaga ao atravessar várias camadas (o gradiente evanescente das sigmoides). |
| *Dropout* 0,2 | Em cada passo do treino, 20% dos neurónios de cada camada são desligados ao acaso. A rede não pode depender de neurónios específicos, o que reduz o sobreajustamento. No `predict` o *dropout* está desligado. |
| Adam | Descida do gradiente com uma taxa de aprendizagem adaptada a cada peso; funciona bem sem muita afinação. |
| Saída linear + MSE (regressão) | O normal para um alvo contínuo. A perda de Huber está disponível (`loss="huber"`): é quadrática nos erros pequenos e linear nos grandes, por isso as taxas extremas pesam menos. |
| Saída sigmoide + entropia cruzada binária (classificação) | A sigmoide dá um valor entre 0 e 1, interpretado como probabilidade. A entropia cruzada penaliza muito uma probabilidade confiante e errada. |

**Tamanho das redes** (pesos + vieses). As colunas de entrada são as que saem do `build_preprocessor` em cada *fold*:
97 a 99 na regressão (6 *folds*) e 117 a 123 na classificação (3 *folds*). O número varia porque o one-hot aprende as
categorias no treino de cada *fold*.

| Arquitetura | Parâmetros (regressão, 97 colunas) | Parâmetros (classificação, 117–123 colunas) |
|---|---|---|
| (64) | 6 337 | 7 617 – 8 001 |
| (128, 64) | 20 865 | 23 425 – 24 193 |
| (256, 128, 64) | 66 305 | 71 425 – 72 961 |

Na regressão, mesmo a rede maior tem bem menos parâmetros do que empréstimos de treino (53 000 a 296 000 por *fold*). Na
classificação não é assim: no *fold* 1 há 2 991 empréstimos para até 73 000 parâmetros. O que trava o sobreajustamento
é o *dropout* e sobretudo a paragem antecipada, que pára o treino quando a validação interna deixa de melhorar (na
verificação da secção 8, a rede maior parou à época 11 nesse *fold*).

---

## 3. Hiperparâmetros

| Parâmetro | Omissão | Grelha | Notas |
|---|---|---|---|
| `hidden_layers` | (128, 64) | (64), (128, 64), (256, 128, 64) | Uma, duas e três camadas, como na grelha da secção 3 do relatório |
| `learning_rate` | 0,001 | 0,001 e 0,0003 | 0,001 é o valor de referência do Adam; 0,0003 é mais lento mas mais estável nas redes maiores |
| `dropout` | 0,2 | fixo | |
| `batch_size` | 512 | fixo | Empréstimos por atualização dos pesos |
| `max_epochs` | 50 | fixo | Orçamento da secção 3 |
| `patience` | 5 | fixo | Épocas sem melhoria antes de parar |
| `validation_fraction` | 0,15 | fixo | Últimas linhas do treino do *fold*, usadas na paragem antecipada |
| `random_state` | 42 | fixo | `config.RANDOM_SEED` |
| `loss` (regressão) | `"mse"` | — | `"huber"` como alternativa, se as taxas extremas pesarem demais |
| `class_weight` (classificação) | `"balanced"` | — | Ver secção 5 |

A grelha tem 3 × 2 = 6 configurações por tarefa: 36 ajustes na regressão (6 *folds*) e 18 na classificação (3 *folds*).

---

## 4. Paragem antecipada temporal e alvo padronizado

**Paragem antecipada.** A rede treina no máximo 50 épocas (uma época é uma passagem completa pelas linhas de treino). As
linhas do treino do *fold* são divididas pela ordem em que chegam:

```
treino do fold, ordenado por issue_d
|------------------------ 85% ------------------------|------ 15% ------|
  atualizam os pesos da rede (baralhadas em cada época)   validação interna
                                                          (os mais recentes)
```

No fim de cada época mede-se um critério na validação interna. Se não melhorar durante 5 épocas, o treino pára e ficam os
pesos da melhor época (`restore_best_weights`). A validação interna são as **últimas** linhas e não uma parte aleatória:
como os dados chegam ordenados por `issue_d`, são os empréstimos mais recentes do *fold*. Assim a paragem não usa
empréstimos posteriores para decidir quando parar de treinar com os anteriores. É a mesma lógica da janela expansível,
aplicada dentro de cada *fold*.

- **Requisito para quem chama:** os dados têm de vir ordenados por `issue_d`. Os ficheiros `regressao_dev.pkl` e
  `classificacao_treino.pkl` **não** estão ordenados; é preciso `sort_values("issue_d", kind="stable")` antes da
  validação cruzada.
- O corte usa o mesmo arredondamento do `validation_split` do Keras (`ceil(n × 0,85)` linhas de treino). A divisão é
  feita no código, e não pelo `validation_split`, para os pesos das classes poderem ser calculados só nas linhas que
  treinam a rede.
- Critério: na regressão, a perda (MSE ou Huber) da validação interna; na classificação, a AUC (secção 5).

**Alvo padronizado (regressão).** A rede não aprende diretamente a `int_rate` (cerca de 5 a 30 pontos percentuais), mas
`(int_rate − média) / desvio-padrão`. O `predict` desfaz a conta, por isso as previsões saem em pontos percentuais. Com o
alvo à volta de 0 e de escala 1, a saída linear começa perto do valor certo e o Adam não gasta as primeiras épocas só a
"subir" até aos 13 pontos. A média e o desvio são calculados dentro do `fit`, só com o treino do *fold* (como os scalers
do pré-processamento), e o MSE no alvo padronizado é o MSE original a dividir por uma constante, por isso a melhor época
é a mesma.

---

## 5. Classificação: convenções e desequilíbrio

**Convenções do scikit-learn** (as mesmas da regressão logística e da Random Forest):

- `classes_` são as duas classes por ordem: `["Default", "Good standing"]`. Também aceita rótulos 0/1.
- A saída sigmoide é a probabilidade de `classes_[1]`. O `predict_proba` devolve 2 colunas pela ordem de `classes_`.
- A probabilidade de *Default* obtém-se com `proba[:, list(modelo.classes_).index("Default")]`.
- O `predict` usa o limiar 0,5. O limiar final (o que maximiza o F1 macro nas previsões fora do *fold*) é escolhido
  **fora** do modelo, a partir do `predict_proba`, e fica fixo antes do teste.
- Com 3 classes (por exemplo, *At risk*) dá erro: o modelo é binário, como a classificação definida na limpeza.

**Desequilíbrio** (12,3% de *Default* no treino). Com `class_weight="balanced"`, cada classe tem o peso
`n / (2 × n_classe)` na perda, calculado **só nas linhas que atualizam os pesos** (sem a validação interna). Um erro num
*Default* pesa cerca de 7 vezes mais do que num *Good standing*, por isso a rede não aprende a prever sempre a classe
maioritária. Não se reamostra nada, e a validação interna, a validação do *fold* e o teste nunca são reequilibrados.
Também se pode passar um dicionário `{classe: peso}` ou `None`.

**Porque é que a paragem antecipada usa a AUC e não a perda.** O Keras aplica os pesos das classes à perda do treino,
mas não à perda da validação. Com pesos, as probabilidades de *Default* sobem; a perda da validação sem pesos castigaria
essa subida e pararia o treino pelo motivo errado. A AUC só depende da **ordem** das probabilidades (se os *Default*
ficam acima dos *Good standing*), por isso não é afetada pelos pesos, e é uma das métricas de escolha da classificação.

---

## 6. Reprodutibilidade

`keras.utils.set_random_seed(42)` no início de cada `fit` fixa as sementes do Python, do NumPy e do TensorFlow, ou seja,
os pesos iniciais, o *dropout* e a ordem dos lotes. Com a mesma semente e os mesmos dados, dois ajustes dão previsões
idênticas (testado). Noutra máquina, a ordem das somas em CPU (oneDNN) pode dar diferenças mínimas de arredondamento.

---

## 7. Custo computacional

Medido num portátil (CPU), no maior *fold* da regressão: 296 180 empréstimos de treino (2013T1–2014T3) e 74 098 de
validação (2014T4).

| Arquitetura | Taxa | Épocas (melhor) | Tempo por época | Treino | Previsão (74 098) |
|---|---|---|---|---|---|
| (64) | 0,001 | 19 (14) | 0,9 s | 16 s | 0,2 s |
| (64) | 0,0003 | 24 (19) | 0,8 s | 19 s | 0,2 s |
| (128, 64) | 0,001 | 26 (21) | 1,2 s | 32 s | 0,2 s |
| (128, 64) | 0,0003 | 40 (35) | 1,3 s | 51 s | 0,2 s |
| (256, 128, 64) | 0,001 | 21 (16) | 2,4 s | 51 s | 0,2 s |
| (256, 128, 64) | 0,0003 | 24 (19) | 2,4 s | 57 s | 0,2 s |

- Nenhuma configuração chegou às 50 épocas, por isso o limite da secção 3 chega.
- Os 6 *folds* somam cerca de 3,4 vezes as linhas do maior, por isso a grelha completa da regressão (36 ajustes) deve
  demorar cerca de 15 minutos. Na classificação (até 17 424 linhas por *fold*), cada ajuste demora 3 a 5 segundos.
- O custo do treino cresce com o número de linhas × parâmetros × épocas; a previsão é uma multiplicação de matrizes por
  camada e é praticamente instantânea.
- **Gravação:** o estimador treinado grava-se e lê-se com `joblib` (compressão 3: ~0,7 MB para a rede maior), com
  previsões iguais depois de lido.

---

## 8. Verificação com dados reais

Estes números servem só para confirmar que a MLP funciona dentro do pipeline completo. **Não são os resultados finais:**
não houve afinação, e a comparação entre modelos vem da validação cruzada completa.

**Regressão, *fold* 1** (treino 2013T1–T2, 53 350 empréstimos; validação 2013T3, 37 562):

| Modelo | MAE (p.p.) | RMSE (p.p.) |
|---|---|---|
| Mediana do treino | 3,72 | — |
| Ridge (alpha = 1) | 1,73 | 2,17 |
| MLP (64) | 1,66 | 2,10 |
| MLP (128, 64) | 1,70 | 2,14 |
| MLP (256, 128, 64) | 1,71 | 2,16 |

No *fold* 6 (validação 2014T4, tabela da secção 7) o MAE das MLP sobe para cerca de 2,7–2,8 p.p. Pode indicar uma
mudança no nível das taxas no fim de 2014, que a regressão não consegue acompanhar porque não pode usar a data. Fica
para a análise dos resíduos por trimestre.

**Classificação, 3 *folds* anuais** (validação 2009, 2010 e 2011). O F1 macro usa o limiar 0,5, porque o limiar final
ainda não foi escolhido.

| Modelo | AUC (média ± desvio) | F1 macro (média ± desvio) |
|---|---|---|
| Regressão logística, `class_weight="balanced"` | 0,664 ± 0,020 | 0,562 ± 0,007 |
| MLP (64) | 0,665 ± 0,025 | 0,556 ± 0,024 |
| MLP (128, 64) | 0,665 ± 0,028 | 0,555 ± 0,024 |
| MLP (256, 128, 64) | 0,666 ± 0,025 | 0,563 ± 0,017 |

A MLP fica ao nível da regressão logística. Prever o *default* só com a informação disponível na concessão é difícil, a
relação parece próxima de linear e o treino é pequeno (3 000 a 17 000 empréstimos), por isso uma rede tem pouco por onde
ganhar. O *fold* 1, com menos de 3 000 empréstimos de treino, é o pior e o mais variável.

---

## 9. Testes (`tests/test_deep.py`)

| Grupo | O que garante |
|---|---|
| Interface do scikit-learn | `get_params`, `clone` sem a rede treinada, uso dentro de um `Pipeline` com `set_params`, `NotFittedError` antes do `fit` |
| Regressão | aprende (MAE abaixo de metade do da mediana) e prevê nas unidades originais; `huber` aceite, outras perdas recusadas |
| Reprodutibilidade | mesma semente dá previsões idênticas; outra semente dá previsões diferentes |
| Paragem antecipada | a validação interna são as últimas linhas (estragar o alvo no fim faz a perda disparar, no início não); pára `patience` épocas depois da melhor e guarda essa |
| Classificação | ordem de `classes_`, `predict_proba` com 2 colunas que somam 1, `predict` com limiar 0,5, rótulos 0/1, recusa de 3 classes |
| Pesos das classes | `balanced` igual à fórmula do scikit-learn nas linhas de treino da rede; dicionário e `None`; com pesos há mais previsões de *Default* |
| Dados inválidos | valores em falta, número de colunas diferente e X/y de tamanhos diferentes dão erros claros |

Os testes usam dados sintéticos pequenos, em que se conhece a resposta certa, e redes pequenas, para correrem depressa.

---

## 10. Limitações

- **Ordem dos dados.** A paragem antecipada só é temporal se os dados chegarem ordenados por `issue_d`. A classe não
  recebe as datas e não o pode verificar.
- **Lotes no *fold* 1 da classificação.** Com 2 543 linhas de treino da rede e lotes de 512, cada época dá só 5
  atualizações dos pesos. A rede precisa de mais épocas, mas cabe nas 50 (parou à época 40 com a rede menor).
- **Calibração.** Com pesos das classes, as probabilidades da classificação ficam deslocadas para cima e não devem ser
  lidas como probabilidades reais de *default*. A AUC e a escolha do limiar não são afetadas.
- **Interpretação.** A rede não tem coeficientes; interpreta-se com a importância por permutação, como os outros modelos.
- **Determinismo entre máquinas.** Mesma semente, mesmos resultados na mesma máquina; noutra podem surgir diferenças
  mínimas de arredondamento.

---

## Teoria (slides)

| Tema | Slides |
|---|---|
| Redes neuronais | T04, p. 64–80 |
| ReLU, gradiente evanescente, *dropout*, paragem antecipada, Adam | T05, p. 3–18 |
| *Embeddings* e *transfer learning* (porque não se usam aqui) | T05, p. 32–38 e 42–43 |
| Matriz de confusão, F1, AUC | T03, p. 22–32 |
| MAE, RMSE, R² | T03, p. 33–36 |
| Validação cruzada e sobreajustamento | T03, p. 58–64 |
