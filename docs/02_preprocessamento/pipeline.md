# Pipeline de pré-processamento

Última parte do pré-processamento. A limpeza (`01_preprocessamento`, `01b_rejeitados`) só
usou regras fixas. Aqui ficam as **transformações aprendidas** (imputação, *clipping*,
padronização, *encoding*), que o enunciado manda ajustar só no treino:

> *"Split before learning transformations. Fit imputers, scalers, encoders, feature
> selection, PCA, clipping/winsorization (...) only on the training portion of each fold.
> Use pipelines in both Python and R."*

Código: `src/lending/features.py` (`load_eligibility`, `TaskFeatures`, `forbidden_columns`),
`src/lending/preprocessing.py` (`build_preprocessor`, `DropEmptyColumns`, `QuantileClipper`,
`ExpandingWindowSplit`). Notebook: `notebooks/01c_pipeline.ipynb`.

---

## 1. Colunas de cada tarefa

O código lê diretamente a tabela de elegibilidade da Fase 1
(`docs/01_formulacao_problema/elegibilidade_colunas.md`). Se a tabela mudar, o pipeline
muda com ela; não há duas versões que possam divergir.

Entram no pipeline base as colunas ✅ de cada tarefa, menos:

| Ficam de fora | Colunas | Porquê |
|---|---|---|
| ⚠️ condicionais | `zip_code`, `addr_state`, `issue_d`; `earliest_cr_line` na regressão | Só depois de a condição ser analisada (ex.: geografia e *fairness*) |
| Texto livre | `emp_title`, `title` | Dezenas de milhares de valores; só como *features* novas |
| Datas | `earliest_cr_line` (classificação, clustering) | Só transformadas (ex.: antiguidade do crédito), como *features* novas |

| Tarefa | Montantes em USD | Outras numéricas | Categorias | Total |
|---|---:|---:|---:|---:|
| Regressão | 17 | 70 | 3 | 90 |
| Classificação | 19 | 71 | 8 | 98 |
| Clustering | 15 | 57 | 1 | 73 |

Categorias: regressão `home_ownership`, `purpose`, `application_type`; classificação também
`grade`, `sub_grade`, `verification_status`, `initial_list_status`,
`verification_status_joint`.

---

## 2. Verificação contra *leakage*

`features.py` tem as listas do enunciado (*"Note on data leakage"*):
- proibidas em todas as tarefas: `loan_status`, os pagamentos e recuperações, as datas e o
  FICO de depois da concessão, `debt_settlement_flag*` e os prefixos `hardship_` e
  `settlement_`;
- proibidas só na regressão: `int_rate` (alvo), `grade`, `sub_grade`, `installment`,
  `funded_amnt`, `funded_amnt_inv`, `issue_d`.

O `TaskFeatures` **dá erro** se alguma coluna proibida aparecer entre as de uma tarefa. A
tabela de elegibilidade já as exclui; a verificação protege contra um erro na tabela.

---

## 3. Validação temporal

`ExpandingWindowSplit`: cada *fold* treina com todos os períodos anteriores e valida no
período seguinte (janela expansível). Nunca se treina com o futuro de quem se valida, e os
empréstimos do mesmo período ficam sempre juntos.

| Tarefa | Período | *Fold* | Treino | Validação | Linhas treino | Linhas validação |
|---|---|---:|---|---|---:|---:|
| Regressão | trimestre | 1 | 2013T1–T2 | 2013T3 | 53 350 | 37 562 |
| | | 2 | 2013T1–T3 | 2013T4 | 90 912 | 43 855 |
| | | 3 | 2013 | 2014T1 | 134 767 | 47 388 |
| | | 4 | 2013–2014T1 | 2014T2 | 182 155 | 55 332 |
| | | 5 | 2013–2014T2 | 2014T3 | 237 487 | 58 693 |
| | | 6 | 2013–2014T3 | 2014T4 | 296 180 | 74 098 |
| Classificação | ano | 1 | 2007–2008 | 2009 | 2 991 | 5 280 |
| | | 2 | 2007–2009 | 2010 | 8 271 | 9 153 |
| | | 3 | 2007–2010 | 2011 | 17 424 | 14 098 |

O teste final (2015) não entra em nenhum *fold*.

---

## 4. Passos do pipeline

`build_preprocessor` constrói um `ColumnTransformer` com três ramos. O que é aprendido no
`fit` (só com o treino) está entre parênteses.

| Ramo | Passos |
|---|---|
| Montantes em USD | colunas vazias saem (quais) → log(1 + x) → *clipping* aos percentis 0,5% e 99,5% (limites) → mediana + indicador de "em falta" (medianas) → padronização (média, desvio) |
| Outras numéricas | colunas vazias saem (quais) → *clipping* (limites) → mediana + indicador (medianas) → padronização (média, desvio) |
| Categorias | colunas vazias saem (quais) → "em falta" como categoria → *one-hot*, com as categorias abaixo de 0,5% do treino juntas (categorias) |

| Passo | Porquê | Alternativa rejeitada |
|---|---|---|
| Retirar colunas vazias no treino | Não há nada para aprender (ex.: 37 colunas do *bureau* no treino da classificação) | Imputar: inventaria valores para anos inteiros |
| log(1 + x) nos montantes | Rendimentos e saldos muito assimétricos | Deixar a escala original: os extremos dominariam a padronização |
| *Clipping* 0,5%–99,5% | Reduz o peso dos extremos que a limpeza deixou (ex.: `revol_util` 892%) sem retirar linhas | Remover outliers: perderia empréstimos reais |
| Mediana + indicador | O enunciado pede para distinguir "não aplicável" de "em falta" e não pôr tudo a 0. O indicador regista **que** faltava (ex.: `mths_since_last_delinq` em falta = nunca houve atraso) | Pôr 0: "0 meses desde o último atraso" diria o contrário |
| "Em falta" como categoria | Mantém a informação de que faltava | Moda: esconderia o valor em falta |
| *One-hot* com categorias raras juntas | `sub_grade` tem 35 níveis; categorias com poucos casos não se aprendem | *Ordinal encoding*: imporia uma ordem que `purpose` não tem |
| Padronização | Mesma escala para todas as variáveis (kNN, redes neuronais) | Min-max: mais sensível aos extremos |

---

## 5. Prova de que o *fit* só usa o treino

No notebook, com o *fold* 1 da regressão (treino 2013T1–T2, validação 2013T3):

| Verificação | Resultado |
|---|---|
| Mediana aprendida do `dti` | 16,98 = mediana do treino (a da validação é 16,57) |
| Média depois da padronização | 0 no treino, −0,05 na validação |
| Parâmetros depois de aplicar à validação | Iguais (o `transform` não aprende) |
| Ajustar de novo só com o treino | O mesmo resultado na validação |
| Valores em falta na saída | 0, em todos os 9 *folds* |

Na classificação (*fold* 1), o `DropEmptyColumns` retira 66 colunas vazias no treino: as 37
do *bureau* registadas só a partir de 2012, as dos pedidos conjuntos e as dos agregados
novos.

---

## 6. Como se usa na Parte 2

```python
model = Pipeline([("pre", build_preprocessor(TaskFeatures("regression"))),
                  ("model", ...)])
cross_validate(model, X, y, cv=list(ExpandingWindowSplit("Q", 2).split(dates)))
```

O pré-processamento fica dentro do modelo e é ajustado de novo em cada *fold*, só com o
treino desse *fold*. No fim, ajusta-se com todo o treino e aplica-se (só `transform`) ao
teste final. As *features* novas (≥ 10) são acrescentadas antes deste pipeline.
