# Avaliação final no teste de 2015

Bloco 8 do plano da Parte 2: congelar os modelos das duas tarefas e avaliá-los uma única vez no
teste de 2015, com intervalos de confiança, a comparação com a validação cruzada e a evidência
para a discussão (mudança temporal, *follow-up*, viés de seleção e equidade).

| Ficheiro | Conteúdo |
|---|---|
| `scripts/freeze_models.py` | Congela uma tarefa: escolhas, limiares e pipelines treinados em todo o treino |
| `src/lending/final_evaluation.py` | Congelamento, coortes de teste, `FinalEvaluation` (métricas, bootstrap, confusão, grupos, deriva, gráficos) |
| `notebooks/05_avaliacao_final.ipynb` | O único notebook que usa o teste |
| `tests/test_final_evaluation.py` | Testes do congelamento, do bootstrap e da avaliação |

---

## 1. Ordem dos passos

1. **Pré-requisitos.** O bloco 4 no `main` (grelha completa da regressão em
   `reports/results/regression_cv.csv`) e os CSV do 04c (`classification_comparison.csv` e
   `classification_final_model.csv`).
2. **Congelar as duas tarefas** (a mesma máquina serve para as duas):
   ```
   python scripts/freeze_models.py regression
   python scripts/freeze_models.py classification
   ```
3. **Commit e tag** antes de abrir o teste:
   ```
   git add reports/results/regression_frozen.json reports/results/classification_frozen.json
   git commit -m "Modelos congelados antes do teste de 2015"
   git tag modelos-congelados
   git push && git push --tags
   ```
4. **Correr o notebook 05** do início ao fim, uma vez.
5. **Commit** dos CSV `reports/results/*_test*.csv` e `test_drift.csv`, das figuras
   `reports/figures/final_*.png` e do notebook com os outputs.

O que garante que o teste só é usado uma vez:

- O congelamento não lê o teste. O notebook 05 não treina nem escolhe nada.
- O notebook para se o JSON de uma tarefa não for igual ao da tag (`tag_status`).
- Cada pipeline é lido só depois de confirmar o hash SHA-256 gravado no JSON.
- As previsões do teste ficam em `data/processed/test_predictions/<tarefa>.pkl`, identificadas
  pelas escolhas congeladas. Correr o notebook de novo lê-as do ficheiro; se as escolhas
  mudarem, dá erro. O script de congelamento também recusa congelar de novo uma tarefa cujo
  teste já foi previsto (só com `--force`, e isso tem de ser explicado no relatório).

---

## 2. O que é congelado

Para cada tarefa, `reports/results/<tarefa>_frozen.json` (no Git) e `models/<tarefa>/*.joblib`
(fora do Git: vão no zip ou no OneDrive da entrega).

| Campo | Conteúdo |
|---|---|
| `models` | Uma entrada por modelo: configuração (`config_id`, `params`), pré-processamento, colunas tiradas, amostra máxima, limiar (classificação), métricas da validação cruzada em "média ± desvio", ficheiro, hash, linhas de treino, tempo de treino |
| `final_model` | O modelo que segue para o deployment |
| `train` | Linhas e período do treino |
| `git_commit`, `uncommitted_code_changes`, `versions` | Commit do código que treinou, se havia código por gravar, versões das bibliotecas |

**Que modelos vão ao teste.** A melhor configuração de cada modelo na validação cruzada,
*baselines* incluídas: é a tabela de comparação do enunciado, agora com as colunas do teste. Na
classificação vai também o modelo final do 04c (regressão logística sem `issue_year`), ao lado da
logística com `issue_year`.

**Fontes das escolhas.**

- Regressão: o `regression_cv.csv`, pelo `runner.comparison_table()` (MAE médio). O script recusa
  congelar se a grelha não estiver completa.
- Classificação: os CSV do 04c. O limiar de cada modelo foi escolhido nas previsões fora do fold,
  que só existem na máquina onde a grelha correu; os CSV já o têm, por isso qualquer máquina pode
  congelar. O script confirma que o `classification_comparison.csv` corresponde ao
  `classification_cv.csv`: todas as configurações completas e a mesma AUC média.

**O modelo final.**

- Regressão: por omissão, o melhor MAE médio na validação cruzada, que é a regra fixada antes
  da grelha. Com `--final "<modelo>"` escolhe-se outro, e a razão tem de ficar no relatório.
- Classificação: o modelo final do 04c, com o seu limiar (`classification_final_model.csv`).

**Treino.** Cada pipeline é treinado com `runner.refit` em todo o treino da tarefa, com as mesmas
regras da validação cruzada: dados ordenados por data, pré-processamento ajustado no treino, a
mesma amostra do kNN (50 000 na regressão) e as mesmas colunas tiradas. A paragem antecipada da
MLP e do LightGBM usa os 15% de empréstimos mais recentes do treino. Se o script for
interrompido, o mesmo comando continua (`--refit` treina tudo de novo).

---

## 3. Decisões da avaliação

- **Limiar.** Cada modelo de classificação usa o limiar congelado. O limiar nunca é afinado no
  teste, mesmo que a taxa de *Default* de 2015 seja diferente da do treino.
- **Bootstrap.** 1 000 reamostragens dos empréstimos de teste com reposição, intervalo pelos
  percentis 2,5% e 97,5% (T03). As reamostragens são as mesmas do `evaluation.bootstrap_ci`, com
  a mesma semente, mas calculadas como pesos (quantas vezes sai cada linha), o que é muito mais
  rápido em centenas de milhares de linhas. Os testes confirmam que os intervalos são iguais. Todos
  os modelos usam as mesmas reamostragens, por isso a diferença entre dois modelos em cada uma é o
  bootstrap emparelhado.
- **Comparação emparelhada.** O modelo final contra cada um dos outros: MAE na regressão; F1
  macro e AUC na classificação. Uma diferença é clara quando o intervalo de 95% não contém o 0.
- **Matrizes de confusão.** A 2 × 2 da classificação binária e a 3 × 2, com as linhas nas 3
  classes do enunciado (Good standing, At risk, Default), para mostrar onde caem os *At risk* que
  não existiam no treino.
- **Mudança temporal.**
  - O erro por mês de emissão em 2015 (MAE e viés na regressão; taxa real, % prevista e P(Default)
    média na classificação).
  - O PSI de cada variável principal entre o treino e o teste. Convenção habitual em risco de
    crédito: abaixo de 0,1 é estável, de 0,1 a 0,25 é moderado e acima de 0,25 é grande.
- **Equidade.** O erro do modelo final por região dos EUA (a partir de `addr_state`, que não entra
  nos modelos), por tipo de habitação e por quintil de rendimento (com os limites do treino).
  - Regressão: MAE e viés.
  - Classificação: taxa real de *Default*, *recall* de *Default* e falsos alarmes.
  - Grupos com menos de 1 000 empréstimos juntam-se em "outros".
- **Resultado mau no teste:** não se muda o modelo; reporta-se e discute-se.

---

## 4. Resultados que o notebook grava

| Ficheiro | Conteúdo |
|---|---|
| `regression_test.csv`, `classification_test.csv` | Tabela de comparação: validação cruzada, teste e IC 95% |
| `classification_test_all_metrics.csv` | Todas as métricas da classificação no teste (macro e micro) |
| `*_test_paired.csv` | Bootstrap emparelhado do modelo final contra os outros |
| `classification_test_confusion.csv` | Matrizes 2 × 2 e 3 × 2, em contagens e em % |
| `*_test_subgroups.csv` | Equidade por região, habitação e rendimento |
| `test_drift.csv` | Deriva das variáveis (PSI) nas duas tarefas |
| `figures/final_reg_*.png`, `figures/final_cls_*.png` | Validação cruzada contra teste, erro por mês, equidade |

---

## 5. Problemas possíveis

- **"Sem a tag modelos-congelados certa".**
  - "sem tag": faltam o commit ou a tag (passo 3).
  - "diferente": o JSON mudou depois da tag. Não se apaga a tag para a refazer sem o dizer no
    relatório.
- **"não é o ficheiro congelado (o hash mudou)".** O ficheiro em `models/` não é o que foi
  congelado (por exemplo, o modelo foi treinado de novo depois da tag).
- **Ficheiros grandes.** A Random Forest da regressão (300 árvores, folhas de 10, 370 mil
  empréstimos) pode ocupar centenas de MB. O script mostra o tamanho de cada ficheiro; vão para o
  OneDrive, não para o Git.
- **Memória do kNN.** A previsão do kNN no teste guarda os 101 vizinhos de cada empréstimo
  (centenas de MB). A cache é limpa logo a seguir a cada modelo.

---

## 6. Ligação aos outros blocos

- **Bloco 9 (deployment).** Os pipelines congelados estão em `models/<tarefa>/`. Cada ficheiro
  guarda um dicionário com `pipeline`, `config_id`, `model`, `task` e `fit_seconds`. O JSON diz
  qual é o modelo final e o seu limiar. Os números do teste referem-se a estes pipelines; se o
  deployment os treinar de novo com 2015 incluído, isso tem de ficar dito no plano.
- **Bloco 11 (sumário executivo).** A secção 6 do notebook imprime os números em linguagem
  simples: o erro médio da taxa em p.p. e quantos *defaults* o modelo apanha em cada 100.
