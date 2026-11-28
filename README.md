# projeto-cd-lending-club

Projeto de Ciência de Dados - 2026/2027

## 📂Estrutura do projeto

```text
projeto-cd-lending-club/
│
├── data/                     # Não está no Git (gerado localmente)
│   ├── raw/                  # Dados originais (CSVs do Kaggle)
│   └── processed/            # Dados processados (pickle)
│
├── src/
│   └── lending/              # Pacote Python: classes usadas pelos notebooks
│
├── notebooks/                # Jupyter Notebooks (.ipynb)
│
├── scripts/                  # Grelha da validação cruzada e congelamento dos modelos (Parte 2)
│
├── tests/                    # Testes do pacote (pytest)
│
├── R/                        # Testes de hipóteses em R (R Markdown)
│
├── reports/
│   ├── figures/              # Gráficos exportados para o relatório LaTeX
│   ├── results/              # Resultados de cada experiência (CSV) e modelos congelados (JSON)
│   └── relatorio/            # Fonte LaTeX do relatório
│
├── models/                   # Modelos treinados (Parte 2); não está no Git, ver "Modelos treinados"
│
├── docs/                     # Documentação de cada fase e dicionário de dados
│
├── .gitignore                # Ficheiros e diretórios ignorados pelo Git
├── pyproject.toml            # Pacote lending e configuração do pytest
├── README.md                 # Documentação do projeto
└── requirements.txt          # Dependências Python (versões fixas)
```

## Tarefas

- Regressão: prever `int_rate` com informação pré-pricing
- Classificação: desfecho do empréstimo em 3 classes (Good standing / At risk / Default); na prática binária, porque a classe At risk não existe no treino (ver relatório, secção II-F)
- Clustering: mutuários e estados

## Setup

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

R (testes de hipóteses): R ≥ 4.1 e pandoc (incluído no RStudio).

```r
  install.packages(c("rmarkdown", "knitr", "data.table", "boot", "recipes"))
```

Testes do pacote (não precisam dos dados):

```text
pytest -q
```

## Dados

Fazer download de https://www.kaggle.com/datasets/wordsforthewise/lending-club e colocar em `data/raw/` os ficheiros `accepted_2007_to_2018Q4.csv.gz` e `rejected_2007_to_2018Q4.csv.gz`, sem descomprimir.

## Como correr

Cada notebook lê os ficheiros gerados pelos anteriores, por isso a ordem importa:

1. `00_conversao_dados.ipynb`: CSV -> pickle
2. `01_preprocessamento.ipynb`: limpeza, partição temporal, coortes e teste final
3. `01b_rejeitados.ipynb`: limpeza dos pedidos rejeitados
4. `01c_pipeline.ipynb`: pipeline de pré-processamento e validação temporal
5. `02_eda.ipynb`: análise exploratória, PCA e UMAP
6. `02b_dados_testes_hipoteses.ipynb`: exporta para CSV os dados dos testes em R
7a. `R/testes_hipoteses.Rmd`: testes de hipóteses; gera `reports/testes_hipoteses.html`
   ```text
   Rscript -e "rmarkdown::render('R/testes_hipoteses.Rmd', output_dir = 'reports')"
   ```
  7b. `R/pipeline_pca.Rmd`: o mesmo pipeline em R (recipes), com verificação cruzada do PCA
8. `03_features.ipynb`: features novas, verificadas no pipeline

Parte 2 (modelação):

9. `04a_knn.ipynb`: kNN de raiz (estrutura de dados e verificação; ver `docs/04_modelos/knn.md`)
10. Grelha da regressão, no terminal, e depois `04b_regressao.ipynb`: os 8 modelos na validação cruzada, comparação, custo e interpretação (ver `docs/04_modelos/regressao.md`)
    ```text
    python scripts/run_cv.py regression
    ```
    A grelha tem 252 ajustes (cerca de 35 minutos de treino). Se for interrompida, o mesmo comando continua onde parou; `--models` e `--exclude` correm só parte dos modelos. O notebook lê os resultados e não volta a treinar a grelha.
11. `04c_classificacao.ipynb`: classificação: os 8 modelos na validação cruzada, comparação, interpretação e modelo final (cerca de 25 minutos; ver `docs/04_modelos/classificacao.md`)
12. `04d_clustering.ipynb`: clustering de mutuários: k-means, GMM e DBSCAN, escolha de k, estabilidade e risco dos quatro grupos (gravado com `RECALCULAR = True`: calcula tudo do zero em cerca de 25 minutos; com `RECALCULAR = False` reaproveita os resultados guardados e corre em cerca de 40 segundos; ver `docs/04_modelos/clustering.md`)
13. Congelar os modelos antes do teste: grava a configuração e o limiar de cada modelo em `reports/results/<tarefa>_frozen.json` e os modelos treinados em todo o treino em `models/` (cerca de 4 minutos)
    ```text
    python scripts/freeze_models.py regression
    python scripts/freeze_models.py classification
    ```
    Os dois JSON estão no Git com a tag `modelos-congelados`, feita antes de abrir o teste.
14. `05_avaliacao_final.ipynb`: avaliação final no teste de 2015, corrida uma única vez depois da tag: comparação com a validação cruzada e intervalos de confiança por bootstrap, matrizes de confusão, erro ao longo de 2015, equidade e deriva das variáveis. Não treina nem escolhe nada; voltar a correr lê as previsões guardadas (ver `docs/05_avaliacao/avaliacao_final.md`)

## Modelos treinados

A pasta `models/` (cerca de 120 MB) não está no Git: está no OneDrive, em LINK_DO_ONEDRIVE. Para a usar, extrair para `models/` na raiz do projeto. O notebook 05 confirma, antes de ler cada modelo, que o ficheiro é o congelado (hash SHA-256 em `reports/results/<tarefa>_frozen.json`).

## Entrega da Parte 1

- Relatório: `reports/relatorio_parte1.pdf` (`reports/relatorio/`)
- Notebook de exploração e pré-processamento: os notebooks 00 a 03, pela ordem acima
- Trabalho em R: `R/testes_hipoteses.Rmd`, `R/pipeline_pca.Rmd` e os HTML gerados em `reports/`
- Documentação: `docs/` (uma pasta por fase) e as docstrings do pacote `src/lending`

## Entrega da Parte 2

- Relatório final: em preparação
- Sumário executivo: em preparação
- Notebooks de modelação: 04a a 05, pela ordem acima
- Código: o pacote `src/lending`, os scripts em `scripts/` e os testes em `tests/`
- Resultados: `reports/results/` (CSV de cada experiência e modelos congelados) e `reports/figures/`
- Modelos treinados: OneDrive (ver "Modelos treinados")
- Documentação: `docs/04_modelos/` e `docs/05_avaliacao/`

## Grupo

- 2103823 - João Pedro Sá Vieira
- 2075526 - Rodrigo Sousa Soares