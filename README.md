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
├── R/                        # Testes de hipóteses em R (R Markdown)
│
├── reports/
│   ├── figures/              # Gráficos exportados para o relatório LaTeX
│   └── relatorio/            # Fonte LaTeX do relatório
│
├── models/                   # Modelos treinados (Parte 2)
│
├── docs/                     # Documentação de cada fase e dicionário de dados
│
├── .gitignore                # Ficheiros e diretórios ignorados pelo Git
├── README.md                 # Documentação do projeto
└── requirements.txt          # Dependências Python
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

9. `04a_knn.ipynb`: kNN de raiz (estrutura de dados e verificação)
10. `04c_classificacao.ipynb`: classificação: os 8 modelos na validação cruzada, comparação, interpretação e modelo final (cerca de 25 minutos; ver `docs/04_modelos/classificacao.md`)
11. `04d_clustering.ipynb`: clustering de mutuários: k-means, GMM e DBSCAN, escolha de k, estabilidade e risco dos quatro grupos (cerca de 40 segundos com os resultados guardados, 25 minutos do zero; ver `docs/04_modelos/clustering.md`)

## Entrega da Parte 1

- Relatório: `reports/relatorio_parte1.pdf` (`reports/relatorio/`)
- Notebook de exploração e pré-processamento: os notebooks 00 a 03, pela ordem acima
- Trabalho em R: `R/testes_hipoteses.Rmd`, `R/pipeline_pca.Rmd` e os HTML gerados em `reports/`
- Documentação: `docs/` (uma pasta por fase) e as docstrings do pacote `src/lending`

## Grupo

- 2103823 - João Pedro Sá Vieira
- 2075526 - Rodrigo Sousa Soares