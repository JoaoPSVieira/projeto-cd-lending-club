# projeto-cd-lending-club
Projeto de Ciência de Dados - 2026/2027


## 📂Estrutura do projeto
```text
ds-lending-club/
│
├── data/
│   ├── raw/                  # Dados originais (CSVs do Kaggle)
│   └── processed/            # Dados processados (Parquet)
│
├── src/
│   └── lending/              # Pacote Python
│       └── __init__.py
│
├── notebooks/                # Jupyter Notebooks (.ipynb)
│
├── R/                        # Análises e código em R
│
├── reports/
│   └── figures/              # Gráficos exportados para o relatório LaTeX
│
├── models/                   # Modelos treinados (Parte 2)
│
├── docs/                     # Documentação, enunciado e notas
│
├── .gitignore                # Ficheiros e diretórios ignorados pelo Git
├── README.md                 # Documentação do projeto
└── requirements.txt          # Dependências Python
```

## Tarefas
- Regressão: prever `int_rate` com informação pré-pricing
- Classificação: desfecho do empréstimo em 3 classes (Good standing / At risk / Default)
- Clustering: mutuários e estados

## Setup
```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dados
Fazer download de https://www.kaggle.com/datasets/wordsforthewise/lending-club e colocar os CSVs em `data/raw/`.

## Grupo
- 2103823 - João Pedro Sá Vieira
- Soares