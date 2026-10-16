"""
Caminhos do projeto e constantes partilhadas

Motivo:
    Os caminhos são calculados a partir da localização deste ficheiro, e não da pasta
    onde o notebook está a correr. Assim o código funciona igual em qualquer computador.

    Os nomes de ficheiros e constantes ficam num único sítio, em vez de espalhados
    pelas classes e notebooks
"""

from pathlib import Path

# ----------
# - Pastas -
# ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # csv originais do Kaggle (não mexidos)
PROCESSED_DIR = DATA_DIR / "processed"  # ficheiros gerados por nós (pickles)
DOCS_DIR = PROJECT_ROOT / "docs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------
# - Ficheiros de origem e as cópias em pickle -
# ---------------------------------------------
ACCEPTED_CSV = RAW_DIR / "accepted_2007_to_2018Q4.csv.gz"
REJECTED_CSV = RAW_DIR / "rejected_2007_to_2018Q4.csv.gz"

ACCEPTED_RAW_PKL = PROCESSED_DIR / "accepted_raw.pkl"
REJECTED_RAW_PKL = PROCESSED_DIR / "rejected_raw.pkl"

# Partições do accepted, ainda em texto (só linhas separadas, nenhum valor alterado)
ACCEPTED_DEV_RAW_PKL = PROCESSED_DIR / "accepted_dev_raw.pkl"
ACCEPTED_TEST_RAW_PKL = PROCESSED_DIR / "accepted_test_raw.pkl"

# Registo das linhas removidas/separadas até chegar ao conjunto de desenvolvimento
ACCEPTED_DEV_LOG_PKL = PROCESSED_DIR / "accepted_dev_exclusions.pkl"

# Desenvolvimento limpo (tipos convertidos, correções e exclusões da limpeza) e os
# registos de exclusões de cada tarefa
ACCEPTED_DEV_CLEAN_PKL = PROCESSED_DIR / "accepted_dev_clean.pkl"
COHORT_LOGS_PKL = PROCESSED_DIR / "accepted_dev_cohort_logs.pkl"

# -----------------------------------------------------------------------
# - Partições temporais (docs/01_formulacao_problema/desenho_split.md) -
# -----------------------------------------------------------------------
# Meses "AAAA-MM", com início e fim incluídos. Os cortes são feitos ao mês,
# por isso os empréstimos do mesmo mês ficam sempre na mesma partição.
PARTITIONS = {
    "development": ("2007-06", "2014-12"),  # treino, validação, EDA, testes de hipóteses
    "test": ("2015-01", "2015-12"),         # teste final: usado uma única vez, no fim
    "reserved": ("2016-01", "2018-12"),     # não usado na Parte 1
}

# Partições da classificação (desenho_split.md, secção 4): só entram no treino os
# empréstimos de 36 meses cujo prazo terminou antes da data de treino simulada (2015-01).
# Os de 2012-2014 ainda não tinham terminado: ficam num embargo, fora da classificação.
CLASSIFICATION_PARTITIONS = {
    "train": ("2007-06", "2011-12"),
    "embargo": ("2012-01", "2014-12"),
    "test": ("2015-01", "2015-12"),
}

# Mês em que o ficheiro foi extraído: último pagamento registado (last_pymnt_d)
# (docs/02_preprocessamento/descricao_dados.md, secção 4)
EXTRACTION_MONTH = "2019-03"

# ------------
# Constantes -
# ------------
# Número de colunas de cada ficheiro - Para verificação se a leitura der outro número ou algo correu mal.
EXPECTED_COLUMNS = {
    "accepted": 151,
    "rejected": 9,
}

# Semente única para tudo o que é aleatório (Assim é possível replicar os resultados por usar a mesma seed)
RANDOM_SEED = 42
