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

# ---------------------------------------------------------------------------
# Partições (Etapa 2: limpeza + split) Dados já limpos e com tipos corretos
# ---------------------------------------------------------------------------

REG_DEV_PKL = PROCESSED_DIR / "regressao_dev.pkl"          # regressão: 2013-01 a 2014-12
CLS_TRAIN_PKL = PROCESSED_DIR / "classificacao_treino.pkl" # classificação: 36m, 2007-06 a 2011-12
ACCEPTED_DEV_PKL = PROCESSED_DIR / "aceites_dev.pkl"       # aceites: 2007-06 a 2014-12 (testes de hipóteses)
REJECTED_DEV_PKL = PROCESSED_DIR / "rejeitados_dev.pkl"    # rejeitados: 2007 a 2014
TEST_PKL = PROCESSED_DIR / "teste_2015.pkl"                # teste final

# Coluna do alvo da classificação e ordem fixa das classes. Usar sempre esta ordem em tabelas e figuras
CLASS_COL = "target"
CLASS_ORDER = ["Good standing", "Default"]

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
