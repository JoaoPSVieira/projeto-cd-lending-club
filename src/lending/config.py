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
