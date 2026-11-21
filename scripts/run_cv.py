"""
Corre a validação cruzada completa de uma tarefa a partir do terminal

Os notebooks 04b e 04c leem os resultados; as grelhas, que podem demorar horas (sobretudo a
Random Forest e a MLP), correm melhor aqui do que dentro de um notebook. O runner retoma onde
parou: pode-se interromper (Ctrl+C) e voltar a correr o mesmo comando.

Uso, a partir da pasta do projeto:
    python scripts/run_cv.py regression
    python scripts/run_cv.py classification
    python scripts/run_cv.py regression --models "Ridge,LightGBM"
    python scripts/run_cv.py regression --exclude "Random Forest,MLP"

Resultados:
    reports/results/<tarefa>_cv.csv           (entra no Git)
    data/processed/oof/<tarefa>/...           (previsões fora do fold, fora do Git)
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    import lending  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from lending import config
from lending.experiment import CrossValidationRunner
from lending.model_catalog import classification_specs, regression_specs
from lending.storage import PickleStore

TASKS = {"regression": (config.REG_DEV_PKL, regression_specs),
         "classification": (config.CLS_TRAIN_PKL, classification_specs)}


def _names(text):
    return [name.strip() for name in text.split(",") if name.strip()]


def select_specs(specs, models=None, exclude=None):
    """Filtra o catálogo pelos nomes dos modelos (erro se um nome não existir)."""
    names = [s.name for s in specs]
    for asked in _names(models or "") + _names(exclude or ""):
        if asked not in names:
            raise SystemExit(f"Modelo desconhecido: {asked!r}. Modelos do catálogo: {names}")
    if models:
        specs = [s for s in specs if s.name in _names(models)]
    if exclude:
        specs = [s for s in specs if s.name not in _names(exclude)]
    return specs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validação cruzada temporal de uma tarefa.")
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--models", help='Só estes modelos, separados por vírgulas (ex.: "Ridge,MLP")')
    parser.add_argument("--exclude", help="Todos menos estes modelos, separados por vírgulas")
    args = parser.parse_args(argv)

    path, catalog = TASKS[args.task]
    specs = select_specs(catalog(), args.models, args.exclude)
    data = PickleStore.load(path, expected_type=pd.DataFrame)
    runner = CrossValidationRunner(args.task)

    n_folds = len(runner.folds(data))
    planned = {(s.name, cid, f) for s in specs for cid, _ in s.configs() for f in range(1, n_folds + 1)}
    done = {(r.model, r.config_id, r.fold) for r in runner.results().itertuples()}
    print(f"{args.task}: {len(data):,} linhas, {n_folds} folds, {len(specs)} modelos | "
          f"{len(planned & done)} de {len(planned)} ajustes já feitos")

    start = time.perf_counter()
    runner.run(data, specs)
    print(f"\nTerminado em {(time.perf_counter() - start) / 60:.1f} min. "
          f"Resultados: {runner.results_path.relative_to(PROJECT_ROOT)}")
    print(runner.comparison_table().to_string())


if __name__ == "__main__":
    main()
