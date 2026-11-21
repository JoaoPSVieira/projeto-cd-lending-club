"""
Congela os modelos de uma tarefa antes do teste de 2015 (bloco 8)

Para cada modelo da tabela de comparação da validação cruzada (a melhor configuração de cada
um, baselines incluídas), grava a configuração e o limiar, treina o pipeline em TODO o treino
e grava-o. Não lê o teste. O JSON entra no Git; a seguir faz-se a tag modelos-congelados e só
então se corre o notebook 05.

Uso, a partir da pasta do projeto:
    python scripts/freeze_models.py regression
    python scripts/freeze_models.py regression --final "MLP"
    python scripts/freeze_models.py classification

Fontes das escolhas:
    regressão       reports/results/regression_cv.csv (grelha completa, bloco 4)
    classificação   reports/results/classification_comparison.csv e
                    classification_final_model.csv (notebook 04c, com os limiares)

Resultados:
    reports/results/<tarefa>_frozen.json      (entra no Git)
    models/<tarefa>/<modelo>.joblib           (fora do Git: zip ou OneDrive na entrega)

Se for interrompido, o mesmo comando continua: os modelos já gravados com a mesma
configuração não se treinam de novo (--refit treina todos outra vez).
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

from lending.experiment import CrossValidationRunner
from lending.final_evaluation import (TAG, TEST_PREDICTIONS_DIR, TRAIN_PKL, cv_table, fit_frozen,
                                      frozen_models, frozen_path, tag_status, task_catalog,
                                      write_frozen)
from lending.storage import PickleStore


def check_grid(runner, catalog, data):
    """Na regressão, a grelha tem de estar completa: escolher com uma grelha a meio é errado."""
    n_folds = len(runner.folds(data))
    planned = {(name, cid, f) for name, spec in catalog.items() for cid, _ in spec.configs()
               for f in range(1, n_folds + 1)}
    done = {(r.model, r.config_id, r.fold) for r in runner.results().itertuples()}
    missing = planned - done
    if missing:
        counts = pd.Series([m for m, _, _ in missing]).value_counts().to_dict()
        raise SystemExit(f"A grelha não está completa ({len(missing)} ajustes em falta: {counts}). "
                         f"Primeiro: python scripts/run_cv.py {runner.task.name}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Congela os modelos de uma tarefa antes do teste.")
    parser.add_argument("task", choices=sorted(TRAIN_PKL))
    parser.add_argument("--final", help="Modelo final (por omissão: na regressão o melhor da "
                                        "validação cruzada; na classificação o do 04c)")
    parser.add_argument("--refit", action="store_true", help="Treinar de novo os já gravados")
    parser.add_argument("--force", action="store_true",
                        help="Congelar mesmo que o teste já tenha sido aberto ou a tag exista")
    args = parser.parse_args(argv)

    opened = (TEST_PREDICTIONS_DIR / f"{args.task}.pkl").exists()
    tagged = frozen_path(args.task).exists() and tag_status(args.task) in ("ok", "diferente")
    if (opened or tagged) and not args.force:
        reason = "o teste já foi previsto" if opened else f"a tag {TAG} já tem este ficheiro"
        raise SystemExit(f"Não congelo de novo: {reason}. Mudar os modelos depois disso tem de "
                         "ser explicado no relatório; se for mesmo preciso, usar --force.")

    runner = CrossValidationRunner(args.task, verbose=False)
    catalog = task_catalog(args.task)
    data = PickleStore.load(TRAIN_PKL[args.task], expected_type=pd.DataFrame)
    if args.task == "regression":
        check_grid(runner, catalog, data)

    record = frozen_models(args.task, cv_table(runner), catalog, final_model=args.final)
    print(f"{args.task}: {len(record['models'])} modelos, modelo final: {record['final_model']}\n")
    start = time.perf_counter()
    record = fit_frozen(record, runner, catalog, data, refit=args.refit)
    record = write_frozen(record)

    summary = pd.DataFrame([{"modelo": e["model"], "final": "sim" if e["model"] == record["final_model"] else "",
                             "configuração": e["params"] or "—", "limiar": e["threshold"],
                             runner.task.metric + " (CV)": e["cv"].get(runner.task.metric, "—"),
                             "n treino": e["n_train"], "treino (s)": e["fit_seconds"],
                             "ficheiro (MB)": e["size_mb"]} for e in record["models"]])
    if args.task == "regression":
        summary = summary.drop(columns="limiar")
    print("\n" + summary.to_string(index=False))
    print(f"\nTerminado em {(time.perf_counter() - start) / 60:.1f} min. "
          f"Gravado: {frozen_path(args.task).relative_to(PROJECT_ROOT)}")
    if record["uncommitted_code_changes"]:
        print("\nAVISO: há mudanças por gravar em src/ ou scripts/. Fazer commit do código e "
              "congelar de novo (--refit), para o JSON apontar para o código que treinou os modelos.")


if __name__ == "__main__":
    main()
