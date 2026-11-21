"""
Análise dos resultados da validação cruzada - Fase 5 (comparação e avaliação)

Lê o que o runner (experiment.py) guardou e prepara as tabelas e os gráficos dos notebooks
04b (regressão) e 04c (classificação). Tudo usa só dados de desenvolvimento: as previsões
fora do fold e os modelos treinados nos folds. O teste de 2015 não é usado aqui.

Requisitos do enunciado cumpridos:
    - "Present the mean and the variation of the metric in the cross-validation procedures":
      fold_table() e plot_folds() mostram a métrica em cada fold.
    - "Report training sample sizes, tuning procedures, and any computational limitations":
      params_table() (efeito dos hiperparâmetros) e cost_table() (linhas e tempos).
    - "Strengths, weaknesses, and lessons learned": as interpretações (importância por
      permutação, resíduos por grupo e ao longo do tempo) dão a base para essa discussão.

Exemplo (04b):
    runner = CrossValidationRunner("regression")
    analysis = CVAnalysis(runner, data)
    analysis.fold_table()                      # MAE por trimestre de validação, por modelo
    analysis.residuals_by("sub_grade")         # erro do melhor modelo por sub-grau
    analysis.permutation_importance(spec)      # colunas originais que mais pesam
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from lending import config
from lending.evaluation import classification_metrics
from lending.plot_style import CATEGORICAL, MUTED, SEQUENTIAL_CMAP

BASELINE_STYLE = {"color": MUTED, "linestyle": "--", "linewidth": 1.2}


class CVAnalysis:
    """
    Tabelas e gráficos dos resultados da validação cruzada de uma tarefa

    Parameters:
        runner (CrossValidationRunner): Runner da tarefa, já corrido (lê o CSV e as previsões)
        data (pd.DataFrame): A mesma coorte de treino passada ao runner
        baselines (tuple[str, ...] | None): Nomes das baselines, desenhadas a tracejado; por
            omissão, as do catálogo ("Mediana", "Mediana por prazo", "Classe maioritária",
            "Taxa do grau")

    Methods:
        best_config(model), fold_table(metric), plot_folds(metric), plot_tuning(model, grid),\n        params_table(model),
        cost_table(), improvement(baseline), oof(model), residuals_by(column),
        permutation_importance(spec), plot_importance(importance), plot_pred_vs_real(model)
    """

    def __init__(self, runner, data, baselines=None):
        self.runner = runner
        self.task = runner.task
        self.data = runner.prepare(data)
        self.baselines = tuple(baselines or ("Mediana", "Mediana por prazo",
                                             "Classe maioritária", "Taxa do grau"))

    # ------------------------------------------------------------------ escolhas

    def _best(self):
        best = self.runner.best()
        if best.empty:
            raise ValueError("Ainda não há configurações completas nos resultados.")
        return best.set_index("model")

    def best_config(self, model):
        """config_id da melhor configuração de um modelo."""
        return self._best().loc[model, "config_id"]

    @property
    def models(self):
        """Modelos com uma configuração completa, do melhor para o pior."""
        return list(self._best().index)

    # ------------------------------------------------------------------ métricas por fold

    def fold_table(self, metric=None):
        """
        Métrica de cada fold na melhor configuração de cada modelo

        Na classificação, as métricas com limiar usam o limiar escolhido para a configuração
        (o mesmo do summary()); a AUC não depende do limiar.

        Returns:
            pd.DataFrame: Uma linha por modelo, uma coluna por período de validação
        """
        metric = metric or self.task.metric
        best = self._best()
        results = self.runner.results()
        periods = (results.drop_duplicates("fold").set_index("fold")["valid_period"].sort_index())
        rows = {}
        for model, row in best.iterrows():
            if self.task.is_classification:
                oof = self.runner.oof(model, row["config_id"])
                values = {f: classification_metrics(g["y_true"], g["p_default"],
                                                    row["threshold"])[metric]
                          for f, g in oof.groupby("fold")}
            else:
                part = results[(results["model"] == model) & (results["config_id"] == row["config_id"])]
                values = dict(zip(part["fold"], part[metric]))
            rows[model] = {periods[f]: v for f, v in values.items()}
        return pd.DataFrame(rows).T[list(periods)]

    def plot_folds(self, metric=None, ax=None, models=None):
        """Linha da métrica ao longo dos folds: uma cor e um marcador por modelo, baselines a tracejado."""
        metric = metric or self.task.metric
        table = self.fold_table(metric)
        if models is not None:
            table = table.loc[[m for m in table.index if m in models]]
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        colors, markers, dashes = plt.cm.tab10.colors, "osD^vP*X", ["--", ":", "-."]
        n_model = n_base = 0
        for model, values in table.iterrows():
            if model in self.baselines:
                style = {**BASELINE_STYLE, "linestyle": dashes[n_base % len(dashes)]}
                n_base += 1
            else:
                style = {"color": colors[n_model % len(colors)], "marker": markers[n_model % len(markers)]}
                n_model += 1
            ax.plot(table.columns, values.to_numpy(), label=model, **style)
        ax.set(xlabel="período de validação", ylabel=metric)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        return ax

    def plot_tuning(self, model, grid, metric=None, ax=None):
        """
        Efeito dos hiperparâmetros: o 1.º da grelha no eixo x (pela ordem da grelha), o 2.º como
        uma linha por valor

        Parameters:
            model (str): Nome do modelo
            grid (dict): A grelha do ModelSpec (dá a ordem dos valores)
        """
        metric = metric or self.task.metric
        table = self.params_table(model, metrics=[metric])
        names = list(grid)
        x_param = names[0]
        order = {json.dumps(v): i for i, v in enumerate(grid[x_param])}
        labels = [str(v) for v in grid[x_param]]
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 3.2))
        groups = table.groupby(table[names[1]].map(json.dumps)) if len(names) > 1 else [("", table)]
        for i, (value, part) in enumerate(groups):
            x = part[x_param].map(lambda v: order[json.dumps(v)])
            part = part.assign(_x=x.to_numpy()).sort_values("_x")
            label = f"{names[1]}={json.loads(value)}" if len(names) > 1 else None
            ax.plot(part["_x"], part[f"{metric}_mean"], marker="o",
                    color=plt.cm.tab10.colors[i % 10], label=label)
        ax.set_xticks(range(len(labels)), labels)
        ax.set(title=model, xlabel=x_param)
        if len(names) > 1:
            ax.legend(fontsize=7)
        return ax

    # ------------------------------------------------------------------ afinação e custo

    def params_table(self, model, metrics=None):
        """
        Todas as configurações de um modelo, com os hiperparâmetros em colunas

        Mostra o procedimento de afinação: o efeito de cada hiperparâmetro na métrica.

        Returns:
            pd.DataFrame: Uma linha por configuração, da melhor para a pior
        """
        summary = self.runner.summary()
        part = summary[summary["model"] == model].reset_index(drop=True)
        params = pd.DataFrame([json.loads(p) for p in part["params"]])
        metrics = metrics or [self.task.metric]
        columns = [c for m in metrics for c in (f"{m}_mean", f"{m}_std")]
        extra = ["threshold"] if self.task.is_classification else []
        return pd.concat([params, part[columns + extra + ["fit_seconds", "n_folds"]]], axis=1)

    def cost_table(self):
        """
        Custo computacional de cada modelo na validação cruzada

        Returns:
            pd.DataFrame: configurações, ajustes, linhas de treino (mín. e máx.), tempo total
                de treino (min) e tempos médios de treino e de previsão por ajuste (s)
        """
        results = self.runner.results()
        table = results.groupby("model").agg(
            configuracoes=("config_id", "nunique"), ajustes=("fold", "size"),
            treino_min_linhas=("n_train", "min"), treino_max_linhas=("n_train", "max"),
            treino_total_min=("fit_seconds", lambda s: s.sum() / 60),
            treino_medio_s=("fit_seconds", "mean"), previsao_media_s=("predict_seconds", "mean"))
        prep = results.drop_duplicates("fold")["prep_seconds"].sum() / 60
        table.attrs["pre_processamento_min"] = prep
        return table.sort_values("treino_total_min", ascending=False).round(2)

    def improvement(self, baseline, metric=None):
        """
        Melhoria relativa da melhor configuração de cada modelo face a uma baseline

        Para métricas de erro (MAE, RMSE) é a redução do erro; para as outras, o aumento.

        Returns:
            pd.Series: Em percentagem
        """
        metric = metric or self.task.metric
        means = self._best()[f"{metric}_mean"]
        base = means[baseline]
        sign = -1 if metric in ("mae", "rmse") else 1
        return (sign * (means - base) / abs(base) * 100).round(1)

    # ------------------------------------------------------------------ previsões fora do fold

    def oof(self, model, cid=None, columns=()):
        """
        Previsões fora do fold de uma configuração (por omissão, a melhor), com colunas dos
        dados originais e, na regressão, o resíduo (previsto - real)

        Returns:
            pd.DataFrame
        """
        cid = cid or self.best_config(model)
        oof = self.runner.oof(model, cid)
        if columns:
            extra = self.data.loc[oof["row"], list(columns)].reset_index(drop=True)
            oof = pd.concat([oof.reset_index(drop=True), extra], axis=1)
        if not self.task.is_classification:
            oof["residual"] = oof["y_pred"] - oof["y_true"]
        return oof

    def residuals_by(self, column, model=None, cid=None, freq=None):
        """
        Erro da regressão por grupo (ex.: sub_grade, term) ou ao longo do tempo (issue_d)

        Parameters:
            column (str): Coluna dos dados originais (não precisa de ser um preditor)
            model (str | None): Por omissão, o melhor modelo
            freq (str | None): Para colunas de data, o período ("M" mês, "Q" trimestre)

        Returns:
            pd.DataFrame: n, viés (resíduo médio: > 0 = sobrestima a taxa) e MAE por grupo
        """
        if self.task.is_classification:
            raise ValueError("residuals_by é para a regressão.")
        model = model or self.models[0]
        oof = self.oof(model, cid, columns=[column] if column != "issue_d" else [])
        key = oof[column]
        if freq is not None:
            key = pd.DatetimeIndex(key).to_period(freq).astype(str)
        grouped = oof.groupby(key.to_numpy())["residual"]
        return pd.DataFrame({"n": grouped.size(), "vies": grouped.mean(),
                             "mae": grouped.apply(lambda r: r.abs().mean())}).round(3)

    def plot_pred_vs_real(self, model=None, ax=None, gridsize=60):
        """Previsto contra real (densidade em hexágonos) nas previsões fora do fold."""
        model = model or self.models[0]
        oof = self.oof(model)
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4.5))
        hb = ax.hexbin(oof["y_true"], oof["y_pred"], gridsize=gridsize, mincnt=1, bins="log",
                       cmap=SEQUENTIAL_CMAP)
        low, high = np.percentile(oof[["y_true", "y_pred"]].to_numpy(), [0.5, 99.5])
        ax.plot([low, high], [low, high], color="0.3", linewidth=1)
        ax.set(xlabel="real", ylabel="previsto", title=model)
        plt.colorbar(hb, ax=ax, label="empréstimos (log)")
        return ax

    # ------------------------------------------------------------------ importância

    def permutation_importance(self, spec, cid=None, fold=-1, n_repeats=3, max_rows=20_000):
        """
        Importância por permutação das colunas ORIGINAIS, na validação de um fold

        O modelo é treinado no treino do fold (runner.fold_pipeline) e cada coluna original
        é baralhada na validação, uma de cada vez: a importância é quanto a métrica piora.
        Baralhar uma coluna original também baralha as features novas que dependem dela
        (ex.: loan_amnt e loan_to_income), por isso a importância é a da variável inteira.

        Parameters:
            spec (ModelSpec): O modelo
            cid (str | None): Configuração; por omissão, a melhor
            fold (int): Fold (como no CSV) ou -1 para o último
            n_repeats (int): Permutações por coluna
            max_rows (int): Amostra da validação (semente 42), para limitar o tempo

        Returns:
            pd.DataFrame: importância média e desvio, por coluna, da maior para a menor. Na
                regressão em p.p. de MAE; na classificação em pontos de AUC.
        """
        pipe, X_valid, y_valid = self.runner.fold_pipeline(self.data, spec, cid, fold)
        if len(X_valid) > max_rows:
            rng = np.random.default_rng(config.RANDOM_SEED)
            keep = np.sort(rng.choice(len(X_valid), max_rows, replace=False))
            X_valid, y_valid = X_valid.iloc[keep], y_valid.iloc[keep]
        scoring = "roc_auc" if self.task.is_classification else "neg_mean_absolute_error"
        result = permutation_importance(pipe, X_valid, y_valid, scoring=scoring,
                                        n_repeats=n_repeats, random_state=config.RANDOM_SEED)
        return (pd.DataFrame({"importancia": result.importances_mean,
                              "desvio": result.importances_std}, index=X_valid.columns)
                .sort_values("importancia", ascending=False))

    @staticmethod
    def plot_importance(importance, top=15, ax=None, xlabel="importância"):
        """Barras horizontais das colunas mais importantes."""
        part = importance.head(top).iloc[::-1]
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 0.3 * len(part) + 1))
        ax.barh(part.index, part["importancia"], xerr=part["desvio"], color=CATEGORICAL[0])
        ax.set(xlabel=xlabel)
        return ax
