"""
Análise exploratória (EDA) — Fase 2 do enunciado.

Requisitos do enunciado que este módulo cobre:
    1. Estatística descritiva e visualizações (histogramas e gráficos de relação)
       para investigar padrões, outliers e associações.
    3. EDA feita só com dados de desenvolvimento, separada do teste final.
    (O ponto 2, PCA e UMAP, está em lending/dimred.py.)

Princípios:
    - As classes NÃO conhecem nomes de colunas: recebem-nos por parâmetro (nota do
      professor sobre valores "hardcoded"). Os nomes ficam no notebook.
    - As classes NÃO alteram os dados: só calculam tabelas e desenham figuras.
      Remover outliers ou imputar é limpeza/pré-processamento, não EDA.
    - Gráficos com muitos pontos usam uma amostra reprodutível (ReproducibleSampler).
    - Cada figura pode ser gravada em reports/figures/ para entrar no relatório.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lending import plot_style as ps
from lending.config import FIGURES_DIR, RANDOM_SEED
from lending.sampling import ReproducibleSampler


# =============================================================================
# Tabelas
# =============================================================================

class DescriptiveAnalysis:
    """
    Tabelas de estatística descritiva. Não altera os dados.

    Attributes:
        df (pd.DataFrame): Dados de desenvolvimento (já limpos e com tipos corretos).

    Methods:
        summary(cols): Estatísticas de colunas numéricas (inclui % em falta e assimetria).
        missing(cols): % de valores em falta, ordenado.
        outliers(cols, k): Contagem de valores fora das barreiras de Tukey (IQR).
        by_category(cat_col, target_col): Tamanho e média do alvo por categoria.
        correlation(cols, method): Matriz de correlação.
    """

    def __init__(self, df):
        """
        Parameters:
            df (pd.DataFrame): Dados de desenvolvimento.
        """
        self.df = df

    def summary(self, cols):
        """
        Estatística descritiva das colunas numéricas indicadas.

        Inclui a assimetria (skew): valores muito acima de 1 indicam cauda longa à
        direita (ex.: rendimento), o que sugere transformação logarítmica e escalas
        robustas mais tarde.

        Parameters:
            cols (list[str]): Colunas numéricas.

        Returns:
            pd.DataFrame: Uma linha por coluna.
        """
        data = self.df[cols]
        table = data.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
        table.insert(1, "missing_%", data.isna().mean().mul(100))
        table["skew"] = data.skew()
        return table.round(2)

    def missing(self, cols):
        """
        Percentagem de valores em falta por coluna, da maior para a menor.

        Parameters:
            cols (list[str]): Colunas a analisar.

        Returns:
            pd.Series: % em falta, ordenada.
        """
        return self.df[cols].isna().mean().mul(100).sort_values(ascending=False).round(2)

    def outliers(self, cols, k=1.5):
        """
        Conta valores fora das barreiras de Tukey: [Q1 - k*IQR, Q3 + k*IQR] (T-02).

        É só diagnóstico: nada é removido. Num dataset de crédito, valores extremos
        (rendimentos muito altos, utilização > 100%) podem ser reais.

        Parameters:
            cols (list[str]): Colunas numéricas.
            k (float): Multiplicador do IQR (1,5 é o valor clássico).

        Returns:
            pd.DataFrame: Limites, número e % de valores abaixo e acima.
        """
        rows = []
        for col in cols:
            values = self.df[col].dropna()
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - k * iqr, q3 + k * iqr
            n_low = int((values < low).sum())
            n_high = int((values > high).sum())
            rows.append({
                "coluna": col,
                "limite_inf": low,
                "limite_sup": high,
                "n_abaixo": n_low,
                "n_acima": n_high,
                "%_fora": 100 * (n_low + n_high) / max(len(values), 1),
            })
        return pd.DataFrame(rows).set_index("coluna").round(2)

    def by_category(self, cat_col, target_col, min_count=0):
        """
        Para cada categoria: número de empréstimos, % do total e média do alvo.

        - Alvo binário (0/1), ex.: is_default -> a média é a taxa de default.
        - Alvo contínuo, ex.: int_rate -> a média é a taxa de juro média.

        Parameters:
            cat_col (str): Coluna categórica.
            target_col (str): Coluna do alvo.
            min_count (int): Esconde categorias com menos casos (médias pouco fiáveis).

        Returns:
            pd.DataFrame: Ordenado pela média do alvo.
        """
        grouped = self.df.groupby(cat_col, observed=True)[target_col]
        table = pd.DataFrame({
            "n": grouped.size(),
            "%_total": grouped.size() / len(self.df) * 100,
            f"media_{target_col}": grouped.mean(),
        })
        table = table[table["n"] >= min_count]
        return table.sort_values(f"media_{target_col}", ascending=False).round(3)

    def correlation(self, cols, method="spearman"):
        """
        Matriz de correlação.

        Por omissão usa Spearman (por postos): não exige normalidade nem relação linear
        e é pouco sensível a outliers, o que convém em variáveis muito assimétricas.
        Pearson mede só relação linear (T-02, filtros de seleção de variáveis).

        Parameters:
            cols (list[str]): Colunas numéricas.
            method (str): "spearman" ou "pearson".

        Returns:
            pd.DataFrame: Matriz de correlação.
        """
        return self.df[cols].corr(method=method).round(2)


# =============================================================================
# Gráficos
# =============================================================================

class EDAPlotter:
    """
    Gráficos da EDA, com o estilo do projeto, prontos para o relatório.

    Todos os métodos recebem o DataFrame e os nomes das colunas, devolvem a figura e,
    se for dado `filename`, gravam-na em `figures_dir`.

    Attributes:
        figures_dir (Path): Pasta onde as figuras são gravadas.
        max_points (int): Máximo de pontos desenhados nos gráficos de pontos.
        sampler (ReproducibleSampler): Amostragem reprodutível.

    Methods:
        histograms, missing_bar, hexbin, box_by_category, raincloud,
        rate_by_category, rate_by_bins, correlation_heatmap
    """

    def __init__(self, figures_dir=FIGURES_DIR, max_points=5_000, seed=RANDOM_SEED):
        """
        Parameters:
            figures_dir (str | Path): Pasta de destino das figuras.
            max_points (int): Pontos desenhados no máximo (o resto fica na amostra).
            seed (int): Semente da amostragem.
        """
        self.figures_dir = Path(figures_dir)
        self.max_points = max_points
        self.sampler = ReproducibleSampler(seed)
        ps.apply_style()

    # ------------------------------------------------------------------ distribuições

    def histograms(self, df, cols, log_cols=(), clip_cols=(), clip_quantile=0.995, bins=50,
                   ncols=3, title=None, filename=None):
        """
        Histogramas de várias variáveis numéricas, em grelha (um painel por variável).

        Parameters:
            df (pd.DataFrame): Dados.
            cols (list[str]): Colunas numéricas.
            log_cols (iterable[str]): Colunas mostradas em escala log10(1 + x), úteis para
                variáveis com cauda longa (ex.: rendimento, saldos).
            bins (int): Número de barras.
            ncols (int): Painéis por linha.
            clip_cols (iterable[str]): Colunas com valores extremos raros (ex.: dti, revol_util)
                cuja cauda é escondida SÓ no gráfico, acima de `clip_quantile`, para a forma ser
                legível. Os dados não mudam; o número de valores escondidos aparece no título.
            clip_quantile (float): Percentil de corte para as `clip_cols` (0,995 = p99,5).
            title (str | None): Título geral.
            filename (str | None): Nome do ficheiro (ex.: "hist_regressao.png").

        Returns:
            matplotlib.figure.Figure
        """
        nrows = int(np.ceil(len(cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows), squeeze=False)

        for ax, col in zip(axes.flat, cols):
            values = df[col].dropna()
            label = col
            hidden = 0
            if col in log_cols:
                values = np.log10(1 + values.clip(lower=0))
                label = f"log10(1 + {col})"
            elif col in clip_cols and len(values):
                upper = values.quantile(clip_quantile)
                hidden = int((values > upper).sum())
                values = values[values <= upper]

            ax.hist(values, bins=bins, color=ps.SERIES_1, edgecolor=ps.SURFACE, linewidth=0.3)
            if hidden:
                label = f"{label}  ({hidden:,} > p{clip_quantile*100:g} omitidos)"
            ax.set_title(label, fontsize=8, loc="left")
            ax.set_ylabel("n")
            ax.grid(axis="x", visible=False)

        for ax in list(axes.flat)[len(cols):]:   # painéis vazios
            ax.set_visible(False)

        if title:
            fig.suptitle(title, x=0.01, ha="left", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))  # espaço para o título
        else:
            fig.tight_layout()
        return self._save(fig, filename)

    def missing_bar(self, df, cols, top=30, filename=None):
        """
        Barras horizontais com a % de valores em falta por coluna (só as que têm falta).

        Parameters:
            df (pd.DataFrame): Dados.
            cols (list[str]): Colunas a analisar.
            top (int): Mostra no máximo as `top` colunas com mais falta.
            filename (str | None): Nome do ficheiro.
        """
        pct = df[cols].isna().mean().mul(100)
        pct = pct[pct > 0].sort_values().tail(top)

        fig, ax = plt.subplots(figsize=(6, max(1.5, 0.22 * len(pct) + 0.8)))
        xmax = min(100, pct.max() * 1.2) if len(pct) else 1
        ax.barh(pct.index, pct.values, color=ps.SERIES_1, height=0.7)
        for y, v in enumerate(pct.values):   # rótulo logo a seguir ao fim de cada barra
            ax.text(v + xmax * 0.01, y, f"{v:.1f}%", va="center", fontsize=7,
                    color=ps.INK_SECONDARY)
        ax.set_xlabel("% de valores em falta")
        ax.set_xlim(0, xmax)
        ax.grid(axis="y", visible=False)
        ax.set_title("Valores em falta por coluna", loc="left")
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------ relações

    def hexbin(self, df, x, y, gridsize=45, clip_quantile=0.995, filename=None):
        """
        Relação entre duas variáveis numéricas com muitos pontos (densidade em hexágonos).

        Com centenas de milhares de empréstimos, um scatter vira uma mancha. O hexbin
        mostra onde estão os pontos (cor = quantidade) e acrescenta a mediana de y por
        intervalo de x, que mostra a tendência.

        Parameters:
            df (pd.DataFrame): Dados.
            x, y (str): Colunas (ex.: x="fico_range_low", y="int_rate").
            gridsize (int): Resolução dos hexágonos.
            clip_quantile (float | None): Esconde a cauda extrema de x e y só no gráfico.
            filename (str | None): Nome do ficheiro.
        """
        data = df[[x, y]].dropna()
        if clip_quantile is not None:
            data = data[(data[x] <= data[x].quantile(clip_quantile))
                        & (data[y] <= data[y].quantile(clip_quantile))]

        fig, ax = plt.subplots(figsize=(5, 3.6))
        hb = ax.hexbin(data[x], data[y], gridsize=gridsize, cmap=ps.SEQUENTIAL_CMAP,
                       mincnt=1, bins="log", linewidths=0)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("n de empréstimos (escala log)")
        cbar.outline.set_visible(False)

        # Mediana de y por 20 intervalos de x (tendência)
        bins = pd.qcut(data[x], q=20, duplicates="drop")
        trend = data.groupby(bins, observed=True).agg(xm=(x, "median"), ym=(y, "median"))
        ax.plot(trend["xm"], trend["ym"], color=ps.CATEGORICAL[1], linewidth=2)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        # Uma só linha: o título identifica-a (sem caixa de legenda por cima dos dados)
        ax.set_title(f"{y} vs {x}  (linha laranja: mediana de {y})", loc="left")
        fig.tight_layout()
        return self._save(fig, filename)

    def box_by_category(self, df, num_col, cat_col, order=None, min_count=100,
                        filename=None):
        """
        Caixas de bigodes (boxplots) de uma variável numérica por categoria.

        Útil na regressão: ex.: int_rate por term, por purpose, por home_ownership.
        Os outliers individuais não são desenhados (seriam milhares de pontos); as caixas
        mostram mediana, quartis e bigodes de 1,5×IQR.

        Parameters:
            df (pd.DataFrame): Dados.
            num_col (str): Variável numérica (ex.: "int_rate").
            cat_col (str): Variável categórica (ex.: "purpose").
            order (list | None): Ordem das categorias. Por omissão, pela mediana.
            min_count (int): Ignora categorias com poucos casos.
            filename (str | None): Nome do ficheiro.
        """
        data = df[[num_col, cat_col]].dropna()
        counts = data[cat_col].value_counts()
        keep = counts[counts >= min_count].index
        data = data[data[cat_col].isin(keep)]
        if order is None:
            order = data.groupby(cat_col, observed=True)[num_col].median().sort_values().index.tolist()

        groups = [data.loc[data[cat_col] == c, num_col].values for c in order]
        fig, ax = plt.subplots(figsize=(6, max(1.8, 0.32 * len(order) + 0.8)))
        bp = ax.boxplot(groups, orientation="horizontal", showfliers=False, widths=0.6,
                        patch_artist=True, medianprops={"color": ps.INK, "linewidth": 1.2},
                        whiskerprops={"color": ps.AXIS}, capprops={"color": ps.AXIS})
        for box in bp["boxes"]:
            box.set_facecolor(ps.SERIES_1)
            box.set_alpha(0.85)
        ax.set_yticks(range(1, len(order) + 1), [f"{c}  (n={counts[c]:,})" for c in order])
        ax.set_xlabel(num_col)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{num_col} por {cat_col}", loc="left")
        fig.tight_layout()
        return self._save(fig, filename)

    def raincloud(self, df, num_col, group_col, order=None, log=False, filename=None):
        """
        Gráfico "raincloud" (sugestão do professor): meio violino + caixa + pontos.

        Mostra três coisas ao mesmo tempo para cada grupo (ex.: Good vs Default):
        a forma da distribuição (violino), o resumo (caixa) e os dados reais (pontos,
        de uma amostra reprodutível). Usa no máximo 3 grupos (limite da paleta).

        Parameters:
            df (pd.DataFrame): Dados.
            num_col (str): Variável numérica (ex.: "dti").
            group_col (str): Grupo (ex.: a classe do alvo).
            order (list | None): Ordem dos grupos. A cor segue esta ordem (1.º azul, 2.º laranja),
                por isso usa sempre a mesma ordem em todas as figuras (ex.: CLASS_ORDER).
            log (bool): Mostra log10(1 + x) (para variáveis com cauda longa).
            filename (str | None): Nome do ficheiro.
        """
        data = df[[num_col, group_col]].dropna()
        if order is None:
            order = sorted(data[group_col].unique())
        if len(order) > len(ps.CATEGORICAL):
            raise ValueError("O raincloud suporta no máximo 3 grupos (limite da paleta).")

        values = lambda g: data.loc[data[group_col] == g, num_col]   # noqa: E731
        transform = (lambda v: np.log10(1 + v.clip(lower=0))) if log else (lambda v: v)
        rng = np.random.default_rng(RANDOM_SEED)

        fig, ax = plt.subplots(figsize=(6, 1.3 * len(order) + 0.8))
        for i, (group, color) in enumerate(zip(order, ps.CATEGORICAL)):
            y = len(order) - i            # primeiro grupo em cima
            v = transform(values(group))
            if len(v) < 2:
                continue

            # 1) meio violino (só a metade de cima)
            parts = ax.violinplot(v, positions=[y], orientation="horizontal", widths=0.8,
                                  showextrema=False)
            for body in parts["bodies"]:
                verts = body.get_paths()[0].vertices
                verts[:, 1] = np.clip(verts[:, 1], y, None)
                body.set_facecolor(color)
                body.set_alpha(0.55)

            # 2) caixa estreita
            ax.boxplot(v, positions=[y], orientation="horizontal", widths=0.12,
                       showfliers=False, patch_artist=True,
                       boxprops={"facecolor": ps.SURFACE, "edgecolor": ps.INK_SECONDARY},
                       medianprops={"color": ps.INK}, whiskerprops={"color": ps.INK_SECONDARY},
                       capprops={"color": ps.INK_SECONDARY})

            # 3) pontos (amostra), abaixo da caixa, com ligeiro desvio vertical
            pts = self.sampler.sample(v.to_frame(), min(len(v), self.max_points // len(order)))[num_col]
            ax.scatter(pts, y - 0.25 + rng.uniform(-0.08, 0.08, len(pts)), s=2,
                       color=color, alpha=0.35, linewidths=0)

        ax.set_yticks(range(len(order), 0, -1), [str(g) for g in order])
        ax.set_xlabel(f"log10(1 + {num_col})" if log else num_col)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{num_col} por {group_col}", loc="left")
        fig.tight_layout()
        return self._save(fig, filename)

    def rate_by_category(self, df, cat_col, target_col, order=None, min_count=100,
                         xlabel=None, filename=None):
        """
        Barras com a média de um alvo binário por categoria (ex.: taxa de default por grade).

        Uma só série -> uma só cor. O valor de cada barra está escrito no fim da barra, e a
        linha vertical cinzenta é a taxa global, para comparar.

        Parameters:
            df (pd.DataFrame): Dados.
            cat_col (str): Categoria (ex.: "grade", "purpose").
            target_col (str): Alvo binário 0/1 (ex.: "is_default").
            order (list | None): Ordem das categorias (ex.: A..G). Por omissão, pela taxa.
            min_count (int): Ignora categorias com poucos casos.
            xlabel (str | None): Rótulo do eixo.
            filename (str | None): Nome do ficheiro.
        """
        grouped = df.groupby(cat_col, observed=True)[target_col].agg(["mean", "size"])
        grouped = grouped[grouped["size"] >= min_count]
        if order is None:
            grouped = grouped.sort_values("mean")
        else:
            grouped = grouped.reindex([c for c in order if c in grouped.index][::-1])

        fig, ax = plt.subplots(figsize=(6, max(1.8, 0.3 * len(grouped) + 0.8)))
        ax.barh([str(c) for c in grouped.index], grouped["mean"] * 100, color=ps.SERIES_1, height=0.7)
        for y, (rate, n) in enumerate(zip(grouped["mean"], grouped["size"])):
            ax.text(rate * 100 + 0.3, y, f"{rate*100:.1f}%  (n={n:,})", va="center",
                    fontsize=7, color=ps.INK_SECONDARY)
        overall = df[target_col].mean() * 100
        ax.axvline(overall, color=ps.MUTED, linewidth=1)
        ax.text(overall, len(grouped) - 0.4, f" global {overall:.1f}%", fontsize=7,
                color=ps.MUTED, va="bottom")
        ax.set_xlabel(xlabel or f"% {target_col}")
        ax.set_xlim(0, grouped["mean"].max() * 100 * 1.35)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{target_col} por {cat_col}", loc="left")
        fig.tight_layout()
        return self._save(fig, filename)

    def rate_by_bins(self, df, num_col, target_col, q=10, filename=None):
        """
        Taxa de um alvo binário por decil de uma variável numérica (linha com pontos).

        Mostra se o risco sobe ou desce com a variável e se a relação é linear ou não
        (ex.: taxa de default por decil de dti). Útil antes dos testes de hipóteses.

        Parameters:
            df (pd.DataFrame): Dados.
            num_col (str): Variável numérica.
            target_col (str): Alvo binário 0/1.
            q (int): Número de grupos (10 = decis).
            filename (str | None): Nome do ficheiro.
        """
        data = df[[num_col, target_col]].dropna()
        bins = pd.qcut(data[num_col], q=q, duplicates="drop")
        table = data.groupby(bins, observed=True).agg(x=(num_col, "median"), rate=(target_col, "mean"),
                                                      n=(target_col, "size"))

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(table["x"], table["rate"] * 100, color=ps.SERIES_1, marker="o", markersize=5)
        ax.axhline(data[target_col].mean() * 100, color=ps.MUTED, linewidth=1)
        ax.set_xlabel(f"{num_col} (mediana de cada decil)")
        ax.set_ylabel(f"% {target_col}")
        ax.set_title(f"{target_col} por decil de {num_col}", loc="left")
        fig.tight_layout()
        return self._save(fig, filename)

    def correlation_heatmap(self, df, cols, method="spearman", filename=None):
        """
        Mapa de calor da matriz de correlação, com escala divergente (-1 azul, 0 cinzento,
        +1 vermelho). Os valores com |r| >= 0,3 são escritos na célula; a diagonal (sempre 1)
        fica em branco.

        Parameters:
            df (pd.DataFrame): Dados.
            cols (list[str]): Colunas numéricas (inclui o alvo, se for numérico).
            method (str): "spearman" ou "pearson".
            filename (str | None): Nome do ficheiro.
        """
        corr = df[cols].corr(method=method)
        n = len(cols)
        shown = corr.values.copy()
        np.fill_diagonal(shown, np.nan)          # a diagonal é sempre 1: não interessa
        fig, ax = plt.subplots(figsize=(0.42 * n + 2.2, 0.42 * n + 1.6))
        im = ax.imshow(shown, cmap=ps.DIVERGING_CMAP, vmin=-1, vmax=1)
        ax.set_xticks(range(n), cols, rotation=90)
        ax.set_yticks(range(n), cols)
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                r = corr.values[i, j]
                if i != j and abs(r) >= 0.3:
                    ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=6,
                            color=ps.SURFACE if abs(r) >= 0.7 else ps.INK)
        cbar = fig.colorbar(im, ax=ax, fraction=0.04)
        cbar.set_label(f"correlação de {method}")
        cbar.outline.set_visible(False)
        ax.set_title(f"Correlação ({method})", loc="left")
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------ privados

    def _save(self, fig, filename):
        """Grava a figura (se houver nome) e devolve-a."""
        if filename:
            self.figures_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.figures_dir / filename)
        return fig
