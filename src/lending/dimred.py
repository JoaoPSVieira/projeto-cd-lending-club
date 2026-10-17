"""
Redução de dimensionalidade para a EDA — PCA (linear) e UMAP (não linear).

Requisito do enunciado (Fase 2, EDA, ponto 2): usar pelo menos um método linear e um
não linear. Ligação ao T-02 (slides 71–92):
    - PCA: encontra as direções de maior variância (combinações lineares das variáveis).
      É rápido, interpretável (cada componente tem "pesos" por variável) e capta a
      estrutura GLOBAL dos dados.
    - UMAP: não linear; preserva sobretudo a estrutura LOCAL (vizinhanças), por isso é
      bom a revelar grupos. Os eixos não têm significado e é lento -> usa-se numa amostra.

Regras de leakage (enunciado, "fit learned transformations on training data only"):
    - A imputação, a padronização e o PCA são AJUSTADOS (fit) só nos dados de treino /
      desenvolvimento passados a fit_*. Nada aqui toca no teste.
    - Tudo está num Pipeline do scikit-learn, como o enunciado pede.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from lending import plot_style as ps
from lending.config import FIGURES_DIR, RANDOM_SEED
from lending.sampling import ReproducibleSampler


def log10_1p(x):
    """log10(1 + x), com valores negativos cortados a 0. Função com nome (e não lambda)
    para o pipeline poder ser gravado em pickle."""
    return np.log10(1 + np.clip(x, 0, None))


class DimensionalityReducer:
    """
    Ajusta PCA e UMAP sobre variáveis numéricas, com pré-processamento num Pipeline.

    Pré-processamento (igual para os dois métodos):
        1. log10(1 + x) nas colunas muito assimétricas indicadas (transformação fixa);
        2. imputação pela mediana (aprendida no treino);
        3. padronização: média 0, desvio-padrão 1 (aprendida no treino).
    Sem a padronização, variáveis com números grandes (ex.: rendimento) dominariam o
    PCA só pela escala (T-02, feature scaling).

    Attributes:
        numeric_cols (list[str]): Variáveis usadas.
        log_cols (list[str]): Subconjunto a transformar com log10(1 + x).
        sample_size (int): Tamanho da amostra para o UMAP e para os gráficos.
        pca_pipeline (Pipeline | None): Pipeline ajustado (depois de fit_pca).
        umap_embedding (np.ndarray | None): Coordenadas 2D do UMAP (depois de fit_umap).

    Methods:
        fit_pca(df), explained_variance(), n_components_for(threshold), loadings(...),
        pca_coordinates(df), fit_umap(df, ...), plot_scree(...), plot_embedding(...)
    """

    def __init__(self, numeric_cols, log_cols=(), sample_size=20_000, seed=RANDOM_SEED,
                 figures_dir=FIGURES_DIR):
        """
        Parameters:
            numeric_cols (list[str]): Colunas numéricas a usar.
            log_cols (iterable[str]): Colunas com cauda longa (ex.: annual_inc, revol_bal).
            sample_size (int): Linhas usadas no UMAP e desenhadas nos gráficos.
            seed (int): Semente (amostra, UMAP).
            figures_dir (str | Path): Pasta das figuras.
        """
        self.numeric_cols = list(numeric_cols)
        self.log_cols = [c for c in log_cols if c in self.numeric_cols]
        self.sample_size = sample_size
        self.seed = seed
        self.figures_dir = figures_dir
        self.sampler = ReproducibleSampler(seed)
        self.pca_pipeline = None
        self.umap_embedding = None
        self.umap_index = None
        ps.apply_style()

    # ------------------------------------------------------------------ pré-processamento

    def _preprocessor(self):
        """Constrói (sem ajustar) o pré-processamento: log -> imputação -> padronização."""
        other = [c for c in self.numeric_cols if c not in self.log_cols]
        log_step = ColumnTransformer(
            transformers=[
                ("log", FunctionTransformer(log10_1p, feature_names_out="one-to-one"),
                 self.log_cols),
                ("raw", "passthrough", other),
            ],
            verbose_feature_names_out=False,
        )
        return Pipeline([
            ("log", log_step),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])

    # ------------------------------------------------------------------ PCA

    def fit_pca(self, df):
        """
        Ajusta o pré-processamento e o PCA (todas as componentes) nos dados indicados.

        O PCA é rápido, por isso usa todas as linhas de df (não só a amostra).

        Parameters:
            df (pd.DataFrame): Dados de TREINO / desenvolvimento.

        Returns:
            self
        """
        self._drop_unusable(df)
        self.pca_pipeline = Pipeline([
            ("prep", self._preprocessor()),
            ("pca", PCA(random_state=self.seed)),
        ])
        self.pca_pipeline.fit(df[self.numeric_cols])
        return self

    def _drop_unusable(self, df):
        """
        Retira (com aviso) colunas que não servem para o PCA/UMAP: inexistentes, sempre
        vazias ou constantes. Ex.: uma coluna do bureau que só existe a partir de 2012
        está toda vazia no treino da classificação.
        """
        bad = [c for c in self.numeric_cols
               if c not in df.columns or df[c].notna().sum() == 0 or df[c].nunique() <= 1]
        if bad:
            print(f"[DimensionalityReducer] colunas ignoradas (inexistentes, vazias ou constantes): {bad}")
            self.numeric_cols = [c for c in self.numeric_cols if c not in bad]
            self.log_cols = [c for c in self.log_cols if c not in bad]

    def explained_variance(self):
        """
        Variância explicada por cada componente e acumulada.

        Returns:
            pd.DataFrame: colunas "var_%" e "acumulada_%", uma linha por componente.
        """
        self._check_pca()
        ratio = self.pca_pipeline.named_steps["pca"].explained_variance_ratio_ * 100
        return pd.DataFrame({
            "var_%": ratio,
            "acumulada_%": np.cumsum(ratio),
        }, index=[f"PC{i}" for i in range(1, len(ratio) + 1)]).round(2)

    def n_components_for(self, threshold=0.8):
        """
        Número mínimo de componentes para explicar `threshold` da variância (T-02:
        escolher pelo valor acumulado da variância explicada).

        Parameters:
            threshold (float): Fração (ex.: 0.8 = 80%).
        """
        cumulative = self.explained_variance()["acumulada_%"].values / 100
        return int(np.argmax(cumulative >= threshold) + 1)

    def loadings(self, n_components=2, top=8):
        """
        Pesos das variáveis nas primeiras componentes: dizem o que cada componente "mede".

        Ex.: se PC1 tem pesos altos em revol_bal, tot_cur_bal e total_acc, PC1 resume
        "volume de crédito" do mutuário. É isto que torna o PCA interpretável.

        Parameters:
            n_components (int): Componentes a mostrar.
            top (int): Variáveis com maior |peso| a mostrar por componente.

        Returns:
            pd.DataFrame: Uma coluna por componente, só as variáveis mais relevantes.
        """
        self._check_pca()
        pca = self.pca_pipeline.named_steps["pca"]
        names = self.pca_pipeline.named_steps["prep"].named_steps["log"].get_feature_names_out()
        table = pd.DataFrame(pca.components_[:n_components].T, index=names,
                             columns=[f"PC{i}" for i in range(1, n_components + 1)])
        keep = table.abs().max(axis=1).sort_values(ascending=False).head(top).index
        return table.loc[keep].round(3)

    def pca_coordinates(self, df, n_components=2):
        """
        Projeta df nas primeiras componentes, SEM reajustar (só transform).

        Parameters:
            df (pd.DataFrame): Dados a projetar (ex.: a amostra para o gráfico).
            n_components (int): Componentes devolvidas.

        Returns:
            np.ndarray: shape (len(df), n_components).
        """
        self._check_pca()
        return self.pca_pipeline.transform(df[self.numeric_cols])[:, :n_components]

    # ------------------------------------------------------------------ UMAP

    def fit_umap(self, df, stratify=None, n_neighbors=15, min_dist=0.1):
        """
        Ajusta o UMAP numa amostra reprodutível de df e guarda as coordenadas 2D.

        O pré-processamento é ajustado na mesma amostra. O UMAP serve aqui só para
        visualizar (EDA), não para alimentar modelos.

        Parameters:
            df (pd.DataFrame): Dados de TREINO / desenvolvimento.
            stratify (str | None): Coluna para estratificar a amostra (ex.: a classe).
            n_neighbors (int): Tamanho da vizinhança (maior = mais estrutura global).
            min_dist (float): Distância mínima entre pontos no mapa (menor = grupos mais densos).

        Returns:
            pd.DataFrame: A amostra usada (para depois colorir os pontos).
        """
        import umap   # import local: é lento a carregar e só é preciso aqui

        sample = self.sampler.sample(df, self.sample_size, stratify=stratify)
        self._drop_unusable(sample)
        X = self._preprocessor().fit_transform(sample[self.numeric_cols])
        # n_jobs=1: com random_state fixo o UMAP só é reprodutível num único processo
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2,
                            random_state=self.seed, n_jobs=1)
        self.umap_embedding = reducer.fit_transform(X)
        self.umap_index = sample.index
        return sample

    # ------------------------------------------------------------------ gráficos

    def plot_scree(self, max_components=15, threshold=0.8, filename=None):
        """
        Gráfico da variância explicada (barras) e acumulada (linha), com o limiar marcado.

        Parameters:
            max_components (int): Componentes mostradas.
            threshold (float): Limiar desenhado (ex.: 0.8).
            filename (str | None): Nome do ficheiro.
        """
        ev = self.explained_variance().head(max_components)
        k = self.n_components_for(threshold)
        x = np.arange(1, len(ev) + 1)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(x, ev["var_%"], color=ps.SERIES_1, width=0.7, label="variância explicada")
        ax.plot(x, ev["acumulada_%"], color=ps.CATEGORICAL[1], marker="o", markersize=4,
                label="acumulada")
        ax.axhline(threshold * 100, color=ps.MUTED, linewidth=1)
        ax.text(len(ev) + 0.4, threshold * 100, f"{threshold:.0%}: {k} comp.", fontsize=7,
                color=ps.MUTED, va="bottom", ha="right")
        ax.set_xticks(x)
        ax.set_xlabel("componente principal")
        ax.set_ylabel("% da variância")
        ax.set_ylim(0, 105)
        ax.grid(axis="x", visible=False)
        ax.legend(frameon=False, loc="center right")
        ax.set_title("PCA: variância explicada", loc="left")
        fig.tight_layout()
        return self._save(fig, filename)

    def plot_embedding(self, coords, color_values, kind="categorical", order=None, label=None,
                       title=None, xlabel="dim 1", ylabel="dim 2", filename=None):
        """
        Desenha um mapa 2D (PCA ou UMAP) com os pontos coloridos por uma variável.

        - kind="categorical": classes (ex.: Good / Default). Máx. 3 categorias. Pontos da
          classe mais pequena desenhados por cima, para não ficarem escondidos.
        - kind="continuous": valor numérico (ex.: int_rate), escala sequencial azul.

        Parameters:
            coords (np.ndarray): shape (n, 2).
            color_values (array-like): Valor de cor de cada ponto (mesma ordem de coords).
            kind (str): "categorical" ou "continuous".
            order (list | None): Só para categorias: a cor segue esta ordem (1.º azul,
                2.º laranja, 3.º verde-água). Usa a MESMA ordem em todas as figuras, para que
                cada classe tenha sempre a mesma cor (ex.: CLASS_ORDER no notebook).
            label (str | None): Nome da variável de cor (legenda / barra de cor).
            title (str | None): Título.
            xlabel, ylabel (str): Rótulos dos eixos.
            filename (str | None): Nome do ficheiro.
        """
        color_values = pd.Series(np.asarray(color_values))
        fig, ax = plt.subplots(figsize=(5, 4))

        if kind == "categorical":
            order = order or sorted(color_values.dropna().unique(), key=str)
            if len(order) > len(ps.CATEGORICAL):
                raise ValueError("No máximo 3 categorias (limite da paleta para pontos).")
            colors = dict(zip(order, ps.CATEGORICAL))      # cor fixa por categoria
            # Desenho: a maior classe primeiro (fica por baixo), a menor por cima
            draw = color_values.value_counts().index.tolist()
            for cat in draw:
                mask = (color_values == cat).values
                ax.scatter(coords[mask, 0], coords[mask, 1], s=3, alpha=0.5, linewidths=0,
                           color=colors[cat], label=f"{cat} (n={mask.sum():,})")
            ax.legend(title=label, frameon=False, markerscale=4, loc="best")
        else:
            order = np.argsort(color_values.values)       # valores altos por cima
            sc = ax.scatter(coords[order, 0], coords[order, 1], s=3, linewidths=0,
                            c=color_values.values[order], cmap=ps.SEQUENTIAL_CMAP)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(label or "")
            cbar.outline.set_visible(False)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title or "", loc="left")
        fig.tight_layout()
        return self._save(fig, filename)

    # ------------------------------------------------------------------ privados

    def _check_pca(self):
        if self.pca_pipeline is None:
            raise RuntimeError("Corre fit_pca(df) primeiro.")

    def _save(self, fig, filename):
        if filename:
            from pathlib import Path
            folder = Path(self.figures_dir)
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / filename)
        return fig
