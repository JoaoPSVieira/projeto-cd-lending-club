"""
Pipeline de pré-processamento com ajuste (fit) só no treino

Regras do enunciado que este módulo cumpre:
    - "Fit imputers, scalers, encoders, ... clipping ... only on the training portion of each
      fold. Use pipelines": tudo o que depende dos dados está num Pipeline do scikit-learn.
      O fit vê só o treino; a validação e o teste só passam pelo transform.
    - "Normalize/standardize ... fitting only on training data": StandardScaler no fim.
    - "Distinguish not applicable from missing ... do not zero-fill everything": cada coluna
      numérica com valores em falta ganha um indicador; o valor imputado é a mediana do
      treino, nunca 0. Nas categorias, o valor em falta é uma categoria explícita.
    - Validação que respeita o tempo: ExpandingWindowSplit (janela expansível por período).

Passos (o que é aprendido no fit está entre parênteses):
    Montantes em USD   colunas vazias (quais) -> log(1 + x) -> clipping (percentis)
                       -> imputação (mediana) + indicador -> padronização (média, desvio)
    Outras numéricas   colunas vazias (quais) -> clipping (percentis)
                       -> imputação (mediana) + indicador -> padronização (média, desvio)
    Categorias         colunas vazias (quais) -> "em falta" como categoria
                       -> one-hot (categorias; as raras ficam juntas)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# Percentis do treino usados no clipping (só os 0,5% mais extremos de cada lado são afetados)
CLIP_QUANTILES = (0.005, 0.995)

# Categorias com menos desta fração do treino são juntas numa categoria "rara"
MIN_CATEGORY_FREQUENCY = 0.005

MISSING_CATEGORY = "em falta"


class DropEmptyColumns(BaseEstimator, TransformerMixin):
    """
    Retira as colunas sem nenhum valor no treino

    Uma coluna vazia no treino não tem nada para aprender (ex.: as 37 colunas do bureau
    registadas só a partir de 2012, no treino da classificação de 2007-2011). Quais são
    vazias é aprendido no fit; no transform retiram-se as mesmas, mesmo que na validação
    ou no teste tenham valores.

    Attributes:
        kept_ (list[str]): Colunas com pelo menos um valor no treino
        dropped_ (list[str]): Colunas vazias no treino
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        filled = X.notna().any()
        self.kept_ = list(filled.index[filled])
        self.dropped_ = list(filled.index[~filled])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.kept_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_, dtype=object)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Limita cada coluna aos percentis do treino (winsorização)

    Reduz o peso de valores extremos (ex.: revol_util até 892%, rendimentos de milhões) sem
    retirar linhas. Os limites são aprendidos no fit, só com o treino.

    Attributes:
        lower_ (pd.Series): Limite inferior de cada coluna
        upper_ (pd.Series): Limite superior de cada coluna
    """

    def __init__(self, quantiles=CLIP_QUANTILES):
        self.quantiles = quantiles

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(self.quantiles[0])
        self.upper_ = X.quantile(self.quantiles[1])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(lower=self.lower_, upper=self.upper_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_, dtype=object)


def _numeric_steps(log):
    """Passos das colunas numéricas; com log=True começa por log(1 + x)."""
    steps = [("drop_empty", DropEmptyColumns())]
    if log:
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps += [
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ]
    return Pipeline(steps)


def build_preprocessor(features):
    """
    Pipeline de pré-processamento das colunas de uma tarefa

    Parameters:
        features (TaskFeatures): Colunas de entrada, agrupadas por tratamento

    Returns:
        ColumnTransformer: Ainda por ajustar; usar fit só no treino de cada fold.
            Devolve um DataFrame com nomes de colunas (set_output pandas).
    """
    categorical = Pipeline([
        ("drop_empty", DropEmptyColumns()),
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
        ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=MIN_CATEGORY_FREQUENCY, sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(
        [("money", _numeric_steps(log=True), features.money),
         ("numeric", _numeric_steps(log=False), features.numeric),
         ("categorical", categorical, features.categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


class ExpandingWindowSplit:
    """
    Validação cruzada temporal com janela expansível

    Os dados são divididos em períodos de calendário (ex.: trimestres). Cada fold treina
    com todos os períodos anteriores e valida no período seguinte, por isso nunca se treina
    com o futuro de quem se valida. Os empréstimos do mesmo período ficam sempre juntos.

    Attributes:
        freq (str): Período do pandas ("Q" trimestre, "Y" ano)
        min_train_periods (int): Períodos de treino do primeiro fold

    Methods:
        split(dates): Pares (posições de treino, posições de validação), como no scikit-learn
        describe(dates): Tabela com o período e o tamanho de cada fold
    """

    def __init__(self, freq, min_train_periods):
        self.freq = freq
        self.min_train_periods = min_train_periods

    def _periods(self, dates):
        return pd.Series(pd.DatetimeIndex(dates).to_period(self.freq))

    def split(self, dates):
        """
        Parameters:
            dates (pd.Series | pd.DatetimeIndex): Data de cada linha, pela ordem dos dados

        Yields:
            tuple[np.ndarray, np.ndarray]: Posições das linhas de treino e de validação
        """
        periods = self._periods(dates)
        ordered = sorted(periods.unique())
        if len(ordered) <= self.min_train_periods:
            raise ValueError(f"Só há {len(ordered)} períodos: não chegam para nenhum fold.")
        for k in range(self.min_train_periods, len(ordered)):
            yield (np.flatnonzero((periods < ordered[k]).to_numpy()),
                   np.flatnonzero((periods == ordered[k]).to_numpy()))

    def describe(self, dates):
        """
        Returns:
            pd.DataFrame: Uma linha por fold: períodos de treino, período de validação e
                número de linhas de cada um
        """
        periods = self._periods(dates)
        rows = []
        for n, (train, valid) in enumerate(self.split(dates), start=1):
            rows.append({"fold": n,
                         "treino": f"{periods.iloc[train].min()} a {periods.iloc[train].max()}",
                         "validação": str(periods.iloc[valid].iloc[0]),
                         "linhas treino": len(train), "linhas validação": len(valid)})
        return pd.DataFrame(rows).set_index("fold")
