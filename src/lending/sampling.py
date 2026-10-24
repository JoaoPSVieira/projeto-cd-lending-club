"""
Amostragem reprodutível

O enunciado exige "reproducible sampling" e proíbe usar as primeiras linhas do ficheiro
como amostra. Como o ficheiro está ordenado no tempo, as primeiras linhas são só dos
primeiros anos. Todas as amostras do projeto passam por aqui

Quando são usadas:
    - gráficos com centenas de milhares de pontos (só se desenha uma amostra)
    - UMAP e silhueta, que são lentos em quantidade grandes de dados
"""

import pandas as pd

from lending.config import RANDOM_SEED


class ReproducibleSampler:
    """
    Extrai amostras aleatórias reprodutíveis de um DataFrame

    Attributes:
        seed (int): Semente do gerador aleatório (por omissão é usada a do projeto
            definida no config.py - RANDOM_SEED)

    Methods:
        sample(df, n, stratify=None): Devolve uma amostra de n linhas
    """

    def __init__(self, seed=RANDOM_SEED):
        """
        Parameters:
            seed (int): Semente. A mesma semente dá sempre a mesma amostra
        """
        self.seed = seed

    def sample(self, df, n, stratify=None):
        """
        Devolve uma amostra aleatória de n linhas, ou o DataFrame todo se for mais pequeno que n

        Parameters:
            df (pd.DataFrame): Dados de origem.
            n (int): Tamanho pretendido da amostra.
            stratify (str | None): Nome de uma coluna categórica. Se indicado, a amostra
                mantém a proporção de cada categoria (ex: 87,7% Good / 12,3% Default).

        Returns:
            pd.DataFrame: Amostra com o índice original
        """
        if n >= len(df):
            return df

        if stratify is None:
            return df.sample(n=n, random_state=self.seed)

        # Estratificada (a mesma fração de cada grupo)
        frac = n / len(df)
        parts = [
            group.sample(frac=frac, random_state=self.seed)
            for _, group in df.groupby(stratify, observed=True)
        ]
        return pd.concat(parts).sample(frac=1, random_state=self.seed)  # baralhar
