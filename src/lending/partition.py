"""
Partição temporal com cortes de calendário explícitos

Regras do enunciado que esta classe cumpre:
    - "Define partitions before data-dependent preprocessing": é o primeiro passo depois
      das regras estruturais; nenhuma estatística dos dados é usada para os cortes.
    - "Use explicit calendar cutoffs rather than file order, and keep same-month records
      together": os cortes são meses fixos em config.PARTITIONS, nunca a ordem das linhas.

Decisões de desenho:
1 - As datas são passadas já convertidas:
    O accepted usa issue_d ("Dec-2015") e o rejected usa Application Date ("2015-10-29").
    A classe só recebe datas, por isso serve para os dois ficheiros.

2 - Todas as linhas têm de cair numa e uma só partição:
    Os períodos são validados (ordenados, sem sobreposições nem buracos) e uma data em
    falta ou fora dos períodos dá erro. Nenhuma linha desaparece sem ser registada.
"""

import pandas as pd

from lending import config


class TemporalPartitioner:
    """
    Atribui cada linha a uma partição pelo mês da sua data

    Attributes:
        partitions (dict[str, tuple[pd.Period, pd.Period]]): Nome -> (primeiro mês, último mês)

    Methods:
        assign(dates): Nome da partição de cada linha
        split(df, dates, names=None): DataFrame de cada partição
        summary(dates): Tabela com período, linhas e percentagem de cada partição
    """

    def __init__(self, partitions=None):
        """
        Parameters:
            partitions (dict[str, tuple[str, str]] | None): Nome -> ("AAAA-MM", "AAAA-MM"),
                com início e fim incluídos e por ordem cronológica.
                Por omissão usa config.PARTITIONS.

        Raises:
            ValueError: Se os períodos não forem contíguos, ordenados e sem sobreposição
        """
        partitions = config.PARTITIONS if partitions is None else partitions
        self.partitions = {
            name: (pd.Period(start, freq="M"), pd.Period(end, freq="M"))
            for name, (start, end) in partitions.items()
        }
        self._validate()

    def assign(self, dates):
        """
        Nome da partição de cada linha

        Parameters:
            dates (pd.Series): Datas (datetime64), uma por linha

        Returns:
            pd.Series: Categórica com o nome da partição, com o mesmo índice

        Raises:
            ValueError: Se houver datas em falta ou fora de todas as partições
        """
        if dates.isna().any():
            raise ValueError(f"{dates.isna().sum()} datas em falta: não é possível "
                             "atribuir essas linhas a uma partição.")

        months = dates.dt.to_period("M")
        labels = pd.Series(pd.NA, index=dates.index, dtype="object")
        for name, (start, end) in self.partitions.items():
            labels[(months >= start) & (months <= end)] = name

        outside = labels.isna()
        if outside.any():
            raise ValueError(
                f"{outside.sum()} datas fora das partições (entre {months[outside].min()} "
                f"e {months[outside].max()})."
            )
        return labels.astype(pd.CategoricalDtype(list(self.partitions), ordered=True))

    def split(self, df, dates, names=None):
        """
        Separa as linhas de df pelas partições

        dates pode ter só parte das linhas de df (ex.: sem as linhas já removidas): só
        essas linhas são distribuídas.

        Parameters:
            df (pd.DataFrame): Dados a separar
            dates (pd.Series): Datas das linhas a distribuir; o índice tem de existir em df
            names (list[str] | None): Partições a devolver. Por omissão, todas.
                Pedir só as necessárias evita copiar para a memória as outras.

        Returns:
            dict[str, pd.DataFrame]: Nome -> linhas dessa partição (cópia)
        """
        if not dates.index.isin(df.index).all():
            raise ValueError("Há índices em dates que não existem em df.")

        names = list(self.partitions) if names is None else names
        unknown = set(names) - set(self.partitions)
        if unknown:
            raise ValueError(f"Partições desconhecidas: {sorted(unknown)}")

        labels = self.assign(dates)
        return {name: df.loc[labels.index[labels == name]] for name in names}

    def summary(self, dates):
        """
        Tabela com o período, o número de linhas e a percentagem de cada partição

        Parameters:
            dates (pd.Series): Datas (datetime64), uma por linha

        Returns:
            pd.DataFrame: Uma linha por partição, pela ordem cronológica
        """
        counts = self.assign(dates).value_counts(sort=False)
        return pd.DataFrame({
            "start": [start.strftime("%Y-%m") for start, _ in self.partitions.values()],
            "end": [end.strftime("%Y-%m") for _, end in self.partitions.values()],
            "rows": counts.to_numpy(),
            "pct": (100 * counts / counts.sum()).round(2).to_numpy(),
        }, index=pd.Index(list(self.partitions), name="partition"))

    # Métodos Privados

    def _validate(self):
        """Confirma que os períodos estão ordenados, sem sobreposições nem buracos"""
        if not self.partitions:
            raise ValueError("É preciso pelo menos uma partição.")

        previous_end = None
        for name, (start, end) in self.partitions.items():
            if start > end:
                raise ValueError(f"Partição {name!r}: o início é depois do fim.")
            if previous_end is not None and start != previous_end + 1:
                raise ValueError(
                    f"Partição {name!r} começa em {start}, mas a anterior acaba em "
                    f"{previous_end}: os períodos têm de ser seguidos, sem buracos "
                    "nem sobreposições."
                )
            previous_end = end
