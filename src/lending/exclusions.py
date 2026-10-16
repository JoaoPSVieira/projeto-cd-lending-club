"""
Registo das linhas removidas ou separadas em cada passo do pré-processamento

Motivo:
    O enunciado pede para documentar todas as exclusões. Em vez de escrever os números
    à mão no fim, cada passo regista-se no momento em que é feito. A tabela final
    (docs e relatório) é gerada a partir deste registo.

Decisões de desenho:
1 - Cada passo tem um tipo:
    "remoção": linhas que deixam de existir (ex.: rodapés que não são empréstimos)
    "partição": linhas que continuam válidas mas são guardadas à parte (ex.: teste final)

2 - O registo confirma a continuidade:
    As linhas antes de um passo têm de ser as linhas depois do passo anterior. Se não
    forem, houve uma remoção que não foi registada e o código dá erro.
"""

from dataclasses import asdict, dataclass

import pandas as pd

KINDS = ("remoção", "partição")


@dataclass(frozen=True)
class ExclusionStep:
    """
    Um passo do registo

    Attributes:
        step (str): Nome curto do passo
        kind (str): "remoção" ou "partição"
        rule (str): Regra aplicada, escrita de forma verificável
        reason (str): Justificação
        rows_before (int): Linhas antes do passo
        rows_removed (int): Linhas removidas ou separadas
        rows_after (int): Linhas depois do passo
    """

    step: str
    kind: str
    rule: str
    reason: str
    rows_before: int
    rows_removed: int
    rows_after: int


class ExclusionLog:
    """
    Lista ordenada dos passos que reduzem um conjunto de dados

    Attributes:
        dataset (str): Nome do conjunto a que o registo se refere
        steps (list[ExclusionStep]): Passos pela ordem em que foram feitos

    Methods:
        apply(df, remove, step, kind, rule, reason): Regista o passo e devolve as linhas que ficam
        record(step, kind, rule, reason, rows_before, rows_removed): Só regista o passo
        to_frame(): Devolve o registo como tabela pandas
    """

    def __init__(self, dataset):
        """
        Parameters:
            dataset (str): Nome do conjunto (ex.: "accepted -> desenvolvimento")
        """
        self.dataset = dataset
        self.steps = []

    def apply(self, df, remove, step, kind, rule, reason):
        """
        Regista o passo e devolve o DataFrame sem as linhas indicadas

        Parameters:
            df (pd.DataFrame): Dados antes do passo
            remove (pd.Series): Máscara booleana, True nas linhas a remover/separar
            step, kind, rule, reason (str): Ver ExclusionStep

        Returns:
            pd.DataFrame: As linhas que ficam (cópia)
        """
        if not remove.index.equals(df.index):
            raise ValueError("A máscara tem de ter o mesmo índice que o DataFrame.")

        self.record(step, kind, rule, reason,
                    rows_before=len(df), rows_removed=int(remove.sum()))
        return df.loc[~remove]

    def record(self, step, kind, rule, reason, rows_before, rows_removed):
        """
        Regista um passo sem filtrar dados

        Útil quando filtrar o DataFrame inteiro seria uma cópia demasiado grande para a
        memória e as linhas que ficam são selecionadas de outra forma.

        Parameters:
            step, kind, rule, reason (str): Ver ExclusionStep
            rows_before (int): Linhas antes do passo
            rows_removed (int): Linhas removidas ou separadas

        Returns:
            ExclusionStep: O passo registado
        """
        if kind not in KINDS:
            raise ValueError(f"kind tem de ser um de {KINDS}, não {kind!r}.")
        if not 0 <= rows_removed <= rows_before:
            raise ValueError("rows_removed tem de estar entre 0 e rows_before.")
        if self.steps and rows_before != self.steps[-1].rows_after:
            raise ValueError(
                f"O passo {step!r} começa com {rows_before:,} linhas, mas o passo anterior "
                f"terminou com {self.steps[-1].rows_after:,}. Há uma remoção por registar."
            )

        entry = ExclusionStep(step, kind, rule, reason, rows_before, rows_removed,
                              rows_before - rows_removed)
        self.steps.append(entry)
        return entry

    @property
    def rows_after(self):
        """Linhas que ficam depois do último passo (None se o registo estiver vazio)."""
        return self.steps[-1].rows_after if self.steps else None

    def to_frame(self):
        """
        Returns:
            pd.DataFrame: Uma linha por passo, numerada a partir de 1
        """
        frame = pd.DataFrame([asdict(s) for s in self.steps],
                             columns=list(ExclusionStep.__dataclass_fields__))
        frame.index = pd.RangeIndex(1, len(frame) + 1, name="n")
        return frame

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return f"ExclusionLog({self.dataset!r}, {len(self)} passos, {self.rows_after} linhas)"
