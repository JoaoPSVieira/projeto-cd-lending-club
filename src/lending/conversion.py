"""
Conversão dos CSV originais do LendingClub para pickle

Porquê converter:
    Ler os CSV demora muito tempo porque o pandas tem de interpretar
    milhões de linhas de texto. O pickle guarda o DataFrame já em formato binário,
    ou seja, passa a demorar segundos para fazer o mesmo.

Decisões de desenho:
1 - Todas as colunas são lidas como string:
    Esta etapa muda só o formato do ficheiro de CSV para pickle, não interpreta valores
    A conversão de tipos é limpeza e fica para a Phase 2

2 - Só campos vazios contam como "valor em falta":
    Por omissão, o pandas transforma em NaN textos como "NA", "n/a", "null" ou "None"
    Isso é uma decisão de limpeza, por isso aqui só o campo vazio passa a NaN e o resto
    fica como está no ficheiro original

3 - Texto guardado em Arrow (pyarrow):
    O pandas pode guardar texto como objetos Python (que é muito mais pesado)
    ou em Arrow, que é mais compacto.
"""

import platform
import time
from dataclasses import dataclass, fields
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lending.storage import PickleStore


def text_dtype():
    """
    Tipo usado para TODAS as colunas: texto do pandas guardado em Arrow.

    na_value=np.nan faz o valor em falta comportar-se como o NaN habitual do pandas

    Raises:
        ImportError: Se o pyarrow não estiver instalado.
    """
    try:
        return pd.StringDtype(storage="pyarrow", na_value=np.nan)
    except ImportError as e:
        raise ImportError(
            "O pyarrow é necessário para guardar o texto de forma compacta. "
            "Instala com: python -m pip install pyarrow"
        ) from e


class CsvToPickleConverter:
    """
    Converte um CSV (.csv / .csv.gz) num pickle com um único DataFrame.

    O CSV é lido por blocos (chunksize) para controlar a memória durante a leitura
    e para mostrar progresso. No fim, os blocos são juntos num DataFrame e guardados
    com PickleStore.

    Attributes:
        csv_path (Path): CSV de origem (.csv ou .csv.gz - o pandas descomprime sozinho)
        pickle_path (Path): Pickle destino.
        chunksize (int): Número de linhas lidas por bloco
        encoding (str): Codificação do ficheiro de texto
        progress_every (int): Mostra progresso a cada N blocos

    Methods:
        convert(overwrite=False): Faz a conversão e devolve o caminho do pickle.
    """

    def __init__(self, csv_path, pickle_path, chunksize=100_000, encoding="utf-8",
                 progress_every=5):
        """
        Inicializa o conversor (ainda não faz leitura)

        Parameters:
            csv_path (Path): CSV de origem (.csv ou .csv.gz - o pandas descomprime sozinho)
            pickle_path (Path): Pickle destino.
            chunksize (int): Número de linhas lidas por bloco
            encoding (str): Codificação do ficheiro de texto
            progress_every (int): Mostra progresso a cada N blocos
        """
        self.csv_path = Path(csv_path)
        self.pickle_path = Path(pickle_path)
        self.chunksize = chunksize
        self.encoding = encoding
        self.progress_every = progress_every

        # Falha já aqui (e não a meio da leitura) se faltar o pyarrow.
        self._dtype = text_dtype()

    def convert(self, overwrite=False):
        """
        Lê o CSV por blocos, junta tudo num DataFrame de texto e guarda em pickle

        Parameters:
            overwrite (bool): Se False e o pickle já existir, não refaz a conversão.

        Returns:
            Path: Caminho do pickle
        """
        if self.pickle_path.exists() and not overwrite:
            print(f"{self.pickle_path.name} já existe - conversão saltada. "
                  "Para refazer: convert(overwrite=True).")
            return self.pickle_path

        self._check_source()
        header = self._read_header()
        print(f"A converter {self.csv_path.name} ({len(header)} colunas)...")

        start = time.perf_counter()
        df = self._read_in_chunks(start)
        self._validate(df, header)
        read_seconds = time.perf_counter() - start

        print(f"A guardar {self.pickle_path.name}...")
        PickleStore.save(df, self.pickle_path)
        total_seconds = time.perf_counter() - start

        print(f"Concluído: {len(df):,} linhas x {df.shape[1]} colunas | "
              f"leitura {read_seconds:.0f}s, total {total_seconds:.0f}s | "
              f"{_megabytes(self.pickle_path):,.0f} MB em disco")
        return self.pickle_path

    # Métodos Privados

    def _check_source(self):
        """Confirma que o CSV existe e é um ficheiro"""
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Não encontrei {self.csv_path}. Coloca o ficheiro do Kaggle em data/raw/."
            )
        if not self.csv_path.is_file():
            raise IsADirectoryError(
                f"{self.csv_path} é uma pasta, não um ficheiro. "
                "Aponta para o ficheiro .csv/.csv.gz que está lá dentro."
            )

    def _read_header(self):
        """
        Lê só a primeira linha do CSV (nomes das colunas), sem carregar qualquer dado

        Returns:
            list[str]: Nomes das colunas pela ordem do ficheiro
        """
        return pd.read_csv(self.csv_path, nrows=0, encoding=self.encoding).columns.tolist()

    def _read_in_chunks(self, start):
        """
        Lê o CSV bloco a bloco e devolve um único DataFrame.

        Parameters:
            start (float): Instante de início para o progresso.

        Returns:
            pd.DataFrame: Todas as linhas, todas as colunas como texto.
        """
        chunks = []
        rows = 0

        # read_csv com chunksize devolve um leitor: cada iteração entrega um bloco
        with pd.read_csv(
            self.csv_path,
            dtype=self._dtype,         # todas as colunas como texto
            keep_default_na=False,     # não converter "NA", "n/a", "null"
            na_values=[""],            # só o campo vazio é valor em falta
            chunksize=self.chunksize,
            encoding=self.encoding,
        ) as reader:
            for i, chunk in enumerate(reader, start=1):
                chunks.append(chunk)
                rows += len(chunk)
                if i % self.progress_every == 0:
                    print(f"  bloco {i}: {rows:,} linhas lidas "
                          f"({time.perf_counter() - start:.0f}s)")

        if not chunks:
            raise ValueError(f"{self.csv_path.name} não tem linhas de dados.")

        # Com texto em Arrow, o concat não copia os dados, cada bloco passa a ser
        # um pedaço da coluna final.
        # ignore_index=True: índice 0..n-1 contínuo
        return pd.concat(chunks, ignore_index=True)

    def _validate(self, df, header):
        """
        Verificações de higiene antes de gravar

        Raises:
            ValueError: Se as colunas lidas não forem as do cabeçalho, pela mesma ordem
            TypeError: Se alguma coluna não ficou como texto
        """
        if list(df.columns) != header:
            raise ValueError("As colunas lidas não coincidem com o cabeçalho do CSV.")

        not_text = [c for c in df.columns if df[c].dtype != self._dtype]
        if not_text:
            raise TypeError(f"Colunas que não ficaram como texto: {not_text}")


@dataclass(frozen=True)
class DatasetSummary:
    """
    Números de um ficheiro de origem, para o relatório

    Attributes:
        name (str): Nome curto do ficheiro ("accepted" / "rejected")
        n_rows (int): Número de linhas lidas
        n_columns (int): Número de colunas
        csv_mb (float): Tamanho do CSV de origem (MB)
        pickle_mb (float): Tamanho do pickle (MB)
        memory_mb (float): Memória ocupada pelo DataFrame (MB)
        load_seconds (float): Tempo a abrir o pickle (s)
    """

    name: str
    n_rows: int
    n_columns: int
    csv_mb: float
    pickle_mb: float
    memory_mb: float
    load_seconds: float

    @classmethod
    def from_dataframe(cls, name, df, csv_path, pickle_path, load_seconds):
        """
        Cria o resumo a partir de um DataFrame já carregado

        Parameters:
            name (str): Nome curto do ficheiro
            df (pd.DataFrame): DataFrame lido do pickle
            csv_path (str | Path): CSV de origem
            pickle_path (str | Path): Pickle correspondente
            load_seconds (float): Tempo medido a abrir o pickle

        Returns:
            DatasetSummary
        """
        return cls(
            name=name,
            n_rows=len(df),
            n_columns=df.shape[1],
            csv_mb=_megabytes(csv_path),
            pickle_mb=_megabytes(pickle_path),
            # deep=True mede o texto em si, não só as referências
            memory_mb=float(df.memory_usage(deep=True).sum()) / 1e6,
            load_seconds=load_seconds,
        )

    def __str__(self):
        """Uma linha legível para imprimir no notebook."""
        return (f"{self.name}: {self.n_rows:,} linhas x {self.n_columns} colunas | "
                f"CSV {self.csv_mb:,.0f} MB -> pickle {self.pickle_mb:,.0f} MB | "
                f"memória {self.memory_mb:,.0f} MB | abriu em {self.load_seconds:.1f}s")

    def as_dict(self):
        """Devolve o resumo como dicionário (útil para montar uma tabela pandas)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @staticmethod
    def write_markdown(summaries, path, download_date=None):
        """
        Escreve um ficheiro Markdown com a tabela dos ficheiros de origem e as versões usadas

        Parameters:
            summaries (list[DatasetSummary]): Resumos a incluir
            path (str | Path): Ficheiro de destino (ex: docs/dados_origem.md)
            download_date (str | None): Data do download do Kaggle (ex: "2026-10-14")

        Returns:
            Path: Caminho do ficheiro escrito
        """
        import pyarrow  # import local: só para registar a versão

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Dados de origem",
            "",
            "_Gerado automaticamente por `notebooks/00_conversao_dados.ipynb` - não editar à mão._",
            "",
            "- Fonte: Kaggle, *Lending Club Loan Data* (wordsforthewise/lending-club)",
            f"- Data do download: {download_date or 'por preencher'}",
            f"- Gerado em: {date.today().isoformat()}",
            f"- Versões: Python {platform.python_version()}, pandas {pd.__version__}, "
            f"pyarrow {pyarrow.__version__}, numpy {np.__version__}",
            "- Conversão: todas as colunas lidas como texto; só campos vazios contam como em falta.",
            "",
            "| Ficheiro | Linhas | Colunas | CSV (MB) | Pickle (MB) | Memória (MB) | Abrir pickle (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for s in summaries:
            lines.append(
                f"| {s.name} | {s.n_rows:,} | {s.n_columns} | {s.csv_mb:,.0f} | "
                f"{s.pickle_mb:,.0f} | {s.memory_mb:,.0f} | {s.load_seconds:.1f} |"
            )

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _megabytes(path):
    """Tamanho de um ficheiro em MB"""
    return Path(path).stat().st_size / 1e6
