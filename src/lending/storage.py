"""
Guardar e ler objetos em pickle

Notas de arquitetura:
1. O save() escreve primeiro para "<ficheiro>.tmp" e só no fim renomeia para o nome final.
   Se a escrita for interrompida por erro, falta de memória, PC desligado,
   ou outro motivo, não fica um .pkl meio escrito com o nome certo,
   que mais tarde parecesse válido.

2. load() pode confirmar que o ficheiro contém o tipo esperado (ex: DataFrame).

Avisos importantes:
- Segurança: pickle.load executa código definido dentro do ficheiro por isso é necessário
    tomar cuidado com o código a executar
- Compatibilidade: um pickle depende das versões das bibliotecas, ou seja, é recomendado
    usar as mesmas versões do requirements.txt
"""

import pickle
import warnings
from pathlib import Path

import pandas as pd


class PickleStore:
    """
    Guarda e lê objetos Python em ficheiros pickle.

    Os métodos são estáticos e recebem sempre o caminho.

    Methods:
        save(obj, path): Guarda obj em path, com escrita atómica.
        load(path, expected_type=None): Lê o objeto de path e, opcionalmente, valida o tipo.
    """

    # Suporta objetos com mais de 4 GB e é o mais eficiente.
    PROTOCOL = pickle.HIGHEST_PROTOCOL

    @staticmethod
    def save(obj, path):
        """
        Guarda um objeto em pickle

        Parameters:
            obj: Objeto a guardar (DataFrame).
            path (str | Path): Caminho do ficheiro .pkl destino.

        Returns:
            Path: Caminho do ficheiro
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Ficheiro temporário
        tmp_path = path.with_name(path.name + ".tmp")

        try:
            with open(tmp_path, "wb") as f:
                pickler = pickle.Pickler(f, protocol=PickleStore.PROTOCOL)
                pickler.dump(obj)
            # Só chega aqui se a escrita terminou sem erros
            tmp_path.replace(path)
        except BaseException:
            # Qualquer falha apaga o temporário.
            tmp_path.unlink(missing_ok=True)
            raise

        return path

    @staticmethod
    def load(path, expected_type=None):
        """
        Lê um objeto de um ficheiro pickle

        Parameters:
            path (str | Path): Caminho do ficheiro .pkl
            expected_type (type | None): Se indicado, confirma que o objeto lido é deste tipo

        Returns:
            O objeto guardado no ficheiro

        Raises:
            FileNotFoundError: Se o ficheiro não existir
            TypeError: Se o objeto lido não for do tipo esperado
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Não existe o ficheiro {path}. "
                "Corre primeiro o notebook 00_conversao_dados.ipynb!"
            )

        with open(path, "rb") as f:
            obj = pickle.load(f)

        if expected_type is not None and not isinstance(obj, expected_type):
            raise TypeError(
                f"{path.name} contém um {type(obj).__name__}, "
                f"mas era esperado um {expected_type.__name__}."
            )
        return obj