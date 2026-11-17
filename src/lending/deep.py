"""
Redes neuronais densas (MLP) em Keras/TensorFlow, com a interface do scikit-learn

Porque é que a rede está "embrulhada" (wrapper) numa classe do scikit-learn:
    Todos os modelos (kNN, Ridge, árvores, Random Forest, LightGBM e a MLP) passam pelo
    mesmo Pipeline([FeatureBuilder, build_preprocessor, modelo]) e pelo mesmo runner da
    validação cruzada. Para isso, a MLP tem de se comportar como um estimador do
    scikit-learn: hiperparâmetros só no __init__ (get_params/set_params e clone funcionam),
    fit(X, y) e predict(X). A rede Keras só é criada dentro do fit, por isso cada fold
    começa de uma rede nova e nunca reaproveita pesos de outro fold.

Arquitetura (justificada na secção 3 do relatório):
    Entrada (colunas do pré-processamento) -> [Dense(n, ReLU) -> Dropout] x camadas -> saída
    - Dados tabulares, sem estrutura espacial nem sequencial: camadas densas chegam.
    - ReLU nas camadas escondidas: não satura para valores positivos, o que evita o
      gradiente evanescente das sigmoides em redes com várias camadas.
    - Dropout: desliga ao acaso uma fração dos neurónios em cada passo do treino, o que
      reduz o sobreajustamento (a rede não pode depender de um neurónio em particular).
    - Adam: descida do gradiente com taxa de aprendizagem adaptada a cada peso.
    - Grelha prevista: camadas (64,), (128, 64), (256, 128, 64) x 2 taxas de aprendizagem.

Paragem antecipada temporal:
    A rede treina no máximo max_epochs épocas (passagens completas pelo treino). No fim de
    cada época mede-se a perda num conjunto de validação interno; se não melhorar durante
    `patience` épocas, o treino pára e ficam os pesos da melhor época. O conjunto interno
    são as ÚLTIMAS linhas do treino do fold (validation_split do Keras corta o fim dos
    arrays antes de baralhar). Como os dados chegam ordenados por issue_d, são os
    empréstimos mais recentes: a paragem nunca espreita o futuro de dentro do fold.

Reprodutibilidade:
    keras.utils.set_random_seed fixa as sementes do Python, do NumPy e do TensorFlow no
    início de cada fit (pesos iniciais, dropout e baralhar dos lotes). Em CPU, a ordem das
    somas pode dar diferenças mínimas de arredondamento entre máquinas.
"""

import logging
import os

# Esconde as mensagens informativas do TensorFlow (só ficam avisos e erros)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Cada fold cria uma rede nova e o TensorFlow avisa que "recompila" a função de previsão
# (retracing). É esperado e inofensivo; sem isto as grelhas enchiam o output de avisos.
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import keras  # noqa: E402  (tem de vir depois da variável de ambiente)
import numpy as np  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402

from lending import config  # noqa: E402

# Pedidos por lote no predict (só afeta a memória, não o resultado)
PREDICT_BATCH_SIZE = 8192


class _KerasMLP(BaseEstimator):
    """
    Parte comum da MLP de regressão e da de classificação (não usar diretamente)

    Attributes:
        hidden_layers (tuple[int, ...]): Neurónios de cada camada escondida, ex.: (128, 64)
        dropout (float): Fração de neurónios desligados em cada passo do treino (0 = sem)
        learning_rate (float): Taxa de aprendizagem do Adam
        batch_size (int): Empréstimos por lote (cada lote dá uma atualização dos pesos)
        max_epochs (int): Máximo de épocas (o orçamento da secção 3 é 50)
        patience (int): Épocas sem melhoria na validação interna antes de parar
        validation_fraction (float): Fração final do treino usada na paragem antecipada
        random_state (int): Semente
        verbose (int): 0 = silencioso, 1 = barra de progresso, 2 = uma linha por época

    Atributos aprendidos no fit:
        model_ (keras.Model): A rede treinada
        history_ (dict[str, list[float]]): Perda no treino e na validação interna por época
        best_epoch_ (int): Época com a menor perda na validação interna (pesos guardados)
        n_epochs_ (int): Épocas efetivamente corridas
        n_features_in_ (int): Número de colunas de entrada
    """

    def __init__(self, hidden_layers=(128, 64), dropout=0.2, learning_rate=1e-3,
                 batch_size=512, max_epochs=50, patience=5, validation_fraction=0.15,
                 random_state=config.RANDOM_SEED, verbose=0):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    @staticmethod
    def _as_array(X):
        """Converte X (DataFrame do pré-processamento) numa matriz float32 sem falhas."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X tem de ter 2 dimensões; tem {X.ndim}.")
        if not np.isfinite(X).all():
            raise ValueError("A MLP não aceita valores em falta nem infinitos: "
                             "passar primeiro pelo build_preprocessor.")
        return X

    def _build(self, n_features, output_activation):
        """Cria a rede (ainda sem pesos treinados)."""
        model = keras.Sequential(name=type(self).__name__)
        model.add(keras.Input(shape=(n_features,)))
        for units in self.hidden_layers:
            model.add(keras.layers.Dense(units, activation="relu"))
            if self.dropout > 0:
                model.add(keras.layers.Dropout(self.dropout))
        model.add(keras.layers.Dense(1, activation=output_activation))
        return model

    def _fit_network(self, X, y, loss, output_activation, class_weight=None):
        """
        Treina uma rede nova com paragem antecipada nas últimas linhas do treino

        Parameters:
            X (np.ndarray): Matriz float32 já validada, pela ordem temporal
            y (np.ndarray): Alvo já preparado pela subclasse
            loss (str): Função de perda do Keras ("mse", "huber", "binary_crossentropy")
            output_activation (str): "linear" (regressão) ou "sigmoid" (classificação)
            class_weight (dict | None): Peso de cada classe na perda

        Returns:
            self
        """
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction tem de estar entre 0 e 1.")
        keras.utils.set_random_seed(self.random_state)
        self.n_features_in_ = X.shape[1]
        self.model_ = self._build(self.n_features_in_, output_activation)
        self.model_.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                            loss=loss)
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience,
                                             restore_best_weights=True)
        history = self.model_.fit(X, y, batch_size=self.batch_size, epochs=self.max_epochs,
                                  validation_split=self.validation_fraction, shuffle=True,
                                  class_weight=class_weight, callbacks=[stop],
                                  verbose=self.verbose)
        self.history_ = {k: [float(v) for v in values] for k, values in history.history.items()}
        self.n_epochs_ = len(self.history_["loss"])
        self.best_epoch_ = int(np.argmin(self.history_["val_loss"])) + 1
        return self

    def _raw_predict(self, X):
        """Saída da rede (antes de qualquer conversão da subclasse), como vetor."""
        check_is_fitted(self, "model_")
        X = self._as_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X tem {X.shape[1]} colunas; a rede foi treinada com "
                             f"{self.n_features_in_}.")
        return self.model_.predict(X, batch_size=PREDICT_BATCH_SIZE, verbose=0).ravel()


class KerasMLPRegressor(RegressorMixin, _KerasMLP):
    """
    MLP de regressão: saída linear (uma unidade), perda MSE por omissão

    O alvo é padronizado dentro do fit (média e desvio-padrão do treino do fold) e o
    predict desfaz a padronização, por isso as previsões saem nas unidades originais
    (pontos percentuais da int_rate). Com o alvo à volta de 0 e desvio 1, a saída linear
    começa perto da escala certa e o Adam converge mais depressa. A média e o desvio são
    aprendidos só no treino, como os scalers do pré-processamento.

    Attributes:
        loss (str): "mse" (erro quadrático médio) ou "huber" (quadrático perto de 0 e linear
            nos erros grandes, menos sensível a taxas extremas)
        (os restantes estão em _KerasMLP)

    Atributos aprendidos no fit (além dos de _KerasMLP):
        y_mean_ (float), y_std_ (float): Média e desvio-padrão do alvo no treino

    Exemplo:
        pipe = Pipeline([("features", FeatureBuilder("regression")),
                         ("pre", build_preprocessor(extended)),
                         ("model", KerasMLPRegressor(hidden_layers=(128, 64)))])
        pipe.fit(X_train, y_train)          # X_train ordenado por issue_d
        pred = pipe.predict(X_valid)        # em pontos percentuais
    """

    def __init__(self, hidden_layers=(128, 64), dropout=0.2, learning_rate=1e-3,
                 batch_size=512, max_epochs=50, patience=5, validation_fraction=0.15,
                 random_state=config.RANDOM_SEED, verbose=0, loss="mse"):
        super().__init__(hidden_layers=hidden_layers, dropout=dropout,
                         learning_rate=learning_rate, batch_size=batch_size,
                         max_epochs=max_epochs, patience=patience,
                         validation_fraction=validation_fraction,
                         random_state=random_state, verbose=verbose)
        self.loss = loss

    def fit(self, X, y):
        """
        Parameters:
            X (pd.DataFrame | np.ndarray): Saída do build_preprocessor, ordenada por issue_d
            y (pd.Series | np.ndarray): Alvo (int_rate em pontos percentuais)

        Returns:
            self
        """
        if self.loss not in ("mse", "huber"):
            raise ValueError(f"loss tem de ser 'mse' ou 'huber'; recebeu {self.loss!r}.")
        X = self._as_array(X)
        y = np.asarray(y, dtype=np.float32).ravel()
        if len(y) != len(X):
            raise ValueError(f"X tem {len(X)} linhas e y tem {len(y)}.")
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        return self._fit_network(X, (y - self.y_mean_) / self.y_std_, loss=self.loss,
                                 output_activation="linear")

    def predict(self, X):
        """
        Returns:
            np.ndarray: Previsões nas unidades originais do alvo
        """
        return self._raw_predict(X) * self.y_std_ + self.y_mean_
