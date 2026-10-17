"""
Estilo das figuras do projeto

O porquê um módulo só para isto:
    Todas as figuras ficam com as mesmas características de cores, letra,
    grelha, etc, sem repetir

Rsegras de cor usada:
    - CATEGORICAL: identidade (ex: classe Good vs Default). Usamos 3 cores num
        gráfico de pontos
    - SEQUENTIAL_CMAP: magnitude só tem um tom, de claro a escuro
    - DIVERGING_CMAP: sinal -> azul / cinzento / vermelho,
        com o cinzento neutro no zero
    - Uma única série usa sempre a cor 1 (neste caso azul)
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Superfície e tinta (texto, eixos, grelha)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Cores categóricas, por esta ordem (1: azul, 2: laranja, 3: verde-azulado)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
SERIES_1 = CATEGORICAL[0]

# Sequencial (azul, de claro para escuro)
_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
               "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("lending_seq", _BLUE_STEPS)

# Divergente (azul <-> cinzento <-> vermelho)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "lending_div", ["#256abf", "#f0efec", "#e34948"]
)


def apply_style():
    """Aplica o estilo do projeto ao matplotlib"""

    mpl.rcParams.update({
        # Letra com tamanhos legíveis no PDF
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        # Cores do texto e dos eixos
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,

        # Superfície
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,

        # Grelha com linha fina e contínua
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,

        # Sem caixa à volta do gráfico
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Linhas finas
        "lines.linewidth": 2,
        "patch.linewidth": 0,

        # Ciclo de cores
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),

        # Exportação para o relatório
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
