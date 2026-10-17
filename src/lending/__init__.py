"""
lending - codebase do projeto de Ciência de Dados (LendingClub, 2007–2018).

Os notebooks importam daqui as classes; a lógica não fica escrita nos notebooks.

Módulos:
    config      Caminhos do projeto e constantes partilhadas (seed, colunas esperadas).
    storage     Guardar e ler objetos em pickle (PickleStore).
    conversion  Etapa 0: conversão dos CSV originais para pickle e resumo dos ficheiros.
    sampling    Amostras aleatórias reprodutíveis (nunca as primeiras linhas).
    plot_style  Estilo e cores comuns a todas as figuras do relatório.
    eda         Fase 2 (EDA): tabelas descritivas (DescriptiveAnalysis) e gráficos (EDAPlotter).
    dimred      Fase 2 (EDA): redução de dimensionalidade, PCA e UMAP (DimensionalityReducer).
"""
