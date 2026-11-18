"""
lending - codebase do projeto de Ciência de Dados (LendingClub, 2007–2018).

Os notebooks importam daqui as classes; a lógica não fica escrita nos notebooks.

Módulos:
    config        Caminhos, partições temporais e constantes partilhadas.
    storage       Guardar e ler objetos em pickle (PickleStore).
    conversion    Etapa 0: conversão dos CSV originais para pickle e resumo dos ficheiros.

    Fase 2 - pré-processamento
    parsing       Conversões determinísticas de texto (datas, %, prazo, anos de emprego).
    schema        Tipo e unidade de cada coluna do accepted (SchemaParser, SchemaReport).
    cleaning      Regras de limpeza: linhas que não são empréstimos, condições modificadas.
    cleaner       AcceptedCleaner: aplica ao teste as mesmas regras do desenvolvimento.
    quality       Valores em falta, duplicados, mutuários repetidos, verificações de consistência.
    rejected      RejectedCleaner: tipos, nomes e limpeza do ficheiro de rejeitados.
    partition     TemporalPartitioner: cortes de calendário explícitos.
    cohorts       CohortBuilder: coorte de cada tarefa e alvo da classificação.
    exclusions    ExclusionLog: registo de todas as linhas removidas ou separadas.
    features      TaskFeatures: colunas de entrada de cada tarefa (tabela de elegibilidade).
    preprocessing Pipeline com fit só no treino e validação com janela expansível.

    Fase 2 - análise exploratória
    sampling      Amostras aleatórias reprodutíveis (nunca as primeiras linhas).
    plot_style    Estilo e cores comuns a todas as figuras do relatório.
    eda           Tabelas descritivas (DescriptiveAnalysis) e gráficos (EDAPlotter).
    dimred        Redução de dimensionalidade: PCA e UMAP (DimensionalityReducer).

    Fase 2 - testes de hipóteses
    hypotheses    Tabelas em CSV para os testes em R (HypothesisDataExporter).

    Fase 3 - seleção de modelos
    feature_engineering  Features novas: fichas (FEATURES) e FeatureBuilder (entra no pipeline).

    Fase 4 -
    experiment    Validação cruzada temporal de todos os modelos (CrossValidationRunner, ModelSpec).
    evaluation    Métricas, limiar da classificação, matriz de confusão e bootstrap.
    baselines     Baselines: mediana por prazo e taxa de default do grau.
    boosting      LightGBM com paragem antecipada nas linhas mais recentes.
    model_catalog Modelos e grelhas da secção 3 (regression_specs, classification_specs).
    knn           kNN de raiz em NumPy (NeighborSearch, KNNRegressor, KNNClassifier).

"""