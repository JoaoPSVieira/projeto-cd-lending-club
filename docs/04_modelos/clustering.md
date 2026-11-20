# Clustering de mutuários (Fase 4 e 5)

Bloco 6 do plano da Parte 2. Agrupa os 466 124 empréstimos aceites de 2007-06 a 2014-12 pelo
perfil de crédito no momento do pedido, compara três algoritmos (k-means, mistura de gaussianas
e DBSCAN), escolhe o número de grupos, avalia a estabilidade e dá significado de negócio aos
grupos com a taxa de juro e a taxa de *default*, calculadas depois do ajuste.

| Ficheiro | Conteúdo |
|---|---|
| `notebooks/04d_clustering.ipynb` | O bloco inteiro, com os resultados gravados (8 secções) |
| `src/lending/clustering.py` | Variáveis e pré-processamento (`BorrowerProfiles`), medidas (`ClusterEvaluator`, `dunn_index`), grelhas (`kmeans_spec`, `gmm_spec`, `ClusterGrid`), concordância (`match_labels`), estabilidade, DBSCAN e risco dos grupos |
| `tests/test_clustering.py` | 58 testes |
| `reports/results/clustering_*.csv` | Resultados (secção 10 deste documento) |
| `reports/figures/mod_clu_*.png` | Figuras do relatório |
| `data/processed/clustering/borrowers/` | Grupo de cada empréstimo em cada configuração (fora do git) |

---

## 1. Requisitos do enunciado

| Requisito | Onde |
|---|---|
| *"Apply at least two clustering algorithms"* | k-means (04d, secção 2) e GMM (secção 3); DBSCAN como terceiro (secção 8) |
| *"Vary their relevant hyperparameters, including the number of clusters"* | k de 2 a 10 nos dois; covariância `full`/`diag` no GMM; `eps` x `min_samples` no DBSCAN |
| *"Describe the resulting groups and assess their stability and interpretability"* | Perfis, mapa de calor e PCA (secção 6); estabilidade entre sementes, metades disjuntas e períodos (secção 5); risco (secção 7) |
| *"For clustering, use appropriate internal measures alongside substantive interpretation"* | WSS (cotovelo), silhueta, Davies-Bouldin e Dunn (implementado de raiz), com a leitura de negócio |
| *"Prepare one comparison table for each task, including baselines"* | Secção 4: `clustering_comparison.csv`, com uma partição aleatória e os quartis do FICO como referências |
| *"Report training sample sizes, tuning procedures, and any computational limitations"* | Secção 3 deste documento: tamanhos, amostras da silhueta, do Dunn e do DBSCAN, tempos |
| *"Discuss ... selection bias, unequal follow-up, temporal change, and fairness"* | Secções 6 e 9 deste documento; 04d, secções 5 e 7 |
| Exemplo do enunciado: *"borrower clustering (FICO, DTI, credit age, utilization)"* | As 8 variáveis incluem as quatro (secção 2) |

---

## 2. Dados e variáveis

- **Coorte:** os 466 124 empréstimos aceites do desenvolvimento (2007-06 a 2014-12), sem
  filtros além da limpeza da Parte 1. Entram também os `Current`, porque o ajuste não usa o
  desfecho. O teste de 2015 nunca é usado: o clustering é descritivo.
- **Variáveis do ajuste** (`PROFILE_FEATURES`):

  | Variável | Porquê |
  |---|---|
  | `fico_range_low` | FICO (o `fico_range_high` é sempre `low + 4`; as duas fariam o FICO contar a dobrar) |
  | `dti` | Endividamento face ao rendimento |
  | `revol_util` | Utilização do crédito rotativo |
  | `annual_inc` | Rendimento (com log) |
  | `loan_to_income` | Esforço do pedido |
  | `revol_to_income` | Dívida de cartões face ao rendimento |
  | `open_acc_pct` | Parte do historial ainda aberta |
  | `credit_history_years` | Antiguidade do crédito |

- **Fora do ajuste:**
  - `ever_delinquent` e `has_public_record` só descrevem os grupos;
  - a taxa de juro, o grau e o desfecho servem para dar significado aos grupos (secção 7), não para os formar.
- **Antiguidade do crédito:** é calculada com o `issue_d` como data de referência. Na tabela
  de elegibilidade da Fase 1, o `issue_d` passou de ❌ a ⚠️ no clustering, só para este fim,
  e a `credit_history_years` passou a ser elegível no clustering. O clustering é descritivo e
  não tem momento de previsão.
- **Pré-processamento** (`BorrowerProfiles.pipeline()`):
  - os passos são os do pipeline da Parte 1: `FeatureBuilder` → log no rendimento → clipping
    aos percentis 0,5–99,5 → mediana → padronização;
  - a diferença é que **não há indicadores de "em falta"**. Uma coluna 0/1 rara, depois de
    padronizada, vale dezenas de desvios nas poucas linhas com 1 (com as 340 faltas do
    `revol_util`, cerca de 37) e dominaria a distância euclidiana;
  - o pipeline é ajustado em todos os empréstimos do clustering (não há teste).
- **Descritiva:**
  - as faltas são mínimas;
  - a maior correlação de Spearman entre variáveis é 0,41, por isso ficam as 8;
  - 79% dos empréstimos são de 2013–2014, e a população mudou no tempo (o DTI mediano subiu de 10 para 17,6).

---

## 3. Algoritmos, grelhas e medidas

| Algoritmo | Grelha | Ajuste | Tempo por configuração |
|---|---|---|---|
| k-means (`kmeans_spec`) | k de 2 a 10; `n_init=10`, semente 42 | 466 124 empréstimos | 1–6 s |
| GMM (`gmm_spec`) | k de 2 a 10 x covariância `full`/`diag`; `n_init=3` | 466 124 empréstimos | 3–97 s |
| DBSCAN (`dbscan_scan`) | `eps` de 0,8 a 2,0 x `min_samples` 8, 16, 32 | amostra de 20 000 | 1–8 s |

- **Medidas internas** (`ClusterEvaluator`), sempre nos mesmos pontos:
  - WSS e Davies-Bouldin em todos os empréstimos;
  - silhueta numa amostra de 20 000 e Dunn numa amostra de 5 000. São O(n²) e impossíveis com 466 124 pontos. A amostra só depende do número de linhas e da semente, por isso é a mesma em todas as configurações.
- **Índice de Dunn** (`dunn_index`):
  - menor distância entre grupos / maior diâmetro de um grupo;
  - implementado de raiz com NumPy (não existe no scikit-learn);
  - calculado por blocos de 1 000 linhas, para limitar a memória;
  - verificado contra a matriz de distâncias completa.
- **GMM:** também o BIC/AIC e a certeza da atribuição (média da probabilidade do grupo escolhido).
- **`ClusterGrid`:**
  - grava uma linha por configuração em `clustering_borrowers.csv` e o grupo de cada empréstimo num pickle;
  - é retomável como o runner da validação cruzada: o que já existe não volta a correr;
  - as análises fora da grelha (estabilidade, DBSCAN) usam `cached_csv`;
  - `RECALCULAR = True` no notebook apaga tudo e recalcula.

---

## 4. Resultados

**k-means** (secção 2 do 04d):

| k | Descida da WSS | Silhueta | Davies-Bouldin | Dunn |
|---|---|---|---|---|
| 2 | — | **0,151** | 2,17 | 0,051 |
| 3 | 10,9% | 0,145 | 1,98 | 0,045 |
| **4** | 9,9% | 0,142 | **1,78** | **0,059** |
| 5 | 5,6% | 0,133 | 1,89 | 0,042 |
| 10 | 3,1% | 0,113 | 1,77 | 0,038 |

**GMM** (secção 3):
- o BIC desce sempre com k, por isso não escolhe k;
- o `full` tem sempre um BIC ~4% mais baixo do que o `diag`;
- o k-means ganha nas medidas euclidianas em todos os k (elas favorecem grupos redondos);
- o Davies-Bouldin tem um mínimo local em k = 4 nos três modelos;
- todos os ajustes convergiram.

**Tabela de comparação** (secção 4, k = 4; ARI e concordância face ao k-means):

| Método | Silhueta | Davies-Bouldin | Dunn | ARI | Concordância |
|---|---|---|---|---|---|
| Partição aleatória (referência) | -0,002 | 813,8 | 0,026 | 0,00 | 26% |
| Quartis do FICO (referência) | 0,017 | 6,84 | 0,045 | 0,11 | 42% |
| **k-means** | **0,142** | **1,78** | **0,059** | 1 | 100% |
| GMM `diag` | 0,108 | 2,03 | 0,034 | 0,45 | 75% |
| GMM `full` | 0,087 | 2,24 | 0,036 | 0,24 | 60% |

- **A concordância usa `match_labels`.** O algoritmo húngaro emparelha os grupos dos dois agrupamentos e conta-se a % de empréstimos no grupo correspondente.
- **O k-means e o GMM `diag` concordam em k = 4 e 5** (75% e 78%), mas não em k = 3 (56%) nem a partir de k = 6.
- **Com k = 4, os três modelos encontram os mesmos quatro tipos de mutuário.**

---

## 5. Escolha: k-means com k = 4

Decidida por:
- **Medidas internas:** o cotovelo da WSS, o mínimo local do Davies-Bouldin nos três modelos e o máximo do Dunn no k-means.
- **Concordância entre algoritmos:** 75% com o GMM `diag`.
- **Estabilidade:** k = 4 é o k mais estável nos dois algoritmos (secção 6).

O k-means é a segmentação principal: tem as melhores medidas internas, é o mais estável e o que
resiste melhor à mudança de período, e um pedido novo recebe o grupo do centro mais próximo. O
GMM `diag` com k = 4 é a solução do segundo algoritmo.

**Leitura obrigatória:** a estrutura é fraca, e não há grupos naturais.
- A silhueta fica entre 0,11 e 0,15 (abaixo de 0,25).
- O Dunn fica perto do valor de uma partição aleatória.
- A PCA mostra uma só nuvem, e o DBSCAN não encontra zonas vazias (secção 8).

O k-means **segmenta um contínuo**. Os grupos valem por serem estáveis e por terem leitura de
negócio, não por estarem separados.

---

## 6. Estabilidade

Todas as comparações são feitas nos mesmos 466 124 empréstimos: cada modelo, ajustado de outra
forma, dá um grupo a todos, e comparam-se as atribuições.

| | ARI entre 5 sementes (média / pior) | ARI entre metades disjuntas (média / pior; 5 divisões) |
|---|---|---|
| k-means k = 4 | **0,991 / 0,984** | **0,979 / 0,970** |
| k-means k = 5 | 0,974 / 0,955 | 0,964 / 0,951 |
| k-means k = 6–10 | 0,96–0,98 | 0,90–0,95 (pior par 0,85) |
| GMM `diag` k = 4 | **0,999 / 0,999** | **0,971 / 0,966** |
| GMM `diag` k = 5 | 0,973 / 0,935 | 0,947 / 0,907 |

- **Com muitos dados, a partição reproduz-se em todos os k.** Isso mostra que é reprodutível,
  não que os grupos sejam naturais. A estabilidade é necessária, não suficiente.
- **Períodos.** Um modelo ajustado em 2007–2012 e outro em 2013–2014 concordam em 87% dos empréstimos com o
  k-means k = 4 (ARI 0,69), contra 99% entre metades da mesma população. O GMM concorda em 79%.
  A mudança temporal desloca as fronteiras, mas os quatro grupos existem nos dois períodos.
- **Composição por ano:** os grupos não são períodos; o que muda é a mistura. O grupo "FICO alto"
  passa de 34–36% em 2009–2011 para 14% em 2013–2014, e "Endividados" de 9,5% (2007) para 29%.

---

## 7. Os quatro grupos

Os nomes são dados por regra (o extremo de uma variável em cada grupo), verificada no notebook.
Por isso não dependem da numeração do k-means.

| Grupo | Peso | Perfil (medianas) | *Default* (IC 95%) | Juro médio | Grau A |
|---|---|---|---|---|---|
| **FICO alto** | 16% | FICO 740, utilização rotativa 24%, DTI 12,6% | **8,6%** [8,4; 8,9] | 10,2% | 50% |
| **Rendimento alto** | 32% | 85 000 USD, montante / rendimento 0,15, historial de 16,6 anos; 64% já tiveram atrasos | 12,6% [12,5; 12,8] | 14,1% | 12% |
| **Historial curto** | 24% | 9,6 anos de historial, 65% das contas abertas, 46 000 USD | 17,8% [17,6; 18,1] | 14,7% | 7% |
| **Endividados** | 28% | DTI 23,5%, montante / rendimento 0,32, saldo rotativo / rendimento 0,37 | **19,1%** [18,8; 19,3] | 14,9% | 9% |
| todos | | FICO 690, DTI 16,9%, 63 000 USD | 14,9% | 13,8% | 16% |

- **Desfecho observável** (`outcome_observed`): só os empréstimos cujo prazo terminou até ao
  corte comum de 31/12/2018. São os de 36 meses e os de 60 meses até 2013-12: 393 131
  empréstimos, 84%. O alvo é o da classificação (`make_target`).
- **A ordem do risco mantém-se dentro de cada ano de 2009 a 2014.** "Historial curto" e
  "Endividados" trocam entre si em 2014 e no GMM: têm um risco parecido.
- **A taxa de juro separa bem o grupo "FICO alto"**, mas pouco os outros três (14,1% a 14,9%,
  para *defaults* de 12,6% a 19,1%). Isto não ajusta o prazo: os "Endividados" têm 41% a 60 meses.
- **PCA:** PC1 (25%) é o endividamento face ao rendimento e PC2 (19%) a maturidade do crédito.
  Os grupos ocupam as quatro direções deste plano.

---

## 8. DBSCAN

- **Grelha:** amostra de 20 000 empréstimos; `eps` de 0,8 a 2,0 x `min_samples` 8, 16 e 32. Com
  `eps` grande, a procura de vizinhos nos 466 124 pontos não cabia em memória.
- **Resultado:** em nenhuma configuração aparece mais do que um grupo com tamanho relevante.
  - Com `eps` pequeno, a maioria dos pontos é ruído.
  - A partir de `eps` = 1,1, um único grupo tem 54% a 99,8%.
  - O segundo grupo nunca passa de 0,3%, e o ARI com o k-means é no máximo 0,06.
- **O gráfico k-distância não tem patamares.** Só tem joelho nos ~2% mais isolados.
- **Conclusão:** por um princípio diferente (densidade), confirma uma só nuvem, sem zonas vazias entre
  tipos de mutuário. Como segmentação, o DBSCAN não serve: dá um grupo e deixa empréstimos sem grupo.

---

## 9. Limitações

- **Estrutura fraca:** os grupos são segmentos de um contínuo, com fronteiras convencionais.
- **As medidas internas favorecem o k-means:** a silhueta, o Davies-Bouldin e a WSS são euclidianas e
  premeiam grupos redondos. Não são um juiz neutro entre algoritmos, por isso pesaram também a
  concordância e a estabilidade.
- **O Dunn é de pior caso** e calculado numa amostra de 5 000: quase não distingue estrutura de
  acaso nestes dados.
- ***Selection bias:*** só há empréstimos aceites. Os grupos descrevem a carteira da LendingClub,
  não todos os candidatos.
- ***Follow-up* desigual:** tratado na taxa de *default* com a regra de observabilidade (84% dos
  empréstimos).
- **Mudança temporal:** a mistura dos grupos mudou muito entre 2009–2011 e 2013–2014. Um modelo
  ajustado num período desloca as fronteiras do outro (87% de concordância).
- **Associação, não causalidade:** os grupos descrevem quem falhou mais. Não dizem que mudar o
  perfil mudaria o risco.
- ***Fairness:*** o ajuste não usa atributos protegidos nem a geografia. Mas o rendimento e a antiguidade do
  crédito podem estar ligados à idade e à condição socioeconómica. Usar os grupos para decidir
  crédito exigiria essa análise.

---

## 10. Resultados gravados (`reports/results/`)

| Ficheiro | Conteúdo |
|---|---|
| `clustering_borrowers.csv` | Uma linha por configuração do k-means (9) e do GMM (18): medidas internas, BIC, certeza, tempos |
| `clustering_comparison.csv` | Tabela de comparação com as referências (k = 4) |
| `clustering_stability.csv` | Estabilidade entre sementes e metades disjuntas (k-means k 2–10, GMM `diag` k 4–5) |
| `clustering_stability_periods.csv` | Estabilidade entre 2007–2012 e 2013–2014 |
| `clustering_groups.csv` | Os quatro grupos finais: perfil, risco e graus |
| `clustering_dbscan.csv` | Grelha do DBSCAN |

---

## 11. Como correr

- **Antes:** `01_preprocessamento.ipynb` (gera `aceites_dev.pkl`).
- **Como está no repositório** (`RECALCULAR = True`), o notebook calcula tudo do zero em cerca
  de 25 minutos:
  - grelha do GMM: ~10 min;
  - estabilidade: ~10 min;
  - DBSCAN: ~2 min.

  As saídas gravadas no notebook e os CSVs vêm dessa execução completa.
- **Durante o desenvolvimento** usou-se `RECALCULAR = False`: as configurações e as análises já
  guardadas são lidas em vez de recalculadas, e o notebook corre em cerca de 40 s. No fim do
  bloco voltou-se a `True` e correu-se tudo do zero. Essa execução deu os mesmos grupos e as
  mesmas medidas da execução por secções; só mudaram os tempos.
- **Alterações a ficheiros da Parte 1 neste bloco:**
  - `docs/01_formulacao_problema/elegibilidade_colunas.md` (`issue_d` ⚠️ no clustering);
  - `src/lending/feature_engineering.py` (`credit_history_years` também no clustering);
  - `docs/03_features/features.md`.

---

## 12. Perguntas prováveis na defesa

- **Porquê k = 4 e não k = 2, que tem a melhor silhueta?** Três das quatro medidas internas, a
  concordância com o GMM e a estabilidade apontam para k = 4. k = 2 só corta a carteira ao meio.
- **Uma silhueta de 0,14 quer dizer que falhou?** Quer dizer que não há grupos naturais (o DBSCAN
  confirma). O resultado é uma segmentação estável e com leitura de negócio, e é assim que se
  apresenta.
- **Porque não escolheram k pelo BIC?** Desce sempre: com 466 124 pontos que não são normais, cada
  componente a mais melhora a verosimilhança mais do que a penalização.
- **Como sabem que os grupos não são um artefacto do algoritmo?** O GMM, com outros pressupostos,
  põe 75% dos empréstimos nos mesmos grupos. Além disso, outra amostra da mesma população dá
  99% de concordância.
- **O juro e o *default* entraram no ajuste?** Não. Foram juntados depois, para dar significado
  aos grupos. Se entrassem, os grupos seriam escalões de risco por construção.
- **Como trataram o *follow-up* desigual?** A taxa de *default* só usa os empréstimos com o prazo
  terminado até 31/12/2018, a regra da classificação.
- **Implementaram alguma medida de raiz?** O índice de Dunn, com NumPy, por blocos, verificado
  contra a definição.
