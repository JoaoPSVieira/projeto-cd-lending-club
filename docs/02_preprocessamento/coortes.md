# Coortes de cada tarefa e tabela de exclusões

Último passo da limpeza do ficheiro *accepted*. Aplica as coortes definidas na Fase 1
(`docs/01_formulacao_problema/coorte_alvo.md` e `desenho_split.md`) ao conjunto de
desenvolvimento limpo, com as duas decisões do grupo tomadas durante a limpeza (secção 4).
Constrói o alvo da classificação, gera a tabela de exclusões e guarda os conjuntos limpos.
Responde ao pedido do enunciado *"Document all exclusions"*.

Código: `src/lending/cohorts.py` (`CohortBuilder`, `make_target`, `TARGET_CLASSES`,
`AT_RISK_STATUSES`), `src/lending/exclusions.py` (`ExclusionLog.branch`,
`write_exclusions_markdown`), `config.REGRESSION_PARTITIONS`,
`config.CLASSIFICATION_PARTITIONS`, `config.CLASS_COL`, `config.CLASS_ORDER`. Notebook:
`notebooks/01_preprocessamento.ipynb`, secções 9 e 10.

---

## 1. Conjunto de desenvolvimento limpo

| Passo | Antes | Removidas | Depois |
|---|---:|---:|---:|
| Linhas que não são empréstimos | 2 260 701 | 33 | 2 260 668 |
| Separar o teste final (2015) | 2 260 668 | 421 095 | 1 839 573 |
| Separar o período reservado (2016–2018) | 1 839 573 | 1 373 228 | 466 345 |
| Condições modificadas depois da concessão | 466 345 | 221 | **466 124** |

Além das exclusões, o conjunto limpo tem os tipos convertidos (`conversao_tipos.md`), os
zeros disfarçados corrigidos (`valores_em_falta.md`) e os tetos 9 999 999 passados a valor
em falta (`valores_suspeitos.md`). Fica em `data/processed/aceites_dev.pkl`
(466 124 × 151) e é o ponto de partida da EDA geral, do clustering e dos testes de hipóteses
que comparam aceites e rejeitados.

A tabela completa, passo a passo e por tarefa, está em `tabela_exclusoes.md` (gerada pelo
notebook).

---

## 2. Coortes

| Tarefa | Passos a partir do conjunto limpo | Empréstimos |
|---|---|---:|
| Regressão (treino) | Fora de 2013-01 a 2014-12 (−95 846); sem `int_rate` (0) | 370 278 |
| Classificação (treino) | Prazo de 60 meses (−128 228); fora de 2007-06 a 2011-12, ou seja, embargo 2012–2014 (−306 374); desfecho desconhecido (0); classe *At risk* (0) | 31 522 |
| Clustering de mutuários | Nenhum filtro adicional | 466 124 |

A regressão fica com 134 767 empréstimos de 2013 e 235 511 de 2014.

Na classificação, os passos "desfecho desconhecido" e "classe *At risk*" removem 0
empréstimos no treino: todos os `Current` e todos os *At risk* do desenvolvimento são de 60
meses e já saíram no primeiro passo. Os passos ficam registados porque no teste (2015)
podem remover empréstimos.

O teste final (2015) recebe as mesmas regras, com o mesmo `CohortBuilder`, apenas na
avaliação final.

### 2.1 Pendentes da Fase 1

| Verificação | Resultado |
|---|---|
| Empréstimos "Does not meet the credit policy" | 2 748 no desenvolvimento limpo, 2 437 na coorte da classificação (todos de 2007–2010) |
| `loan_status` em falta | 0. Os registos "sem valor" vistos na Fase 1 eram as 33 linhas de totais, já removidas |
| Empréstimos de 60 meses antes de 2010 | Nenhum (`descricao_dados.md`) |

---

## 3. Alvo da classificação

O alvo fica na coluna `target` (`config.CLASS_COL`), categórica e ordenada pela ordem de
`config.CLASS_ORDER`: *Good standing*, *Default*. Todas as tabelas e figuras usam esta ordem.

| Classe | `loan_status` de origem | Empréstimos | % |
|---|---|---:|---:|
| Good standing | Fully Paid (25 858), Does not meet… Fully Paid (1 788) | 27 646 | 87,7 |
| Default | Charged Off (3 227), Does not meet… Charged Off (649) | 3 876 | 12,3 |

O treino é desequilibrado (cerca de 7 *Good standing* por cada *Default*). Isto pesa na
escolha das métricas e da validação, que se faz na seleção de modelos. Qualquer
reamostragem só pode ser feita dentro dos *folds* de treino.

---

## 4. Decisões do grupo tomadas na limpeza

A limpeza mostrou dois problemas que o desenho da Fase 1 não previa. O grupo decidiu
resolvê-los da forma descrita abaixo e documentar a razão.

### 4.1 Classificação binária: *Good standing* vs *Default*

**O problema.** *At risk* (em atraso ou em *grace period*) é um estado que só existe
enquanto o empréstimo está **ativo**. A regra de observabilidade escolhida na Fase 1 (só
empréstimos com o prazo terminado na data de treino simulada) deixa no treino apenas
empréstimos já fechados, pagos ou perdidos. Por isso a classe *At risk* tem 0 casos.

No desenvolvimento limpo há 610 empréstimos *At risk*, **todos de 60 meses concedidos em
2013–2014** e ainda ativos na data de extração (≈ março de 2019). Nenhum empréstimo de 36
meses está em atraso, em nenhum ano.

**Porque é um conflito do enunciado e não um erro de código.** O enunciado pede ao mesmo
tempo:
1. uma classificação em **3 classes**, uma das quais (*At risk*) é um estado intermédio;
2. que os rótulos de treino sejam **observáveis na data de treino simulada** (*"a model
   supposedly trained in 2016 cannot use the eventual 2018 outcome of a loan originated in
   2015"*).

O ficheiro tem uma única fotografia do estado de cada empréstimo, tirada em março de 2019.
Não é possível saber se um empréstimo estava em atraso numa data anterior (ex.: 01/01/2015).
Um empréstimo só aparece *At risk* se ainda estiver ativo em 2019, e esses empréstimos não
podem ter rótulo observável numa data de treino anterior ao teste.

**Opções consideradas:**

| Opção | O que muda | A favor | Contra |
|---|---|---|---|
| **A. Manter o desenho estrito (escolhida)** | A classificação fica binária | Cumpre a regra de observabilidade e evita o *unequal follow-up* | *At risk* não pode ser prevista |
| B. Desenho por fotografia | Treino com todos os empréstimos não `Current` de 2007–2014 (36 e 60 meses), com o rótulo de março de 2019 | Há casos *At risk* (610, cerca de 0,13%) | Viola a regra de observabilidade; reintroduz o *unequal follow-up*; *At risk* continua raríssima |

**Decisão:** opção A. As regras de *leakage* e de validação do enunciado são obrigatórias,
e mesmo a opção B só daria 0,13% de *At risk*, demasiado pouco para um modelo aprender a
classe. O mapeamento do enunciado para 3 classes continua no código (`TARGET_CLASSES`), e os
*At risk* saem da coorte num passo próprio da tabela de exclusões, como os `Current`. É uma
limitação reconhecida no relatório.

### 4.2 Regressão: treino só com 2013–2014

**O problema.** 37 colunas do *bureau* só são registadas a partir de 2012
(`valores_em_falta.md`, secção 4): estão vazias em 2007–2011, faltam em parte de 2012 e
ficam completas a partir de 2013. Entre elas estão 8 dos *inputs* sugeridos pelo enunciado
(`mort_acc`, `tot_cur_bal`, `tot_hi_cred_lim`, `bc_util`, `bc_open_to_buy`,
`mo_sin_old_rev_tl_op`, `num_actv_rev_tl`, `pct_tl_nvr_dlq`). Com o treino em 2007–2014,
faltariam em cerca de 15% das linhas, e um indicador de "em falta" seria, na prática, um
indicador do ano de concessão.

**Opções consideradas:**

| Opção | A favor | Contra |
|---|---|---|
| a. Treino 2007–2014, com imputação e indicador de "em falta" | Mais dados | O indicador mede o ano e não o mutuário; a imputação inventa valores para 5 anos inteiros |
| **b. Treino só com 2013–2014 (escolhida)** | As colunas estão completas (≤ 1% em falta); período mais próximo do teste (2015), o que reduz a mudança temporal | Menos dados (370 278 em vez de 466 124) e só dois anos para a validação temporal |
| c. Treino 2007–2014 sem estas colunas | Mais dados | Perde 8 *inputs* sugeridos pelo enunciado |

**Decisão:** opção b (`config.REGRESSION_PARTITIONS`). Os 95 846 empréstimos de 2007-06 a
2012-12 ficam fora do treino da regressão, num passo de tipo "partição" da tabela de
exclusões. Continuam no conjunto limpo e são usados na EDA geral e no clustering.

Consequência para a seleção de modelos: com dois anos, a validação cruzada temporal por ano
dá só um *fold* (treino 2013, validação 2014). A validação passa a ter de ser feita com
cortes mais finos (ex.: janela expansível por trimestre), o que se define na seleção de
modelos.

**Na classificação**, estas 37 colunas estão 100% vazias no treino (2007–2011) e não podem
ser usadas.

---

## 5. Ficheiros gerados

| Ficheiro (`data/processed/`) | Conteúdo |
|---|---|
| `aceites_dev.pkl` | Desenvolvimento limpo, 2007-06 a 2014-12, 466 124 × 151 |
| `regressao_dev.pkl` | Coorte da regressão (treino), 2013-01 a 2014-12, 370 278 × 151 |
| `classificacao_treino.pkl` | Coorte da classificação (treino), 31 522 × 152 (151 colunas + `target`) |
| `accepted_dev_exclusions.pkl` | Registo comum (4 passos) |
| `accepted_dev_cohort_logs.pkl` | Registos da regressão, da classificação e do clustering |

Os nomes estão em `config.py` e foram combinados no grupo. Os conjuntos mantêm todas as
colunas: a escolha das *features* de cada tarefa (tabela de elegibilidade da Fase 1) é
feita depois, no pipeline. Por isso as colunas proibidas (ex.: `total_pymnt`, `grade` na
regressão) **continuam nos ficheiros e nunca podem entrar como preditores**.

As coortes continuam a ser definidas num só sítio, o `CohortBuilder`. Os ficheiros são o
resultado de o aplicar e regeneram-se correndo o notebook.

Ainda não gerados (passos seguintes): `teste_2015.pkl` (teste final com a mesma limpeza) e
`rejeitados_dev.pkl` (pedidos rejeitados).
