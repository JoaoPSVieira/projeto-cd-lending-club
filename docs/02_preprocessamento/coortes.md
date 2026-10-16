# Coortes de cada tarefa e tabela de exclusões

Último passo da limpeza do ficheiro *accepted*. Aplica as coortes definidas na Fase 1
(`docs/01_formulacao_problema/coorte_alvo.md` e `desenho_split.md`) ao conjunto de
desenvolvimento limpo, constrói o alvo da classificação, gera a tabela de exclusões e guarda
o conjunto limpo. Responde ao pedido do enunciado *"Document all exclusions"*.

Código: `src/lending/cohorts.py` (`CohortBuilder`, `make_target`, `TARGET_CLASSES`),
`src/lending/exclusions.py` (`ExclusionLog.branch`, `write_exclusions_markdown`),
`config.CLASSIFICATION_PARTITIONS`. Notebook: `notebooks/01_preprocessamento.ipynb`,
secções 9 e 10.

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
em falta (`valores_suspeitos.md`). Fica em `data/processed/accepted_dev_clean.pkl`
(466 124 × 151) e é o ponto de partida da EDA e dos testes de hipóteses.

A tabela completa, passo a passo e por tarefa, está em `tabela_exclusoes.md` (gerada pelo
notebook).

---

## 2. Coortes

| Tarefa | Passos a partir do conjunto limpo | Empréstimos |
|---|---|---:|
| Regressão | Sem `int_rate` (0 removidos) | 466 124 |
| Classificação (treino) | Prazo de 60 meses (−128 228); fora de 2007-06 a 2011-12, ou seja, embargo 2012–2014 (−306 374); desfecho desconhecido (0) | 31 522 |
| Clustering de mutuários | Nenhum filtro adicional | 466 124 |

Na classificação, o passo "desfecho desconhecido" remove 0 empréstimos porque todos os
`Current` do desenvolvimento são de 60 meses e já saíram no primeiro passo.

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

| Classe | `loan_status` de origem | Empréstimos | % |
|---|---|---:|---:|
| Good standing | Fully Paid (25 858), Does not meet… Fully Paid (1 788) | 27 646 | 87,7 |
| At risk | In Grace Period, Late (16-30), Late (31-120) | **0** | 0,0 |
| Default | Charged Off (3 227), Does not meet… Charged Off (649) | 3 876 | 12,3 |

---

## 4. ⚠️ Questão em aberto: a classe *At risk* não existe no treino

**O que acontece.** *At risk* (em atraso ou em *grace period*) é um estado que só existe
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

**Opções:**

| Opção | O que muda | A favor | Contra |
|---|---|---|---|
| **A. Manter o desenho estrito** | Nada | Cumpre a regra de observabilidade e evita o *unequal follow-up* | O classificador só aprende *Good standing* contra *Default*; *At risk* não pode ser prevista |
| **B. Desenho por fotografia** | Treino com todos os empréstimos não `Current` de 2007–2014 (36 e 60 meses), com o rótulo de março de 2019 | Há casos *At risk* (610, cerca de 0,13%) | Viola a regra de observabilidade; reintroduz o *unequal follow-up*; *At risk* continua raríssima |
| **C. Perguntar ao professor** | — | O conflito está no enunciado; a decisão fica justificada | Depende da resposta |

Até haver decisão, as coortes seguem o desenho da Fase 1 (opção A). A decisão é precisa
antes da modelação (Parte 2) e não afeta a limpeza, a EDA geral, a regressão nem o
clustering.

---

## 5. Ficheiros gerados

| Ficheiro (`data/processed/`) | Conteúdo |
|---|---|
| `accepted_dev_clean.pkl` | Desenvolvimento limpo, 466 124 × 151 |
| `accepted_dev_exclusions.pkl` | Registo comum (4 passos) |
| `accepted_dev_cohort_logs.pkl` | Registos da regressão, da classificação e do clustering |

As coortes não são guardadas à parte: reconstroem-se a partir do conjunto limpo com o
`CohortBuilder`, o que garante que a definição está num só sítio.
