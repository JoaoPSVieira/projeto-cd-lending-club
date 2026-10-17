# Linhas estruturais e partição temporal

Primeiro passo do pré-processamento do ficheiro *accepted*. Aplica a regra do enunciado
*"Define partitions before data-dependent preprocessing"*: antes de qualquer análise de
valores em falta, outliers ou qualidade, faz-se apenas o que não depende de nenhuma
estatística dos dados.

Código: `src/lending/cleaning.py`, `src/lending/parsing.py`, `src/lending/partition.py`,
`src/lending/exclusions.py`. Notebook: `notebooks/01_preprocessamento.ipynb`.

---

## 1. Linhas que não são empréstimos

O CSV do Kaggle é a junção de 16 exportações da LendingClub. Cada exportação termina com
duas linhas de totais, e uma delas inclui um separador. Estas linhas estão **a meio do
ficheiro**, entre exportações, e não apenas no fim.

| Texto na coluna `id` | Linhas |
|---|---:|
| `Total amount funded in policy code 1: <valor>` | 16 |
| `Total amount funded in policy code 2: <valor>` | 16 |
| `Loans that do not meet the credit policy` | 1 |
| **Total** | **33** |

**Regra:** o `id` não é um número inteiro **e** todas as outras 150 colunas estão vazias.
É uma regra estrutural, por isso é aplicada ao ficheiro inteiro, antes da partição. Uma
linha com `id` não numérico mas com dados preenchidos faria o código parar, porque não seria
um rodapé (não existe nenhuma).

---

## 2. Conversão de `issue_d`

`issue_d` (mês de financiamento) está em texto no formato `Mon-AAAA` (ex.: `Dec-2015`). A
conversão usa uma tabela fixa dos 12 meses em inglês, por isso o resultado não depende da
língua do sistema. É uma conversão **determinística**: não estima nada a partir dos dados,
por isso pode ser feita antes da partição. Um valor fora do formato dá erro.

Resultado: todos os 2 260 668 empréstimos têm `issue_d` válido, entre **2007-06 e
2018-12**. Não há nenhuma data posterior ao cutoff de observação (31/12/2018).

---

## 3. Partições

Cortes de calendário definidos na Fase 1 (`docs/01_formulacao_problema/desenho_split.md`),
guardados em `config.PARTITIONS`. Os cortes são feitos ao mês, por isso os empréstimos do
mesmo mês ficam sempre na mesma partição. Cada linha tem de cair numa e uma só partição.

| Partição | Período (`issue_d`) | Empréstimos | % |
|---|---|---:|---:|
| Desenvolvimento | 2007-06 a 2014-12 | 466 345 | 20,63 |
| Teste final | 2015-01 a 2015-12 | 421 095 | 18,63 |
| Reservado | 2016-01 a 2018-12 | 1 373 228 | 60,74 |
| **Total** | | **2 260 668** | 100 |

Empréstimos por ano:

| Ano | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Empréstimos | 603 | 2 393 | 5 281 | 12 537 | 21 721 | 53 367 | 134 814 | 235 629 | 421 095 | 434 407 | 443 579 | 495 242 |
| Partição | Desenv. | Desenv. | Desenv. | Desenv. | Desenv. | Desenv. | Desenv. | Desenv. | Teste | Reserv. | Reserv. | Reserv. |

O desenvolvimento tem só um quinto dos empréstimos, porque a plataforma cresceu muito depois
de 2014. É o custo de manter um teste cronológico com desfechos observáveis.

---

## 4. Ficheiros gerados

| Ficheiro (`data/processed/`) | Conteúdo |
|---|---|
| `accepted_dev_raw.pkl` | Desenvolvimento, 466 345 × 151, ainda em texto |
| `accepted_test_raw.pkl` | Teste final, 421 095 × 151, ainda em texto |
| `accepted_dev_exclusions.pkl` | Registo das exclusões (continuado nos passos seguintes) |

- As partições guardam os valores **exatamente** como no ficheiro original (verificado
  linha a linha). Este passo só separa linhas.
- O índice das linhas é o do ficheiro original, por isso cada empréstimo pode ser
  encontrado no CSV.
- O teste final não volta a ser aberto até à avaliação final. Nessa altura recebe as mesmas
  regras de limpeza, sem as rever.
- O período reservado não é guardado à parte: não é usado na Parte 1 e continua em
  `accepted_raw.pkl`.

---

## 5. Registo de exclusões (até aqui)

| N.º | Passo | Tipo | Regra | Antes | Removidas | Depois |
|---:|---|---|---|---:|---:|---:|
| 1 | Linhas que não são empréstimos | remoção | `id` não numérico e restantes colunas vazias | 2 260 701 | 33 | 2 260 668 |
| 2 | Separar o teste final | partição | `issue_d` de 2015-01 a 2015-12 | 2 260 668 | 421 095 | 1 839 573 |
| 3 | Separar o período reservado | partição | `issue_d` de 2016-01 a 2018-12 | 1 839 573 | 1 373 228 | 466 345 |

Os passos seguintes (limpeza do desenvolvimento e coortes de cada tarefa) continuam este
registo. A tabela completa é gerada no fim do pré-processamento.
