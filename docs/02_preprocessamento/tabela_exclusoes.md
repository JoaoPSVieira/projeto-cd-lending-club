# Tabela de exclusões

_Gerado automaticamente por `notebooks/01_preprocessamento.ipynb` — não editar à mão._

- Gerado em: 2026-10-17
- Tipo "remoção": as linhas deixam de existir. Tipo "partição": as linhas são válidas mas ficam à parte (ex.: teste final, embargo).
- Decisões e justificações: `docs/02_preprocessamento/` e `docs/01_formulacao_problema/`.

## Comum: ficheiro accepted -> desenvolvimento limpo

| N.º | Passo | Tipo | Regra | Justificação | Antes | Removidas | Depois |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Linhas que não são empréstimos | remoção | id não numérico e todas as outras colunas vazias | Totais e separadores das exportações da LendingClub, não são empréstimos | 2,260,701 | 33 | 2,260,668 |
| 2 | Separar o teste final | partição | issue_d de 2015-01 a 2015-12 | Teste final cronológico, guardado à parte para a avaliação final (desenho_split.md) | 2,260,668 | 421,095 | 1,839,573 |
| 3 | Separar o período reservado | partição | issue_d de 2016-01 a 2018-12 | Não usado na Parte 1: os empréstimos de 36 meses ainda não tinham terminado em 2018-12 | 1,839,573 | 1,373,228 | 466,345 |
| 4 | Condições modificadas depois da concessão | remoção | int_rate = 6.00 num grau B a G, ou \|installment - fórmula de amortização\| > 1 USD | Taxa ou prestação alteradas depois da concessão; o enunciado pede os valores originais | 466,345 | 221 | 466,124 |

**Linhas finais: 466,124**

## Regressão (treino)

| N.º | Passo | Tipo | Regra | Justificação | Antes | Removidas | Depois |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Linhas que não são empréstimos | remoção | id não numérico e todas as outras colunas vazias | Totais e separadores das exportações da LendingClub, não são empréstimos | 2,260,701 | 33 | 2,260,668 |
| 2 | Separar o teste final | partição | issue_d de 2015-01 a 2015-12 | Teste final cronológico, guardado à parte para a avaliação final (desenho_split.md) | 2,260,668 | 421,095 | 1,839,573 |
| 3 | Separar o período reservado | partição | issue_d de 2016-01 a 2018-12 | Não usado na Parte 1: os empréstimos de 36 meses ainda não tinham terminado em 2018-12 | 1,839,573 | 1,373,228 | 466,345 |
| 4 | Condições modificadas depois da concessão | remoção | int_rate = 6.00 num grau B a G, ou \|installment - fórmula de amortização\| > 1 USD | Taxa ou prestação alteradas depois da concessão; o enunciado pede os valores originais | 466,345 | 221 | 466,124 |
| 5 | Fora do período de train da regressão | partição | issue_d fora de 2013-01 a 2014-12 | Antes de 2013 faltam as 37 colunas do bureau registadas só a partir de 2012 (decisão do grupo, valores_em_falta.md) | 466,124 | 95,846 | 370,278 |
| 6 | Sem taxa de juro | remoção | int_rate em falta | int_rate é o alvo da regressão | 370,278 | 0 | 370,278 |

**Linhas finais: 370,278**

## Classificação (treino)

| N.º | Passo | Tipo | Regra | Justificação | Antes | Removidas | Depois |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Linhas que não são empréstimos | remoção | id não numérico e todas as outras colunas vazias | Totais e separadores das exportações da LendingClub, não são empréstimos | 2,260,701 | 33 | 2,260,668 |
| 2 | Separar o teste final | partição | issue_d de 2015-01 a 2015-12 | Teste final cronológico, guardado à parte para a avaliação final (desenho_split.md) | 2,260,668 | 421,095 | 1,839,573 |
| 3 | Separar o período reservado | partição | issue_d de 2016-01 a 2018-12 | Não usado na Parte 1: os empréstimos de 36 meses ainda não tinham terminado em 2018-12 | 1,839,573 | 1,373,228 | 466,345 |
| 4 | Condições modificadas depois da concessão | remoção | int_rate = 6.00 num grau B a G, ou \|installment - fórmula de amortização\| > 1 USD | Taxa ou prestação alteradas depois da concessão; o enunciado pede os valores originais | 466,345 | 221 | 466,124 |
| 5 | Prazo de 60 meses | remoção | term != 36 | Não terminam a tempo de o desfecho ser observado (classificação só para 36 meses, desenho_split.md) | 466,124 | 128,228 | 337,896 |
| 6 | Fora do período de train | partição | issue_d fora de 2007-06 a 2011-12 | Embargo: em 01/01/2015 o prazo ainda não tinha terminado, o rótulo não era observável na data de treino simulada | 337,896 | 306,374 | 31,522 |
| 7 | Desfecho desconhecido | remoção | loan_status = Current ou em falta | O desfecho final ainda não é conhecido (enunciado) | 31,522 | 0 | 31,522 |
| 8 | Classe At risk | remoção | loan_status = In Grace Period ou Late | Classificação binária (decisão do grupo): At risk só existe em empréstimos ativos e tem 0 casos no treino | 31,522 | 0 | 31,522 |

**Linhas finais: 31,522**

## Clustering de mutuários

| N.º | Passo | Tipo | Regra | Justificação | Antes | Removidas | Depois |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Linhas que não são empréstimos | remoção | id não numérico e todas as outras colunas vazias | Totais e separadores das exportações da LendingClub, não são empréstimos | 2,260,701 | 33 | 2,260,668 |
| 2 | Separar o teste final | partição | issue_d de 2015-01 a 2015-12 | Teste final cronológico, guardado à parte para a avaliação final (desenho_split.md) | 2,260,668 | 421,095 | 1,839,573 |
| 3 | Separar o período reservado | partição | issue_d de 2016-01 a 2018-12 | Não usado na Parte 1: os empréstimos de 36 meses ainda não tinham terminado em 2018-12 | 1,839,573 | 1,373,228 | 466,345 |
| 4 | Condições modificadas depois da concessão | remoção | int_rate = 6.00 num grau B a G, ou \|installment - fórmula de amortização\| > 1 USD | Taxa ou prestação alteradas depois da concessão; o enunciado pede os valores originais | 466,345 | 221 | 466,124 |
| 5 | Coorte do clustering de mutuários | remoção | nenhum filtro adicional | Usa todos os empréstimos do desenvolvimento (coorte_alvo.md) | 466,124 | 0 | 466,124 |

**Linhas finais: 466,124**

## Teste final (2015): ficheiro accepted -> teste limpo

| N.º | Passo | Tipo | Regra | Justificação | Antes | Removidas | Depois |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Linhas que não são empréstimos | remoção | id não numérico e todas as outras colunas vazias | Totais e separadores das exportações da LendingClub, não são empréstimos | 2,260,701 | 33 | 2,260,668 |
| 2 | Separar o desenvolvimento e o período reservado | partição | issue_d fora de 2015-01 a 2015-12 | Só os empréstimos do teste final (desenho_split.md) | 2,260,668 | 1,839,573 | 421,095 |
| 3 | Condições modificadas depois da concessão | remoção | int_rate = 6.00 num grau B a G, ou \|installment - fórmula de amortização\| > 1 USD | Taxa ou prestação alteradas depois da concessão; o enunciado pede os valores originais | 421,095 | 207 | 420,888 |

**Linhas finais: 420,888**
