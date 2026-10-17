# Pedidos rejeitados

Pré-processamento do ficheiro *rejected*: verificação estrutural, partição temporal,
conversão de tipos e schema. Os pedidos rejeitados não entram na regressão nem na
classificação; servem o clustering de estados e os testes de hipóteses que comparam aceites
e rejeitados (`docs/01_formulacao_problema/coorte_alvo.md`).

Código: `src/lending/rejected.py` (`REJECTED_SCHEMA`, `REJECTED_NAMES`, `RejectedCleaner`),
`src/lending/parsing.py` (`parse_iso_date`, `parse_percent`),
`config.REJECTED_PARTITIONS`. Notebook: `notebooks/01b_rejeitados.ipynb`.

---

## 1. Ficheiro de origem

| Ficheiro | Unidade de observação | Linhas | Colunas | Período (`Application Date`) |
|---|---|---:|---:|---|
| `rejected_2007_to_2018Q4.csv.gz` | Um pedido recusado | 27 648 741 | 9 | 2007-05-26 a 2018-12-31 |

**Verificação estrutural:** ao contrário do *accepted*, não há linhas de totais nem
separadores. Nenhuma linha está totalmente vazia e todas têm data do pedido. Não é preciso
nenhuma remoção antes da partição.

---

## 2. Partição temporal

Mesmos cortes dos aceites (`desenho_split.md`), pela data do pedido:

| Partição | Período | Pedidos | % | Utilização |
|---|---|---:|---:|---|
| Antes do primeiro aceite | 2007-05 | 49 | 0,0 | Fora do estudo |
| **Desenvolvimento** | 2007-06 a 2014-12 | **3 450 084** | 12,5 | Clustering de estados, testes de hipóteses |
| Teste final | 2015 | 2 859 379 | 10,3 | Não usado |
| Reservado | 2016–2018 | 21 339 229 | 77,2 | Não usado na Parte 1 |

- **Os 49 pedidos de maio de 2007** são anteriores ao primeiro empréstimo aceite (2007-06).
  Sem aceites no mesmo período, não entram nas comparações (ex.: taxa de aprovação).
- **O teste dos rejeitados não é usado:** os rejeitados não entram nos modelos
  supervisionados, e o clustering e os testes de hipóteses usam só o desenvolvimento. Fica
  disponível no ficheiro original.
- O número de pedidos rejeitados cresce muito: 337 277 em 2012, 760 942 em 2013 e
  1 933 700 em 2014. O desenvolvimento é dominado por 2013–2014 (78%).

A partição fica em `data/processed/rejected_dev_raw.pkl` (ainda em texto) e o registo em
`rejected_dev_exclusions.pkl`.

---

## 3. Conversão de tipos

O mesmo `SchemaParser` do *accepted*, com o schema `REJECTED_SCHEMA`. As colunas passam a ter
o **nome da coluna equivalente do *accepted*** e a mesma unidade, para que comparar aceites e
rejeitados seja direto.

| Coluna original | Nome usado | Tipo | Conversão | Unidade |
|---|---|---|---|---|
| Amount Requested | `amount_requested` | number | "1000.0" → 1000.0 | USD |
| Application Date | `application_date` | date | "2007-05-26" → data | dia |
| Loan Title | `title` | text | espaços nas pontas retirados | texto |
| Risk_Score | `risk_score` | number | "693.0" → 693.0 | pontos |
| Debt-To-Income Ratio | `dti` | percent | "38.64%" → 38.64 | % (0–100) |
| Zip Code | `zip_code` | category | "481xx" | 3 primeiros dígitos |
| State | `addr_state` | category | sigla | estado |
| Employment Length | `emp_length` | emp_length | "< 1 year" → 0 … "10+ years" → 10 | anos |
| Policy Code | `policy_code` | number | "0.0" → 0.0 | código |

**Comparabilidade com o *accepted*:**
- `dti` fica na escala 0–100, como o `dti` do *accepted* (onde 18.5 = 18,5%).
- `emp_length` usa a mesma tabela de conversão do *accepted* (`EMP_LENGTH_YEARS`).
- `amount_requested` é o montante **pedido**. O equivalente no *accepted* é `loan_amnt`, que
  pode refletir reduções feitas pelo departamento de crédito (dicionário de dados).

**Resultado da conversão no desenvolvimento:** nenhum valor fora do formato (o parser dá
erro nesse caso). Só 15 239 títulos (`title`) com apenas espaços passaram a valor em falta.

O schema completo, com o intervalo observado de cada coluna, está em `schema_rejected.md`
(gerado pelo notebook).

---

## 4. Valores em falta

% em falta por ano do pedido, com os tipos convertidos e antes das correções:

| Coluna | Em falta | Porquê |
|---|---|---|
| `risk_score` | 1–11% até 2013, **13% em 2014** | Até 2013 os valores em falta vinham sobretudo como 0 (secção 5.1); em 2014 passam a vir vazios |
| `title` | até 18% em 2008–2010 | Texto livre não preenchido; não é usado nas análises |
| `emp_length` | 1–2% a partir de 2010 | Não preenchido |
| `zip_code`, `addr_state` | 21–23 pedidos | Não preenchido |

---

## 5. *Risk score*

### 5.1 Zeros → valor em falta

`Risk_Score` = 0 não é uma pontuação possível (o FICO vai de 300 a 850) e o enunciado diz
que conta como valor em falta. Há zeros em todos os anos até 2013 e nenhum em 2014, quando
os valores em falta passam a vir vazios: é o mesmo significado guardado de duas formas.

### 5.2 ⚠️ Mudança de escala a 2013-11-06

| Período | Mínimo | Máximo | Escala |
|---|---:|---:|---|
| Até 2013-11-05 | ≈ 420 | 850 | FICO |
| A partir de 2013-11-06 | 501 (primeiros dias) | 990 | VantageScore (intervalo 501–990) |

A mudança acontece num dia exato. Não pode ser uma mudança nos requerentes: é a LendingClub
a mudar a pontuação que regista. Um `risk_score` de 700 não quer dizer o mesmo antes e
depois desta data, e não há uma conversão exata entre as duas escalas.

**Decisão:** os valores não são alterados. Acrescenta-se a coluna `risk_score_scale`
("FICO" / "VantageScore"), definida só pela data (`VANTAGE_START`). No desenvolvimento:
1 402 189 pedidos em FICO (2007 a 2013-11-05) e 2 022 039 em VantageScore.

**Consequência para as análises:**
- comparar o *risk score* dos rejeitados com o FICO dos aceites (`fico_range_low` /
  `fico_range_high`) só é válido no **período FICO**;
- não se pode calcular uma média ou um *cluster* com as duas escalas misturadas.

---

## 6. DTI

O enunciado avisa que o DTI vem em texto com "%" (convertido na secção 3) e que há
artefactos de introdução, dando o exemplo de 4042.11%. Nos aceites do desenvolvimento, o
DTI vai de 0 a 39,99%. As contagens são do desenvolvimento antes das exclusões da secção 11
(depois delas, os valores corrigidos estão na tabela final).

| Valores | Pedidos | Interpretação | Regra |
|---|---:|---|---|
| −1 | 265 406 | Código de "sem dados": 97% destes pedidos também não têm *risk score* | → valor em falta |
| 9 999, 99 999, 199 998 | 12 513 | Números redondos repetidos milhares de vezes: valores de teto do campo | → valor em falta |
| Acima de 1000% | 8 346 | As dívidas mensais valeriam mais de 10 vezes o rendimento: rendimento mal introduzido. Inclui o 4042.11% do enunciado | → valor em falta |
| 100% a 1000% | 72 456 | Plausível num rejeitado: dívidas acima do rendimento são um motivo de rejeição | Fica |
| 0% | 203 261 | Possível (sem outras dívidas) | Fica |

O corte em 1000% (`DTI_MAX`) é uma escolha: a distribuição não tem um corte natural. Afeta
só cerca de 0,24% dos pedidos, e as análises do DTI dos rejeitados devem usar estatísticas
robustas (mediana, testes de postos), que quase não dependem da posição exata do corte.

---

## 7. Montante pedido

| Valores | Pedidos | Regra |
|---|---:|---|
| 0 USD | 2 | → valor em falta (um pedido sem montante não é um valor real) |
| Acima de 35 000 USD (o máximo dos aceites até 2016) | 17 711 (17 291 em 2014) | Fica: são pedidos reais acima do limite da plataforma |

Na hipótese "o montante pedido difere entre aceites e rejeitados", parte da diferença vem
deste limite da plataforma, e isso tem de ser dito na interpretação.

---

## 8. Anos de emprego

A % de pedidos com "< 1 year" passa de cerca de 27% em 2007–2009 para 66% em 2010 e 85% a
partir de 2011. Uma mudança tão rápida não vem dos requerentes: deve ser o valor por omissão
do formulário. Não há forma de separar quem tinha mesmo menos de um ano de emprego.

**Decisão:** fica como está e documentado. O `emp_length` dos rejeitados não é fiável a
partir de 2010 e não deve ser usado nas comparações com os aceites.

---

## 9. Código de política

Todos os aceites do desenvolvimento têm `policy_code` = 1 (produtos públicos). Os rejeitados
têm 0 e, em 2014, 6 241 pedidos com 2, que o dicionário de dados descreve como *"new products
not publicly available"*.

**Decisão:** excluir `policy_code` = 2. Manter estes pedidos misturaria produtos diferentes
na taxa de aprovação.

---

## 10. Pedidos repetidos

O ficheiro não tem identificador do pedido. Há **19 615 linhas (0,57%) iguais a outra em
todas as colunas**: o mesmo dia, título, código postal, montante, *risk score*, DTI, anos de
emprego e código de política. São sobretudo pedidos incompletos (sem *risk score*, DTI = −1)
submetidos várias vezes no mesmo dia.

**Decisão:** fica a primeira linha de cada grupo. Deixá-las contaria várias rejeições onde
houve uma.

**Limitação:** dois requerentes diferentes com exatamente os mesmos valores no mesmo dia
seriam juntados. É muito improvável, porque teriam de coincidir nas 9 colunas.

---

## 11. Resultado

As regras estão na classe `RejectedCleaner` (`src/lending/rejected.py`) e são aplicadas por
esta ordem:
1. conversão de tipos;
2. exclusões, sobre os valores originais;
3. correções;
4. coluna `risk_score_scale`.

As exclusões vêm antes das correções porque as correções juntam valores diferentes em "em
falta" (ex.: DTI −1 e 9 999), o que criaria falsos repetidos.

| Passo | Antes | Removidas | Depois |
|---|---:|---:|---:|
| Pedidos anteriores ao primeiro aceite (2007-05) | 27 648 741 | 49 | 27 648 692 |
| Separar o teste final (2015) | 27 648 692 | 2 859 379 | 24 789 313 |
| Separar o período reservado (2016–2018) | 24 789 313 | 21 339 229 | 3 450 084 |
| Produtos novos não públicos | 3 450 084 | 6 241 | 3 443 843 |
| Pedidos repetidos | 3 443 843 | 19 615 | **3 424 228** |

| Correção (→ valor em falta) | Valores |
|---|---:|
| `risk_score` = 0 | 86 038 |
| `dti` = −1, 9 999, 99 999, 199 998 | 264 300 |
| `dti` > 1000% (sem os códigos) | 8 253 |
| `amount_requested` = 0 | 2 |

Depois da limpeza faltam 10,9% dos `risk_score`, 8,0% dos `dti` e 1,5% dos `emp_length`.

**Ficheiro:** `data/processed/rejeitados_dev.pkl` (3 424 228 × 10: as 9 colunas e
`risk_score_scale`). A tabela de exclusões gerada está em `tabela_exclusoes_rejeitados.md`.

### Cuidados para a EDA, os testes de hipóteses e o clustering

| Coluna | Cuidado |
|---|---|
| `risk_score` | Comparar com o FICO dos aceites só no período FICO (`risk_score_scale` = "FICO") |
| `dti` | Usar estatísticas robustas: ainda há valores até 1000% |
| `amount_requested` | Comparável com `loan_amnt`, mas inclui pedidos acima do limite da plataforma |
| `emp_length` | Não usar a partir de 2010 |
| Volume | 78% dos pedidos são de 2013–2014: as taxas de aprovação por estado devem ser vistas por período |
