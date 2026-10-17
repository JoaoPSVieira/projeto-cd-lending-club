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

## 4. Para a análise de qualidade

O schema mostra valores que precisam de análise antes de serem usados:

| Coluna | Observado no desenvolvimento | A analisar |
|---|---|---|
| `dti` | de −1 a 50 000 031,49 | Negativos e artefactos de introdução de dados (o enunciado dá o exemplo 4042.11%) |
| `risk_score` | de 0 a 990 | `Risk_Score` = 0 é valor em falta (enunciado); escala até 990, acima do máximo FICO (850) |
| `amount_requested` | de 0 a 1 400 000 | Pedidos de 0 USD e montantes acima do máximo da plataforma |
| `emp_length` | 85% com "< 1 year" | Valor por omissão do formulário? |
| `policy_code` | 0 e 2 | Significado de cada código |
