# Coorte e alvo

Define **que empréstimos entram em cada tarefa** (coorte) e, na classificação, **como se
constrói o alvo** a partir de `loan_status`. Estas regras são fixadas antes da análise
exploratória e aplicam-se de forma determinística (não dependem de estatísticas dos dados).

Para conhecer as categorias possíveis do alvo foram listados os **valores distintos** de
`loan_status`, sem contagens nem distribuições. Foram encontrados 9 valores e alguns
registos sem valor.

---

## 1. Classificação

### 1.1 Mapeamento de `loan_status` para as 3 classes

| `loan_status` | Classe | Justificação |
|---|---|---|
| Fully Paid | **Good standing** | Definição do enunciado |
| Does not meet the credit policy. Status:Fully Paid | **Good standing** | Empréstimo pago; ver 1.2 |
| In Grace Period | **At risk** | Definição do enunciado |
| Late (16-30 days) | **At risk** | Definição do enunciado |
| Late (31-120 days) | **At risk** | Definição do enunciado |
| Charged Off | **Default** | Definição do enunciado |
| Default | **Default** | Definição do enunciado |
| Does not meet the credit policy. Status:Charged Off | **Default** | Empréstimo perdido; ver 1.2 |
| Current | **Excluído** | Desfecho ainda desconhecido (enunciado) |
| *(sem valor)* | **Excluído** | Registo sem estado; verificado na limpeza |

### 1.2 Empréstimos "Does not meet the credit policy"

Estes empréstimos foram concedidos ao abrigo de critérios de crédito anteriores, que a
LendingClub deixou de aplicar. O desfecho (`Status:Fully Paid` ou `Status:Charged Off`) é
conhecido e final.

| Opção | Vantagem | Desvantagem |
|---|---|---|
| **A — Incluir, mapeando pelo desfecho (escolhida)** | São empréstimos reais com desfecho conhecido; não se perde informação | Pertencem a uma política de crédito diferente |
| B — Excluir | População mais homogénea (só a política atual) | Perde casos com desfecho conhecido, sobretudo dos primeiros anos |

**Decisão:** opção A. A indicação "does not meet the credit policy" **não** é usada como
preditor, porque está codificada na coluna do alvo. O número de casos afetados é
registado na tabela de exclusões/mapeamentos da limpeza.

### 1.3 Coorte da classificação

> Empréstimos financiados, com `loan_status` preenchido e diferente de `Current`,
> concedidos no período de desenvolvimento ou de teste definido no desenho do split.

### 1.4 Limitações do alvo

- **"At risk" não é um desfecho final.** É o estado do empréstimo na data de extração
  (fim de 2018): um empréstimo em atraso pode ainda vir a ser pago ou a entrar em default.
  Segue-se a definição do enunciado e a limitação é discutida no relatório.
- **Viés de seleção por excluir os `Current`.** Nos anos mais recentes, os empréstimos que
  já terminaram são sobretudo os que entraram em incumprimento cedo ou foram pagos
  antecipadamente; os restantes ainda estão `Current`. As coortes recentes ficam por isso
  enviesadas. O mesmo acontece com os empréstimos de 60 meses, que precisam de mais tempo
  para terminar. Este problema (*unequal follow-up*) é tratado no desenho do split, através
  da escolha do período e de uma data de observação comum.

---

## 2. Regressão

> Todos os empréstimos financiados com `int_rate` preenchido, incluindo `Current` e
> "Does not meet the credit policy", concedidos no período de desenvolvimento ou de teste
> definido no desenho do split.

A taxa é fixada no pricing e é conhecida para todos os empréstimos, independentemente do
desfecho. Por isso, `loan_status` **não** é usado para filtrar a coorte da regressão.

---

## 3. Clustering

| Análise | Coorte | Unidade |
|---|---|---|
| Mutuários | Empréstimos financiados do período de desenvolvimento | Um empréstimo (perfil do mutuário no pedido) |
| Estados | Pedidos aceites (ficheiro de aceites) **e** rejeitados (ficheiro de rejeitados) do período de desenvolvimento | Um estado dos EUA, com variáveis agregadas |

No clustering de estados, os pedidos rejeitados são associados ao estado pela coluna
`State`, e os aceites por `addr_state`. A data de referência dos rejeitados é
`Application Date`, e a dos aceites é `issue_d`.

---

## 4. Resumo

| Tarefa | Inclui | Exclui |
|---|---|---|
| Regressão | Todos os financiados com `int_rate` | — |
| Classificação | Fully Paid, Late, In Grace Period, Charged Off, Default, "Does not meet…" | `Current`, sem estado |
| Clustering — mutuários | Financiados (desenvolvimento) | Período de teste |
| Clustering — estados | Aceites + rejeitados (desenvolvimento) | Período de teste |
