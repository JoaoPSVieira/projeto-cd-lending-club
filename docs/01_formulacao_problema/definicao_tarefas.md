# Definição das tarefas

Este documento define as três tarefas do projeto antes de qualquer análise dos dados.
Cada definição indica **o que se prevê** (ou descreve), **para quem** (população) e
**em que momento**. Estas decisões servem de base à seleção de variáveis e à divisão
dos dados, e não devem ser alteradas em função dos resultados observados.

Unidade de análise nas tarefas supervisionadas: **um empréstimo financiado**
(uma linha de `accepted_2007_to_2018Q4.csv`).

---

## 1. Regressão - taxa de juro

**Definição:** prever a taxa de juro anual (`int_rate`, em %) que a LendingClub
atribuiria a um pedido de empréstimo, **no momento em que o pedido é submetido, antes
de a taxa ser fixada**, usando apenas a informação do pedido (montante, prazo,
finalidade) e o perfil de crédito do requerente.

| Elemento | Decisão |
|---|---|
| Alvo | `int_rate` (variável contínua, % ao ano) |
| População | Todos os empréstimos financiados, independentemente do estado atual (`loan_status`), porque a taxa é conhecida para todos desde a originação |
| Momento de previsão | Submissão do pedido, antes do *pricing* (antes de serem atribuídos `grade`, `sub_grade` e `installment`) |
| Objetivo | Estimar a taxa que um novo requerente provavelmente receberia e perceber que fatores do perfil de crédito determinam o preço do risco |

---

## 2. Classificação - desfecho do empréstimo

**Definição:** prever, **no momento da concessão do empréstimo** - quando as condições
(taxa, grau de risco, prestação) já estão fixadas, mas nenhum pagamento foi ainda
observado - em qual de três classes de desfecho o empréstimo vai terminar.

| Classe | Valores de `loan_status` |
|---|---|
| Good standing | Fully Paid |
| At risk | In Grace Period, Late (16–30 days), Late (31–120 days) |
| Default | Charged Off, Default |

| Elemento | Decisão |
|---|---|
| Alvo | Classe de desfecho derivada de `loan_status` (3 classes) |
| População | Empréstimos financiados com desfecho conhecido. **Excluem-se os empréstimos `Current`** sem atraso registado, porque o seu desfecho final ainda não é conhecido |
| Momento de previsão | Originação (concessão), depois do *pricing* e antes do primeiro pagamento |
| Objetivo | Identificar, no momento da concessão, empréstimos com maior probabilidade de atraso ou incumprimento |

---

## 3. Clustering - agrupamento descritivo

**Definição:** identificar grupos de perfis semelhantes **sem prever nenhum alvo**. A
tarefa é exploratória e não tem momento de previsão.

| Análise | População | Variáveis (exemplos do enunciado) |
|---|---|---|
| Mutuários | Empréstimos financiados | Perfil de crédito e indicadores de risco no momento do pedido: FICO, DTI, antiguidade do histórico de crédito, utilização do crédito |
| Estados (EUA) | Pedidos aceites e rejeitados, agregados por estado | Taxa de aprovação, montantes pedidos, *risk scores* |

| Elemento | Decisão |
|---|---|
| Alvo | Nenhum (aprendizagem não supervisionada) |
| Momento | Não aplicável - descreve a população, não prevê casos futuros |
| Objetivo | Encontrar segmentos interpretáveis de mutuários e de estados |
