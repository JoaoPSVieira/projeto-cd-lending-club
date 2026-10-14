# Desenho da divisão dos dados (split)

Define, **antes de qualquer pré-processamento dependente dos dados**, que empréstimos são
usados para desenvolvimento (treino e validação) e quais ficam reservados para o **teste
final**. As partições usam **cortes de calendário explícitos** sobre `issue_d` (mês de
financiamento) e não a ordem das linhas no ficheiro. Como os cortes são feitos no início de
cada mês, todos os empréstimos do mesmo mês ficam sempre na mesma partição.

---

## 1. Datas de referência

| Data | Valor | Justificação |
|---|---|---|
| **Cutoff de observação comum** | 31/12/2018 | O dataset cobre empréstimos até ao 4.º trimestre de 2018 (`accepted_2007_to_2018Q4`, enunciado: "2007–2018"). O `loan_status` é tratado como a fotografia do estado de cada empréstimo nesta data. A data vem da documentação do dataset e **não** é inferida a partir das datas de pagamento; na limpeza apenas se verifica a consistência (ex.: nenhum `issue_d` posterior) |
| **Data de treino simulada** | 01/01/2015 | Simula um modelo treinado no início de 2015 e aplicado aos empréstimos concedidos durante 2015 |

---

## 2. Partições globais

```text
issue_d:   2007-06 ──────────────────── 2014-12 │ 2015-01 ── 2015-12 │ 2016-01 ──── 2018-12
                    DESENVOLVIMENTO              │    TESTE FINAL     │   RESERVADO
              (treino, validação, EDA,           │  (usado uma única  │ (não usado na Parte 1)
               testes de hipóteses)              │   vez, no fim)     │
```

| Partição | Período (`issue_d`) | Utilização |
|---|---|---|
| Desenvolvimento | 2007-06 a 2014-12 | Treino, validação, EDA, testes de hipóteses, engenharia de features e ajuste (*fit*) de todas as transformações |
| Teste final | 2015-01 a 2015-12 | Avaliação final dos modelos, **uma única vez**, no fim da Parte 2. Não é usado na EDA nem em nenhuma decisão de modelação |
| Reservado | 2016-01 a 2018-12 | Não é usado na Parte 1. Os empréstimos de 36 meses destes anos ainda não tinham terminado no cutoff de observação |

Os pedidos **rejeitados** seguem os mesmos cortes, usando `Application Date`.

---

## 3. Regressão

| Elemento | Decisão |
|---|---|
| Coorte | Todos os empréstimos financiados com `int_rate` preenchido (36 e 60 meses, incluindo `Current`) |
| Treino + validação | `issue_d` de 2007-06 a 2014-12 |
| Validação | Validação cruzada temporal com **janela expansível** por ano civil: cada *fold* treina com todos os anos anteriores e valida no ano seguinte (ex.: treino até 2010 → valida 2011; …; treino até 2013 → valida 2014) |
| Teste final | `issue_d` de 2015-01 a 2015-12 |

**Observabilidade do alvo:** `int_rate` é conhecido no momento da concessão. Na data de
treino simulada, todos os alvos de treino já eram observáveis, por isso a regra do enunciado
é cumprida sem restrições adicionais.

---

## 4. Classificação

### 4.1 Regra de observabilidade

Um rótulo só é usado se, na data em que o modelo é treinado ou avaliado, o empréstimo já
tiver chegado ao **fim do prazo** (`issue_d + term`). Assim:

- **todos** os empréstimos usados têm um *follow-up* completo, o que elimina o viés de
  *unequal follow-up* descrito em [`coorte_alvo.md`](coorte_alvo.md);
- os rótulos de treino eram observáveis na data de treino simulada, como exige o enunciado.

### 4.2 Consequência: só empréstimos de 36 meses

| Prazo | Para o treino (fim do prazo ≤ 01/01/2015) | Para o teste (fim do prazo ≤ 31/12/2018) |
|---|---|---|
| 36 meses | Concedidos até 2011-12 | Concedidos em 2015 ✔ |
| 60 meses | Concedidos até 2009-12 | Concedidos em 2015 terminam em 2020 ✘ |

Os empréstimos de 60 meses de 2015 não terminam antes do cutoff de observação. Na
prática, também não existem empréstimos de 60 meses no período de treino (a verificar na
limpeza). Por isso, a classificação é definida **apenas para empréstimos de 36 meses**.
É uma limitação do âmbito do modelo, reconhecida no relatório.

### 4.3 Partições da classificação

```text
issue_d:   2007-06 ──── 2011-12 │ 2012-01 ──── 2014-12 │ 2015-01 ── 2015-12
               TREINO           │       EMBARGO        │    TESTE FINAL
   (prazo terminado até 2014-12)│ (prazo termina entre │ (prazo termina em 2018,
                                │  2015 e 2017: rótulo │  antes do cutoff)
                                │  desconhecido em     │
                                │  01/01/2015)         │
```

| Partição | Período (`issue_d`) | Coorte |
|---|---|---|
| Treino + validação | 2007-06 a 2011-12 | 36 meses, `loan_status` ≠ `Current`, não nulo |
| Embargo | 2012-01 a 2014-12 | **Excluído** da classificação: no início de 2015 estes empréstimos ainda não tinham terminado |
| Teste final | 2015-01 a 2015-12 | 36 meses, `loan_status` ≠ `Current`, não nulo |

**Validação:** validação cruzada temporal dentro do período de treino (janela expansível
por ano). Todos os rótulos de treino são conhecidos na data de treino simulada, por isso
podem ser usados para escolher hiperparâmetros; a ordem temporal evita que padrões mais
recentes sejam usados para prever empréstimos mais antigos.

**Desequilíbrio de classes:** qualquer reamostragem (ex.: *oversampling*) é aplicada apenas
dentro dos *folds* de treino. O teste final nunca é reequilibrado.

### 4.4 Limitações

- **Treino mais pequeno**, porque só usa os primeiros anos da plataforma. É o custo de
  cumprir a regra de observabilidade.
- **Mudança temporal**: o modelo é treinado com empréstimos de 2007–2011 e avaliado em
  2015, depois de mudanças na política de crédito e no mercado. Esta diferença faz parte
  do que se pretende avaliar.
- **O rótulo é o estado a 31/12/2018**, e não o estado exato em 01/01/2015. Para empréstimos
  cujo prazo já tinha terminado, a diferença é pequena (ex.: um atraso no fim do prazo que
  mais tarde passou a *Charged Off*).
- **A classe "At risk"** fica rara, porque poucos empréstimos continuam em atraso depois do
  fim do prazo.

---

## 5. Clustering, EDA e testes de hipóteses

| Análise | Dados |
|---|---|
| EDA geral e da regressão | Desenvolvimento (2007-06 a 2014-12) |
| EDA da relação com o alvo da classificação | Treino da classificação (36 meses, 2007-06 a 2011-12) |
| Clustering de mutuários | Desenvolvimento (aceites) |
| Clustering de estados | Desenvolvimento (aceites por `issue_d` + rejeitados por `Application Date`) |
| Testes de hipóteses | Desenvolvimento. Hipóteses sobre o desfecho usam o treino da classificação |

O teste final e o período reservado **nunca** são usados nestas análises.

---

## 6. Outras regras

- **Amostragem reprodutível:** quando for preciso trabalhar com amostras (ex.: UMAP, testes
  com amostras grandes), estas são aleatórias, com semente fixa e documentada, e nunca as
  primeiras linhas do ficheiro.
- **Duplicados:** são detetados pelo `id` do empréstimo na limpeza.
- **Mutuários repetidos:** seriam identificados por `member_id`; a disponibilidade deste
  campo é verificada na limpeza. Se não estiver disponível, um mesmo mutuário pode aparecer
  em partições diferentes, o que é reconhecido como limitação.
- **Transformações aprendidas** (imputação, normalização, *encoding*, PCA, etc.) são
  ajustadas apenas no treino de cada *fold*, através de *pipelines*.
