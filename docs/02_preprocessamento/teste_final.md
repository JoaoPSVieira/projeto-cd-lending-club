# Teste final (2015): a mesma limpeza, sem a rever

Prepara o teste final para a avaliação da Parte 2, aplicando-lhe **exatamente as mesmas
regras** de limpeza do desenvolvimento. Responde a dois pedidos do enunciado: *"Define
partitions before data-dependent preprocessing"* e *"preserve a final test set for final
evaluation"*.

Código: `src/lending/cleaner.py` (`AcceptedCleaner`), `config.TEST_PKL`,
`config.TEST_LOG_PKL`. Notebook: `notebooks/01_preprocessamento.ipynb`, secção 11.

---

## 1. Princípio

As regras de limpeza foram decididas **só com o desenvolvimento** (secções 4 a 8 do
notebook). O teste recebe-as tal como estão, pela mesma ordem, sem nenhum ajuste:

| N.º | Regra | Secção | Documento |
|---:|---|---|---|
| 1 | Conversão de tipos (schema fixo) | 4 | `conversao_tipos.md` |
| 2 | 0 meses com contagem 0 → valor em falta ("nunca aconteceu") | 6.1 | `valores_em_falta.md` |
| 3 | Exclusão dos empréstimos com condições modificadas | 8.1 | `valores_suspeitos.md` |
| 4 | Teto 9 999 999 → valor em falta | 8.2 | `valores_suspeitos.md` |

As quatro regras são **fixas e aplicadas linha a linha**: nenhuma calcula uma estatística
dos dados (médias, quantis, categorias frequentes). Por isso aplicá-las ao teste não é
*leakage*. O que o enunciado proíbe é:
- aprender transformações com o teste (*fit* de imputação, normalização, *encoding*, etc.),
  o que só acontece no pipeline, com o treino de cada *fold*;
- mudar decisões olhando para o teste.

**O que não se faz com o teste:** não se mostra nenhuma estatística, distribuição ou
exemplo. Só se registam quantas linhas cada regra removeu e quantos valores corrigiu,
porque o enunciado pede que todas as exclusões fiquem documentadas.

---

## 2. Garantia de que as regras são as mesmas

As regras estavam espalhadas por células do notebook. A classe `AcceptedCleaner` junta-as
num só objeto, e a exclusão da secção 8.1 do desenvolvimento passou a usar a mesma classe.

Antes de a aplicar ao teste, o notebook prova que a classe faz exatamente o que as secções
4 a 8 fizeram. Aplicada ao desenvolvimento em texto (`accepted_dev_raw.pkl`), dá um
conjunto **idêntico, célula a célula**, ao desenvolvimento limpo
(`pd.testing.assert_frame_equal`), com a mesma exclusão (221) e as mesmas correções
(1 275 + 820 zeros disfarçados, 3 + 2 tetos).

---

## 3. Resultado

| Passo | Antes | Removidas | Depois |
|---|---:|---:|---:|
| Linhas que não são empréstimos | 2 260 701 | 33 | 2 260 668 |
| Separar o desenvolvimento e o período reservado | 2 260 668 | 1 839 573 | 421 095 |
| Condições modificadas depois da concessão | 421 095 | 207 | **420 888** |

| Correção (valor → em falta) | Valores corrigidos |
|---|---:|
| `mths_since_last_record` = 0 com `pub_rec` = 0 | 0 |
| `mths_since_last_delinq` = 0 com `delinq_2yrs` = 0 | 1 |
| `tot_hi_cred_lim` = 9 999 999 | 3 |
| `total_rev_hi_lim` = 9 999 999 | 1 |

Os empréstimos modificados são 0,05% do teste, a mesma proporção do desenvolvimento (221 em
466 345). Os zeros disfarçados quase desaparecem em 2015, o que é coerente com o
desenvolvimento, onde estavam concentrados em 2007–2008.

O teste limpo fica em `data/processed/teste_2015.pkl` (420 888 × 151, mesmos tipos do
`aceites_dev.pkl`) e o registo em `accepted_test_exclusions.pkl`. A tabela de exclusões
(`tabela_exclusoes.md`) tem agora uma secção para o teste final.

---

## 4. O que fica para a avaliação final

- As **coortes do teste** (regressão e classificação) não são construídas agora. Na
  avaliação final aplica-se o mesmo `CohortBuilder` com `partition="test"`. Na
  classificação, o passo "desfecho desconhecido" (`Current`) pode remover empréstimos, e os
  *At risk* que existam contam como *Default*.
- O pipeline de pré-processamento (imputação, *encoding*, padronização) é ajustado no
  treino e só **aplicado** (*transform*) ao teste.
