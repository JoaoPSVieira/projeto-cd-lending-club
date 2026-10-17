# Tabela de exclusões: pedidos rejeitados

_Gerado automaticamente por `notebooks/01b_rejeitados.ipynb` — não editar à mão._

- Gerado em: 2026-10-17
- Tipo "remoção": as linhas deixam de existir. Tipo "partição": as linhas são válidas mas ficam à parte (ex.: teste final, embargo).
- Decisões e justificações: `docs/02_preprocessamento/` e `docs/01_formulacao_problema/`.

## rejected -> desenvolvimento

| N.º | Passo | Tipo | Regra | Justificação | Antes | Removidas | Depois |
|---:|---|---|---|---|---:|---:|---:|
| 1 | Pedidos anteriores ao primeiro aceite | partição | Application Date em 2007-05 | O primeiro empréstimo aceite é de 2007-06: sem aceites no mesmo mês, não há comparação | 27,648,741 | 49 | 27,648,692 |
| 2 | Separar o teste final | partição | Application Date de 2015-01 a 2015-12 | Mesmo corte dos aceites; os rejeitados do teste não são usados (não entram nos modelos) | 27,648,692 | 2,859,379 | 24,789,313 |
| 3 | Separar o período reservado | partição | Application Date de 2016-01 a 2018-12 | Não usado na Parte 1 (mesmo corte dos aceites) | 24,789,313 | 21,339,229 | 3,450,084 |
| 4 | Produtos novos não públicos | remoção | policy_code = 2 | Os aceites são todos policy_code 1: comparar só pedidos dos mesmos produtos | 3,450,084 | 6,241 | 3,443,843 |
| 5 | Pedidos repetidos | remoção | linha igual a uma anterior em todas as colunas (fica a primeira) | O mesmo pedido submetido mais de uma vez no mesmo dia; contaria como várias rejeições | 3,443,843 | 19,615 | 3,424,228 |

**Linhas finais: 3,424,228**
