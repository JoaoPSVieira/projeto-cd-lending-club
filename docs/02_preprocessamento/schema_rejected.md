# Schema: rejected, desenvolvimento

_Gerado automaticamente por `notebooks/01b_rejeitados.ipynb` - não editar à mão._

- Conjunto descrito: rejected, desenvolvimento, 3,450,084 linhas x 9 colunas
- Gerado em: 2026-10-24
- Tipos e unidades: `src/lending/schema.py`. Significado de cada coluna: `docs/01_formulacao_problema/elegibilidade_colunas.md`.
- Mín./máx.: valores observados neste conjunto (vazio se a coluna não tiver valores).
- Exemplos: as categorias mais frequentes.

## Colunas por unidade

| Unidade | Colunas |
|---|---:|
| categoria | 2 |
| USD | 1 |
| data (dia) | 1 |
| texto | 1 |
| pontos | 1 |
| % | 1 |
| anos | 1 |
| código | 1 |

## Colunas

| Coluna | Tipo | Unidade | dtype | Mín. | Máx. | Distintos | Exemplos |
|---|---|---|---|---:|---:|---:|---|
| `amount_requested` | number | USD | float64 | 0 | 1,400,000 | 2,085 |  |
| `application_date` | date | data (dia) | datetime64[ns] | 2007-06 | 2014-12 | 2,771 |  |
| `title` | text | texto | str |  |  | 72,325 |  |
| `risk_score` | number | pontos | float64 | 0 | 990 | 688 |  |
| `dti` | percent | % | float64 | -1 | 50,000,031.49 | 47,445 |  |
| `zip_code` | category | categoria | str |  |  | 998 | 112xx, 300xx, 770xx |
| `addr_state` | category | categoria | str |  |  | 51 | CA, TX, NY |
| `emp_length` | emp_length | anos | float64 | 0 | 10 | 11 |  |
| `policy_code` | number | código | float64 | 0 | 2 | 2 |  |
