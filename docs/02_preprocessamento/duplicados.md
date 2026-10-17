# Duplicados e mutuários repetidos

Segunda parte da análise de qualidade. Responde ao pedido do enunciado: *"Address
identifiable repeated borrowers and duplicate loans (but explain limitations when
identifiers are unavailable)"*.

Código: `src/lending/quality.py` (`DuplicateReport`, `REPEAT_BORROWER_KEY`). Notebook:
`notebooks/01_preprocessamento.ipynb`, secções 2 (unicidade do `id` no ficheiro inteiro) e 7.

---

## 1. Dois problemas diferentes

| Problema | O que é | Porque importa |
|---|---|---|
| **Empréstimo duplicado** | O mesmo empréstimo em mais do que uma linha | Inflaciona os dados; se as cópias caírem em partições diferentes, o modelo é avaliado com casos que já viu (fuga de informação) |
| **Mutuário repetido** | A mesma pessoa com vários empréstimos | Não é um erro (são empréstimos reais), mas as linhas deixam de ser independentes: a mesma pessoa pode aparecer no treino e na validação |

---

## 2. Empréstimos duplicados

| Verificação | Regra | Onde | Resultado |
|---|---|---|---|
| `id` repetido | o mesmo `id` em mais do que uma linha | ficheiro inteiro (2 260 668 empréstimos) | **0** |
| `url` de outro empréstimo | o `loan_id` no fim do `url` difere do `id` | desenvolvimento | **0** |
| Linha repetida | todas as colunas iguais, exceto `id` e `url` | desenvolvimento | **0** |

**Conclusão:** não há empréstimos duplicados. Como os `id` são únicos no ficheiro inteiro,
nenhum empréstimo pode estar em duas partições (desenvolvimento e teste). Nenhuma linha é
removida neste passo.

A verificação no ficheiro inteiro só conta identificadores; não olha para nenhum valor do
teste.

---

## 3. Mutuários repetidos

### 3.1 O identificador não existe

O `member_id` está **vazio em todo o ficheiro** (0 valores preenchidos em 2 260 701 linhas):
a LendingClub retirou-o da exportação pública. Não é possível identificar com certeza os
empréstimos de uma mesma pessoa.

### 3.2 Estimativa com uma chave aproximada

Para ter uma ordem de grandeza, procuram-se empréstimos que coincidem em atributos que mudam
pouco entre dois pedidos da mesma pessoa (`REPEAT_BORROWER_KEY`):

| Coluna | Porquê |
|---|---|
| `earliest_cr_line` | Mês da primeira linha de crédito: fixo para cada pessoa |
| `zip_code`, `addr_state` | Localização (3 primeiros dígitos do código postal) |
| `annual_inc`, `emp_title`, `home_ownership` | Perfil declarado |

Resultado no desenvolvimento:

| Medida | Valor |
|---|---:|
| Empréstimos com a chave completa | 438 729 |
| Empréstimos em grupos com a mesma chave | 2 106 (**0,48%**) |
| Grupos (possíveis mutuários repetidos) | 1 049 (1 041 com 2 empréstimos, 8 com 3) |
| Intervalo mediano entre o 1.º e o último empréstimo do grupo | 8 meses |
| Grupos com todos os empréstimos no mesmo mês | 39 |
| Grupos com empréstimos até 2011 e a partir de 2012 | 128 |

O intervalo mediano de 8 meses é coerente com uma pessoa que volta a pedir um empréstimo
depois de algum tempo. Os 39 grupos com os dois empréstimos no mesmo mês são, mais
provavelmente, pessoas diferentes com o mesmo perfil.

### 3.3 Limitações da estimativa

- **Falsos negativos:** se a pessoa mudar de emprego, de rendimento ou de casa entre os dois
  pedidos, a chave muda e a repetição não é detetada. Por isso 0,48% tende a **subestimar**.
- **Falsos positivos:** duas pessoas diferentes podem coincidir em todas as colunas (o código
  postal só tem 3 dígitos).
- Ficam de fora os empréstimos com alguma coluna da chave vazia (sobretudo `emp_title`).

### 3.4 Decisão

- **Nenhum empréstimo é removido:** os empréstimos de mutuários repetidos são reais e válidos,
  e a chave não é fiável o suficiente para decidir quem é a mesma pessoa.
- **A chave não é usada como *feature*** nem para agrupar a validação.
- **Limitação reconhecida:** a mesma pessoa pode aparecer no treino e na validação (ou no
  teste). Os cortes temporais reduzem o problema, porque os empréstimos da mesma pessoa
  estão, em mediana, a 8 meses de distância e ficam muitas vezes em períodos diferentes, mas
  não o eliminam. Uma validação agrupada por mutuário exigiria o `member_id`. Com cerca de
  0,5% de empréstimos afetados, o efeito esperado nas métricas é pequeno.
