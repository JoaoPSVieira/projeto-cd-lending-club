# Regressão: taxa de juro em T0

Bloco 4 do plano da Parte 2: grelhas, comparação e interpretação da regressão, só com dados de
desenvolvimento (2013–2014). O teste de 2015 fica para o bloco 8.

| Ficheiro | Conteúdo |
|---|---|
| `scripts/run_cv.py` | Corre a grelha no terminal; retoma onde parou |
| `notebooks/04b_regressao.ipynb` | Tabela de comparação, afinação, folds, custo e interpretação |
| `src/lending/model_analysis.py` | `CVAnalysis`: as tabelas e os gráficos (servem também o 04c) |
| `src/lending/experiment.py` | Novo `fold_pipeline` (modelo de um fold, para a interpretação) |
| `tests/test_model_analysis.py` | Testes do `CVAnalysis`, do `fold_pipeline` e do script |

---

## 1. Correr a grelha

```
python scripts/run_cv.py regression
```

São 40 configurações (mais 2 *baselines*) × 6 folds = 252 ajustes. Antes de correr a grelha, o
`teste_rapido.py` dá uma estimativa do tempo de cada família nesta máquina.

- **Interromper e continuar:** Ctrl+C e o mesmo comando outra vez. O que já está no CSV não se repete.
- **Por partes:** `--models "Ridge,LightGBM"` corre só esses; `--exclude "Random Forest,MLP"` corre
  todos menos esses. Por exemplo, os modelos rápidos primeiro e a Random Forest e a MLP de noite.
- **Durante a noite:** no Windows, impedir que o computador suspenda (Definições → Sistema →
  Energia e bateria → Ecrã e suspensão: "Nunca" com o carregador ligado).
- **Orçamento (secção 3 do relatório):** cerca de 1 hora por família. Se uma família passar
  muito disso, reportar o tempo real e, se for preciso, reduzir a grelha dessa família e dizê-lo
  no relatório.

Os resultados ficam em `reports/results/regression_cv.csv`, que entra no Git, e em
`data/processed/oof/regression/`, que fica fora.

---

## 2. O que o notebook 04b produz

| Secção | Tabela ou figura | Ficheiro da figura |
|---|---|---|
| 1. Dados e folds | Tamanho do treino e da validação de cada fold (e do kNN) | — |
| 3. Comparação | Melhor configuração de cada modelo, MAE, RMSE e R² (média ± desvio), melhoria face às *baselines* | — |
| 4. Afinação | Todas as configurações de cada modelo; efeito de cada hiperparâmetro | `mod_reg_afinacao.png` |
| 5. Folds | MAE de cada trimestre de validação | `mod_reg_mae_por_fold.png` |
| 6. Custo | Ajustes, linhas de treino e tempos por modelo | — |
| 7.1 | Previsto contra real (melhor modelo) | `mod_reg_previsto_real.png` |
| 7.2 | Viés por mês de emissão | `mod_reg_vies_tempo.png` |
| 7.3 | Erro por prazo e viés por sub-grau | `mod_reg_vies_subgrau.png` |
| 7.4 | Importância por permutação (último fold) | `mod_reg_importancia.png` |
| 7.5 | Coeficientes do Ridge | `mod_reg_ridge_coef.png` |
| 8. Resumo | Os números principais para o relatório | — |

---

## 3. Decisões da interpretação

- **Só previsões fora do fold.** Todos os gráficos de resíduos usam as previsões da validação
  cruzada, nunca o teste.
- **O sub-grau entra só como diagnóstico.** É atribuído depois do pedido, por isso não é um
  preditor da regressão. Usá-lo para agrupar os erros mostra onde o perfil do pedido não
  chega para reconstruir o preço, sem o usar no modelo.
- **Importância por permutação nas colunas originais**, na validação do último fold, com uma
  amostra de 20 000 empréstimos e 3 permutações por coluna. Baralhar uma coluna original também
  baralha as *features* novas que dependem dela, por isso a importância é a da variável inteira.
  O modelo é o que a validação cruzada avaliou nesse fold (`fold_pipeline`).
- **Coeficientes do Ridge** na melhor configuração treinada em todo o treino. Com as colunas
  padronizadas, cada coeficiente é a variação da taxa, em p.p., por um desvio-padrão da coluna.

---

## 4. O que segue para o relatório e para o bloco 8

- A tabela da secção 3 é a tabela de comparação da regressão pedida no enunciado.
- O melhor modelo e a segunda melhor família seguem para a avaliação final (bloco 8):
  `runner.refit(data, spec)` com a melhor configuração, no teste de 2015, com bootstrap e o
  bootstrap emparelhado entre os dois.
- A secção 8 do notebook imprime os números principais e lista os pontos a discutir.
