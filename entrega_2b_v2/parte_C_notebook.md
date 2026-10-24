---
# Parte C — Primeiros *insights* (base da secção 2b do relatório)

Só dados de desenvolvimento (o teste de 2015 não foi usado). Para cada ponto: **o que se vê → o que significa → o que muda nas fases seguintes**. Os valores com "cerca de" foram lidos nos gráficos. As associações são descritivas e não implicam causalidade.

**Distribuições**
1. Variáveis monetárias com caudas muito longas (`annual_inc`: assimetria 26,2, máximo 7,5 M USD) → log10(1 + x) e limitação aos percentis 0,5–99,5 do treino, no pipeline.
2. Contagens de incidentes quase sempre 0 (80,8% sem `delinq_2yrs`, 84,5% sem `pub_rec`); o critério de Tukey não se aplica (IQR = 0) → indicadores binários.
3. Limites de política visíveis (montante ≤ 35 000 USD, DTI < 40, FICO ≥ 660 em 2013–2014) → só observamos empréstimos aceites (limitação).

**Regressão (`int_rate`)**
4. Só três variáveis têm |ρ| > 0,3 com a taxa: FICO (−0,42), `bc_open_to_buy` (−0,38) e `bc_util` (0,32). A mediana desce de cerca de 15,5% para cerca de 8% ao longo do FICO, mas com muita dispersão → é preciso um modelo multivariado.
5. Prazo: taxa média de 17,1% a 60 meses contra 12,8% a 36 meses (+4,3 p.p.) → a *baseline* "mediana por `term`" é relevante. A finalidade também conta (`credit_card` 12,5%; `small_business` 18,5%), mas com muita variação dentro de cada finalidade.

**Classificação**
6. Desequilíbrio de 87,7% / 12,3% → F1 *macro*, AUC e *recall* de Default; balanceamento só nos *folds* de treino.
7. O default sobe de forma monotónica com o grau: 5,9% (A) → 35,9% (G), 6,1× → a *baseline* por `grade` é exigente.
8. Cada variável sozinha separa pouco: as distribuições das duas classes sobrepõem-se muito, e nenhuma correlação com o default passa 0,3 → expectativas moderadas.
9. Relações não lineares: no FICO, o default desce de cerca de 21% para cerca de 4%, com a maior queda logo no 1.º decil; desce com o rendimento até ~100 000 USD; tem forma de J na utilização → árvores/*ensembles*, ou transformações nos modelos lineares.

**Associações e redundância**
10. Pares quase redundantes: `tot_cur_bal`–`tot_hi_cred_lim` (0,98), `revol_util`–`bc_util` (0,81), `loan_amnt`–`installment` (0,99) e `fico_range_low`–`int_rate` (−0,81) → seleção de features ou regularização. O montante está associado ao rendimento (0,50) → rácios face ao rendimento.

**Mudança da população**
11. De 2007–2011 para 2013–2014: FICO mediano de 710 para 685, DTI de 13,1 para 17,3 e utilização de 48,1% para 57,8% → justifica a divisão temporal; o classificador treinado em 2007–2011 pode perder desempenho em 2015 (limitação).

**PCA vs UMAP**
12. PCA: 10 de 20 componentes (regressão) e 7 de 14 (classificação) para 80% da variância. Nas duas tarefas, a PC1 é a dimensão financeira (limites, saldos, contas) e a PC2 é a pressão sobre o crédito (utilização vs FICO); a taxa varia sobretudo ao longo da PC2.
13. O UMAP forma ilhas, mas não organiza a taxa nem as classes melhor do que o PCA; os defaults aparecem em todas as zonas → não há estrutura não linear forte; a classificação vai ser difícil.
