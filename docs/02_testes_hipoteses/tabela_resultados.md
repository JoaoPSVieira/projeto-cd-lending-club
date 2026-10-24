# Testes de hipóteses: tabela de resultados

_Gerado automaticamente por `R/testes_hipoteses.Rmd` - não editar à mão._

- Gerado em: 2026-10-24
- Só dados de desenvolvimento; o teste final (2015) não é usado.
- p-valores com a correção de Holm (4 testes, alfa = 0.05).
- IC 95% por bootstrap (percentis; 2000 réplicas, 500 nos testes com milhões de linhas; semente 42).
- Magnitude: rank-biserial pelos limiares de Romano et al. (2006); V de Cramér pelos de Cohen (1988).
- Justificação de cada teste, pressupostos e interpretação: `reports/testes_hipoteses.html`.

|Hipótese                                        |Dados                                                 |n                                   |Teste                         |p (Holm)  |Efeito [IC 95%]                         |Magnitude   |Decisão    |
|:-----------------------------------------------|:-----------------------------------------------------|:-----------------------------------|:-----------------------------|:---------|:---------------------------------------|:-----------|:----------|
|(a) DTI maior nos Default                       |treino da classificação (2007-06 a 2011-12)           |3876 Default / 27646 Good           |Mann-Whitney unilateral       |2.4e-11   |rank-biserial = 0.065 [0.045; 0.084]    |desprezável |rejeita H0 |
|(b) purpose associado ao default                |treino da classificação (2007-06 a 2011-12)           |31522 empréstimos, 14 purposes      |Qui-quadrado de independência |< 2.2e-16 |V de Cramér = 0.104 [0.093; 0.118]      |pequeno     |rejeita H0 |
|(c1) montante difere entre aceites e rejeitados |aceites e rejeitados (2007-06 a 2014-12)              |466124 aceites / 3424226 rejeitados |Mann-Whitney bilateral        |< 2.2e-16 |rank-biserial = 0.197 [0.195; 0.198]    |pequeno     |rejeita H0 |
|(c2) risk score menor nos rejeitados            |aceites e rejeitados (2007-06 a 2013-10, escala FICO) |200882 aceites / 1267436 rejeitados |Mann-Whitney unilateral       |< 2.2e-16 |rank-biserial = -0.614 [-0.615; -0.612] |grande      |rejeita H0 |
