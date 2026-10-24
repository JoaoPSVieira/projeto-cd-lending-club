# Schema: accepted, desenvolvimento

_Gerado automaticamente por `notebooks/01_preprocessamento.ipynb` - não editar à mão._

- Conjunto descrito: accepted, desenvolvimento, 466,345 linhas x 151 colunas
- Gerado em: 2026-10-24
- Tipos e unidades: `src/lending/schema.py`. Significado de cada coluna: `docs/01_formulacao_problema/elegibilidade_colunas.md`.
- Mín./máx.: valores observados neste conjunto (vazio se a coluna não tiver valores).
- Exemplos: as categorias mais frequentes.

## Colunas por unidade

| Unidade | Colunas |
|---|---:|
| contagem | 42 |
| USD | 33 |
| meses | 17 |
| categoria | 17 |
| % | 11 |
| data (mês) | 11 |
| pontos FICO | 6 |
| texto | 5 |
| booleano | 3 |
| USD/ano | 2 |
| identificador | 1 |
| anos | 1 |
| código | 1 |
| dias | 1 |

## Colunas

| Coluna | Tipo | Unidade | dtype | Mín. | Máx. | Distintos | Exemplos |
|---|---|---|---|---:|---:|---:|---|
| `id` | id | identificador | int64 |  |  | 466,345 |  |
| `member_id` | text | texto | str |  |  | 0 |  |
| `loan_amnt` | number | USD | float64 | 500 | 35,000 | 1,352 |  |
| `funded_amnt` | number | USD | float64 | 500 | 35,000 | 1,354 |  |
| `funded_amnt_inv` | number | USD | float64 | 0 | 35,000 | 9,843 |  |
| `term` | term | meses | float64 | 36 | 60 | 2 |  |
| `int_rate` | number | % | float64 | 5.42 | 26.06 | 507 |  |
| `installment` | number | USD | float64 | 4.93 | 1,409.99 | 55,656 |  |
| `grade` | category | categoria | str |  |  | 7 | B, C, D |
| `sub_grade` | category | categoria | str |  |  | 35 | B3, B4, C1 |
| `emp_title` | text | texto | str |  |  | 201,059 |  |
| `emp_length` | emp_length | anos | float64 | 0 | 10 | 11 |  |
| `home_ownership` | category | categoria | str |  |  | 6 | MORTGAGE, RENT, OWN |
| `annual_inc` | number | USD/ano | float64 | 1,896 | 7,500,000 | 31,906 |  |
| `verification_status` | category | categoria | str |  |  | 3 | Verified, Source Verified, Not Verified |
| `issue_d` | month_year | data (mês) | datetime64[ns] | 2007-06 | 2014-12 | 91 |  |
| `loan_status` | category | categoria | str |  |  | 9 | Fully Paid, Charged Off, Current |
| `pymnt_plan` | flag | booleano | boolean |  |  | 2 | False, True |
| `url` | text | texto | str |  |  | 466,345 |  |
| `desc` | text | texto | str |  |  | 124,435 |  |
| `purpose` | category | categoria | str |  |  | 14 | debt_consolidation, credit_card, home_improvement |
| `title` | text | texto | str |  |  | 61,445 |  |
| `zip_code` | category | categoria | str |  |  | 888 | 945xx, 112xx, 750xx |
| `addr_state` | category | categoria | str |  |  | 50 | CA, NY, TX |
| `dti` | number | % | float64 | 0 | 39.99 | 3,997 |  |
| `delinq_2yrs` | number | contagem | float64 | 0 | 29 | 24 |  |
| `earliest_cr_line` | month_year | data (mês) | datetime64[ns] | 1944-01 | 2011-11 | 664 |  |
| `fico_range_low` | number | pontos FICO | float64 | 610 | 845 | 48 |  |
| `fico_range_high` | number | pontos FICO | float64 | 614 | 850 | 48 |  |
| `inq_last_6mths` | number | contagem | float64 | 0 | 33 | 28 |  |
| `mths_since_last_delinq` | number | meses | float64 | 0 | 188 | 145 |  |
| `mths_since_last_record` | number | meses | float64 | 0 | 129 | 123 |  |
| `open_acc` | number | contagem | float64 | 0 | 84 | 62 |  |
| `pub_rec` | number | contagem | float64 | 0 | 63 | 26 |  |
| `revol_bal` | number | USD | float64 | 0 | 2,568,995 | 58,143 |  |
| `revol_util` | number | % | float64 | 0 | 892.30 | 1,269 |  |
| `total_acc` | number | contagem | float64 | 1 | 156 | 112 |  |
| `initial_list_status` | category | categoria | str |  |  | 2 | f, w |
| `out_prncp` | number | USD | float64 | 0 | 13,227.60 | 9,493 |  |
| `out_prncp_inv` | number | USD | float64 | 0 | 13,227.60 | 9,647 |  |
| `total_pymnt` | number | USD | float64 | 0 | 63,296.88 | 445,363 |  |
| `total_pymnt_inv` | number | USD | float64 | 0 | 63,296.88 | 377,851 |  |
| `total_rec_prncp` | number | USD | float64 | 0 | 35,000.02 | 82,193 |  |
| `total_rec_int` | number | USD | float64 | 0 | 28,192.50 | 294,761 |  |
| `total_rec_late_fee` | number | USD | float64 | -7.0e-10 | 731.84 | 9,356 |  |
| `recoveries` | number | USD | float64 | 0 | 39,444.37 | 52,033 |  |
| `collection_recovery_fee` | number | USD | float64 | 0 | 7,002.19 | 51,657 |  |
| `last_pymnt_d` | month_year | data (mês) | datetime64[ns] | 2007-12 | 2019-03 | 136 |  |
| `last_pymnt_amnt` | number | USD | float64 | 0 | 36,234.44 | 275,918 |  |
| `next_pymnt_d` | month_year | data (mês) | datetime64[ns] | 2008-01 | 2019-04 | 100 |  |
| `last_credit_pull_d` | month_year | data (mês) | datetime64[ns] | 2007-05 | 2019-03 | 140 |  |
| `last_fico_range_high` | number | pontos FICO | float64 | 0 | 850 | 72 |  |
| `last_fico_range_low` | number | pontos FICO | float64 | 0 | 845 | 71 |  |
| `collections_12_mths_ex_med` | number | contagem | float64 | 0 | 20 | 9 |  |
| `mths_since_last_major_derog` | number | meses | float64 | 0 | 188 | 152 |  |
| `policy_code` | number | código | float64 | 1 | 1 | 1 |  |
| `application_type` | category | categoria | str |  |  | 1 | Individual |
| `annual_inc_joint` | number | USD/ano | float64 |  |  | 0 |  |
| `dti_joint` | number | % | float64 |  |  | 0 |  |
| `verification_status_joint` | category | categoria | str |  |  | 0 |  |
| `acc_now_delinq` | number | contagem | float64 | 0 | 5 | 6 |  |
| `tot_coll_amt` | number | USD | float64 | 0 | 9,152,545 | 6,321 |  |
| `tot_cur_bal` | number | USD | float64 | 0 | 8,000,078 | 220,705 |  |
| `open_acc_6m` | number | contagem | float64 |  |  | 0 |  |
| `open_act_il` | number | contagem | float64 |  |  | 0 |  |
| `open_il_12m` | number | contagem | float64 |  |  | 0 |  |
| `open_il_24m` | number | contagem | float64 |  |  | 0 |  |
| `mths_since_rcnt_il` | number | meses | float64 |  |  | 0 |  |
| `total_bal_il` | number | USD | float64 |  |  | 0 |  |
| `il_util` | number | % | float64 |  |  | 0 |  |
| `open_rv_12m` | number | contagem | float64 |  |  | 0 |  |
| `open_rv_24m` | number | contagem | float64 |  |  | 0 |  |
| `max_bal_bc` | number | USD | float64 |  |  | 0 |  |
| `all_util` | number | % | float64 |  |  | 0 |  |
| `total_rev_hi_lim` | number | USD | float64 | 0 | 9,999,999 | 14,612 |  |
| `inq_fi` | number | contagem | float64 |  |  | 0 |  |
| `total_cu_tl` | number | contagem | float64 |  |  | 0 |  |
| `inq_last_12m` | number | contagem | float64 |  |  | 0 |  |
| `acc_open_past_24mths` | number | contagem | float64 | 0 | 53 | 39 |  |
| `avg_cur_bal` | number | USD | float64 | 0 | 958,084 | 55,981 |  |
| `bc_open_to_buy` | number | USD | float64 | 0 | 497,445 | 47,224 |  |
| `bc_util` | number | % | float64 | 0 | 339.60 | 1,294 |  |
| `chargeoff_within_12_mths` | number | contagem | float64 | 0 | 7 | 8 |  |
| `delinq_amnt` | number | USD | float64 | 0 | 70,076 | 727 |  |
| `mo_sin_old_il_acct` | number | meses | float64 | 0 | 649 | 463 |  |
| `mo_sin_old_rev_tl_op` | number | meses | float64 | 3 | 842 | 691 |  |
| `mo_sin_rcnt_rev_tl_op` | number | meses | float64 | 0 | 372 | 215 |  |
| `mo_sin_rcnt_tl` | number | meses | float64 | 0 | 226 | 143 |  |
| `mort_acc` | number | contagem | float64 | 0 | 37 | 36 |  |
| `mths_since_recent_bc` | number | meses | float64 | 0 | 616 | 402 |  |
| `mths_since_recent_bc_dlq` | number | meses | float64 | 0 | 170 | 140 |  |
| `mths_since_recent_inq` | number | meses | float64 | 0 | 25 | 26 |  |
| `mths_since_recent_revol_delinq` | number | meses | float64 | 0 | 180 | 148 |  |
| `num_accts_ever_120_pd` | number | contagem | float64 | 0 | 33 | 30 |  |
| `num_actv_bc_tl` | number | contagem | float64 | 0 | 30 | 28 |  |
| `num_actv_rev_tl` | number | contagem | float64 | 0 | 38 | 37 |  |
| `num_bc_sats` | number | contagem | float64 | 0 | 35 | 36 |  |
| `num_bc_tl` | number | contagem | float64 | 0 | 65 | 56 |  |
| `num_il_tl` | number | contagem | float64 | 0 | 150 | 85 |  |
| `num_op_rev_tl` | number | contagem | float64 | 0 | 62 | 56 |  |
| `num_rev_accts` | number | contagem | float64 | 0 | 105 | 85 |  |
| `num_rev_tl_bal_gt_0` | number | contagem | float64 | 0 | 38 | 36 |  |
| `num_sats` | number | contagem | float64 | 0 | 84 | 62 |  |
| `num_tl_120dpd_2m` | number | contagem | float64 | 0 | 2 | 3 |  |
| `num_tl_30dpd` | number | contagem | float64 | 0 | 4 | 5 |  |
| `num_tl_90g_dpd_24m` | number | contagem | float64 | 0 | 24 | 22 |  |
| `num_tl_op_past_12m` | number | contagem | float64 | 0 | 26 | 26 |  |
| `pct_tl_nvr_dlq` | number | % | float64 | 15 | 100 | 480 |  |
| `percent_bc_gt_75` | number | % | float64 | 0 | 100 | 180 |  |
| `pub_rec_bankruptcies` | number | contagem | float64 | 0 | 12 | 10 |  |
| `tax_liens` | number | contagem | float64 | 0 | 63 | 24 |  |
| `tot_hi_cred_lim` | number | USD | float64 | 0 | 9,999,999 | 225,960 |  |
| `total_bal_ex_mort` | number | USD | float64 | 0 | 2,688,920 | 117,515 |  |
| `total_bc_limit` | number | USD | float64 | 0 | 1,090,700 | 12,190 |  |
| `total_il_high_credit_limit` | number | USD | float64 | 0 | 1,241,783 | 104,846 |  |
| `revol_bal_joint` | number | USD | float64 |  |  | 0 |  |
| `sec_app_fico_range_low` | number | pontos FICO | float64 |  |  | 0 |  |
| `sec_app_fico_range_high` | number | pontos FICO | float64 |  |  | 0 |  |
| `sec_app_earliest_cr_line` | month_year | data (mês) | datetime64[ns] |  |  | 0 |  |
| `sec_app_inq_last_6mths` | number | contagem | float64 |  |  | 0 |  |
| `sec_app_mort_acc` | number | contagem | float64 |  |  | 0 |  |
| `sec_app_open_acc` | number | contagem | float64 |  |  | 0 |  |
| `sec_app_revol_util` | number | % | float64 |  |  | 0 |  |
| `sec_app_open_act_il` | number | contagem | float64 |  |  | 0 |  |
| `sec_app_num_rev_accts` | number | contagem | float64 |  |  | 0 |  |
| `sec_app_chargeoff_within_12_mths` | number | contagem | float64 |  |  | 0 |  |
| `sec_app_collections_12_mths_ex_med` | number | contagem | float64 |  |  | 0 |  |
| `sec_app_mths_since_last_major_derog` | number | meses | float64 |  |  | 0 |  |
| `hardship_flag` | flag | booleano | boolean |  |  | 2 | False, True |
| `hardship_type` | category | categoria | str |  |  | 1 | INTEREST ONLY-3 MONTHS DEFERRAL |
| `hardship_reason` | category | categoria | str |  |  | 9 | NATURAL_DISASTER, EXCESSIVE_OBLIGATIONS, UNEMPLOYMENT |
| `hardship_status` | category | categoria | str |  |  | 3 | COMPLETED, BROKEN, ACTIVE |
| `deferral_term` | number | meses | float64 | 3 | 3 | 1 |  |
| `hardship_amount` | number | USD | float64 | 1.47 | 382.34 | 571 |  |
| `hardship_start_date` | month_year | data (mês) | datetime64[ns] | 2017-02 | 2019-03 | 25 |  |
| `hardship_end_date` | month_year | data (mês) | datetime64[ns] | 2017-04 | 2019-06 | 27 |  |
| `payment_plan_start_date` | month_year | data (mês) | datetime64[ns] | 2017-02 | 2019-04 | 26 |  |
| `hardship_length` | number | meses | float64 | 3 | 3 | 1 |  |
| `hardship_dpd` | number | dias | float64 | 0 | 32 | 32 |  |
| `hardship_loan_status` | category | categoria | str |  |  | 4 | Late (16-30 days), In Grace Period, Current |
| `orig_projected_additional_accrued_interest` | number | USD | float64 | 4.41 | 1,147.02 | 459 |  |
| `hardship_payoff_balance_amount` | number | USD | float64 | 174.15 | 21,750.75 | 582 |  |
| `hardship_last_payment_amount` | number | USD | float64 | 0.04 | 991.79 | 574 |  |
| `disbursement_method` | category | categoria | str |  |  | 1 | Cash |
| `debt_settlement_flag` | flag | booleano | boolean |  |  | 2 | False, True |
| `debt_settlement_flag_date` | month_year | data (mês) | datetime64[ns] | 2010-02 | 2019-03 | 83 |  |
| `settlement_status` | category | categoria | str |  |  | 3 | COMPLETE, ACTIVE, BROKEN |
| `settlement_date` | month_year | data (mês) | datetime64[ns] | 2009-03 | 2019-03 | 90 |  |
| `settlement_amount` | number | USD | float64 | 44.21 | 33,601 | 5,443 |  |
| `settlement_percentage` | number | % | float64 | 0.20 | 521.35 | 1,224 |  |
| `settlement_term` | number | meses | float64 | 0 | 36 | 27 |  |
