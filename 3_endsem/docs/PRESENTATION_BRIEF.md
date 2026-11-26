# End-term presentation brief: 7 minutes, 8 slides

Order as in the rubric. One claim per slide, one number per caption, only numbers from `results/`. No outside scores, no
platform names, no "state of the art". Say plainly where the coreset does not beat uniform.

| # | Slide | Show | Say (about 50 s each) |
|---|---|---|---|
| 1 | Data and problem | 1,115 hubs x 42 days; 1,133,640 training rows x 342 columns = 387.7M cells; split diagram (train to 8 May 2015, test 9 May-19 Jun, 46,830 rows) | log1p target, RMSLE; closed days kept as 0 (inflates R²); test window = development fold 4, disclosed |
| 2 | Model | table: LightGBM 0.1022 / R² 0.9992 / 59.7 s vs ridge 0.1592 / 0.9980 / 4.3 s | why trees (interactions); why ridge (theory exact, checks the method) |
| 3 | Sampling + Plot 1 | `figures/fig_plot1_rows` | coreset beats uniform from 5k to 200k (0.1070 vs 0.1102 at 50k), ties at 2k; hub-stratified = uniform; ridge: nothing changes |
| 4 | Time + speed-up | `figures/fig_plot2_time` + speed-up table | 50k coreset 2.75x incl. 10 s build; uniform 5.18x but 0.0032 worse; ridge build 18.6 s > full 4.3 s: no saving. Timings from a shared server |
| 5 | Coreset construction | equation, S = 2.47 / 2.12, alpha_max 2.45 / 1.01, `fig_hist_sensitivity` | sum (not mean) in the denominator so S is in [2, 6]; heuristic for trees, no guarantee |
| 6 | Plots 3-5 | `fig_plot345_lgbm` (+ ridge in backup) | train gap 0.373 vs 0.557 at 50k; Plot 4 floor 1.17 = x* own generalisation ratio; slope ratio 0.66 ridge (theory 0.5), 0.33 trees |
| 7 | Columns + the model behind x* | `fig_plot1_columns`; one line on the frozen forecaster | 40 chosen columns 0.1102 vs 40 random 0.1770; improved forecaster +0.0045 over a like-for-like baseline, gate passed, frozen with hashes |
| 8 | Conclusion | readout: chosen m = 50k; +0.0048 RMSLE; 59.7 s -> 21.6 s; coreset beats uniform by 0.0032 | where it does not help (2k rows; ridge); open item: exclusive timing rerun |

## Likely questions (answers in the report, Section 8)
Why sensitivity sampling for trees? Why ridge? What is the noise floor? Is fold 4 a clean test? Why not tune more? Does the
time split shift? Why is Plot 4 above 1?
