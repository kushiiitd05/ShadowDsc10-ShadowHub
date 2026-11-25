# End-term and improvement plan v2 (2026-11-24)

Supersedes the "Improvement work (later)" notes in `STATUS.md`. `improvement_plan/IMPROVEMENT_PLAN.md` stays the long-form phase list; this file records what changed after the 24 Nov verification round and what is running.
Everything marked VERIFIED below was recomputed locally from the raw files, the committed metrics or the code. Anything else is labelled as a claim from an outside review and is not yet confirmed.

## 1. State

| Item | State |
|---|---|
| Best earlier model | CatBoost + LightGBM + LightGBM-Tweedie, log-space mean, seed `20260807`, earlier public score 0.06976 (RMSLE) |
| Model weights | **Not saved by any script and not found anywhere** (repo, Kaggle kernel output, Mac, both GPU servers). The 4.5 h run must be redone to get weights |
| Control rerun | Running on one GPU server (CPU GBDT); data rebuilt and verified (see section 9). Same code, seed and pins (Python 3.11, numpy 1.25.2, pandas 2.3.3, scikit-learn 1.7.2, CatBoost 1.2.10, LightGBM 4.7.0; pyarrow pinned to 17.0.0). Rebuilds all 37 origins and 4 folds, then fits 3 models, saves each model with a SHA-256, timings and a control submission. Script: `endterm/control_rerun.py` |
| Raw data check | Server copies of the four raw CSVs match the repo SHA-256 (`orders_train.csv` = `bc0bf252...1e6`) |
| Phases 0-5 of the improvement plan | Not started |
| End-term experiments | Not started; design fixed in section 4 |

## 2. Corrections found by verification (VERIFIED unless stated)

1. **Test-window collision.** Training dates are 2013-01-01..2015-06-19 (900 days). The last 42 days, 2015-05-09..2015-06-19, are exactly CV fold 4. An 84-day test would cover folds 3 and 4. Any earlier choice made with those folds (feature groups, early stopping, offsets) has seen those dates. Decision needed (section 6).
2. **Open-but-zero hypothesis is refuted as the cause of the CV gap.** The train file has 54 open-but-zero rows in 41 hubs; only 3 fall in 2015. Each fold holds 0 or 1 such row. Removing it lowers RMSLE by only 0.0065-0.0074 per fold (LightGBM: fold 1 0.1250 to 0.1176, fold 2 0.1221 to 0.1155, fold 3 unchanged 0.1450, fold 4 0.1316 to 0.1251). Dropping all zero rows (closed rows included) raises RMSLE, because closed rows carry zero error and dilute the mean. **The CV-versus-score gap (about 0.12-0.14 vs 0.06976) remains unexplained.** Keep reporting `rmsle` and `rmsle_excl_open0`, and do not "fix" CV by dropping zero rows.
3. **Earlier status text had wrong numbers.** On the pooled "all" rows the hub-mean baseline is 0.1342 / 0.1620 / 0.1408 on folds 2-4 and the best all-feature models score 0.1221 / 0.1450 / 0.1207, a 9-14% gain, not 11-14%. Fold 1: LightGBM alone 0.1250 beats the 3-model blend 0.1288.
4. **The saved "corrected" blend is an equal-weight log mean**, so the plan's F3 line "equal weights were never tested" is wrong. The equal-weight log mean of CatBoost and LightGBM beats both members on folds 2-4 (0.1183, 0.1430, 0.1197).
5. **Horizon-bucket bias offsets lose to one global offset** in 6 of 6 fold/model cases (LightGBM 0.1367 vs 0.1329; CatBoost 0.1429 vs 0.1386). The bias sign is not stable: LightGBM fold 1 is +0.03 and folds 2-4 are -0.03 to -0.08. A leave-one-fold-out global offset helps LightGBM (0.1425 to 0.1329 on open rows, but fold 2 gets worse) and does not pass the 0.002 gate for CatBoost. Plan item 1.2 (one global constant per model) stands.
6. **Seasonal fold arithmetic.** A fold at origin 2014-06-19 scores 41,250 rows (11 days x 1,115 hubs + 31 days x 935 hubs), because the 180 gap hubs vanish on 2014-07-01. Report it with and without the gap hubs.
7. **Code facts.** 347 columns, but `feature_columns` returns 341 (+1 encoding); 24 Conversion/VolumePerSession columns are exact duplicates; the fold loader hardcodes 4 folds and 46,830 rows; all three fitters train only on open rows with volume > 0 (so "down-weight open zeros" does nothing until the mask admits them); LightGBM has `subsample` without `subsample_freq`.
8. **LightGBM 4.7.0 behaviour** (tested): `predict` ignores `init_score` (add log L manually); `init_score` switches off `boost_from_average`; `regression_l1` with `linear_tree` is fatal; a validation Dataset can carry its own `init_score` with early stopping.

## 3. Improvement plan changes (model track)

| Step | Change to the long-form plan |
|---|---|
| 0.0 (new, first hour) | Per-fold counts of rows, hubs and open-but-zero rows; RMSLE with and without them; rerun baseline with 2-3 seeds to measure noise. If seed spread of mean CV is >= 0.001 the 0.002 gate is too tight |
| 0.4 | `fit_transform` is the minimum. Better: drop the HubID encoding, or use an expanding history-only encoding (outside review; sklearn docs confirm the leak) |
| 0.5 | Split early stopping by label end date, purge overlap, refit on fixed rounds |
| 0.7 / gate | Five folds as planned (3 of 5 wins, >= 0.002 mean gain, open-zero check). Fold 4 is frozen as the end-term test, so development folds must exclude it for any work that feeds the end-term (section 6) |
| 0.8 | Score-point disagreement (0.08528 vs 0.06976) is one bit of evidence and may reflect a single-model defect: sanity check only, never a stop rule or a tuning target |
| 1.1 | Residual target `log1p(y) - log1p(L)`; L from data before the origin only, same for every row of the 42-day window; Tweedie via `init_score=log(L)` with log L added back at predict time; unit-test it |
| 1.2 | One global held-out offset per model (not per horizon bucket, not per hub) |
| 2.2 | Equal-weight log mean is the baseline blend; keep fitted weights only if they beat it under leave-one-fold-out |
| Screening | Screen changes with LightGBM alone, confirm winners on the full blend, log which model each row used |
| Added candidates | `linear_tree` member (numeric trend features, CPU), hurdle model for demand occurrence (outside review, UNVERIFIED gain), entity-embedding MLP (Guo and Berkhahn 2016), direct models for horizons 22-42, purged diagnostic fold, adversarial validation as a feature check |
| Deferred | Foundation models (one Chronos-2 pilot at most; no verified head-to-head vs tuned GBDT on this data), peer-hub residual features, regime experts, hierarchical reconciliation, uncertainty calibration (all outside suggestions, no local evidence) |
| Removed | Horizon-bucket offsets (refuted by saved OOF), zero-row-dropping to explain the gap (refuted) |

## 4. End-term design (rubric-driven)

- **Split:** one fixed time-ordered split from the training file only. Candidate: test = last 42 days (2015-05-09..2015-06-19), train = everything earlier. Rebuild every feature with the forecast origin at the split date; no lag may reach into the test window. The public labels of this dataset must never be used.
- **Models:** `x*` = LightGBM on `log1p` volume with hyperparameters tuned on a training-only part and frozen; ridge as the second model. Report test RMSE and R2 (and RMSLE).
- **Sensitivity:** `s_i = max_t [ f_i(x~_t)/F_sum(x~_t) + 1/|G_i,t| ]`, `|G|=n` without clusters. Use the SUM of weighted losses for F inside `s_i`, so `sum_i f_i/F = 1` per model and total sensitivity S lies in [2, 6] for five rough models and one group (unit test). Use a mean only in the ratio Plots 3-4. Compute `f_i` on every row of the training split; never score test rows with the rough models. Report S, the `s_i/S` histogram (log x), the effective sample size `(sum w)^2 / sum w^2` and the maximum weight.
- **Sampling:** with replacement, `p_i = s_i/S`, weight `1/(m p_i)`. Same weight convention for uniform (`n/m`). Merge duplicate draws exactly for ridge; for LightGBM compare merged and expanded draws. For ridge do not renormalise weights; use `alpha = n * lambda`.
- **Settings:** identical hyperparameters at every summary size. A scaled `min_data_in_leaf` is a labelled ablation on Plot 1 only.
- **Sizes and repeats:** at least 5 sizes (for example 1k, 3k, 10k, 30k, 100k rows), 5 repeats with new seeds each, median with min-max bars, uniform on every plot. The smallest size is below the hub count (1,115), so add hub-stratified sampling as an extra method. Columns are a separate experiment with one factor varied at a time (25/50/100/200 columns by importance plus correlation clustering; Gaussian or CountSketch only for ridge).
- **Plots:** Plot 3 train gap, Plot 4 test gap (denominator is the train loss, so it levels off at the generalisation gap: draw the `y*=x*` level as a reference and also show the `F_test(x*)` denominator), Plot 5 parameter difference (for trees use the vector of test-point predictions). Expectation for ridge only: Plot 5 slope about half of Plot 3's (measured on a toy fit: gap -0.90 uniform / -0.98 sensitivity, error -0.45 / -0.44). For trees no such relation is guaranteed. For trees call the method "heuristic sensitivity-motivated importance sampling", never a guarantee.
- **Timing:** timing runs get the server alone (fixed core set, fixed `n_jobs`, load average logged); build time includes the 5 rough models and scoring all training rows.
- **Zero rows:** whichever rule is chosen for open-but-zero rows applies identically to `x*`, the coreset and test scoring. Toy ridge result: the 54 rows take about 4.9% of the sensitivity mass (about 1000x their uniform share).
- **k-means:** optional ablation only (cluster groups in the `1/|G|` term, feature-column clustering, hub clusters). Plain k-means++ with Lloyd is enough (1,115 x 50 clusters in 0.12 s). Meta FAIR e-means (reported November 2026, secondary sources only) is not needed.

## 5. Step-by-step flow

1. Control rerun finishes. Compare rebuilt origin hashes with the kernel manifest and diff the control submission with the committed ones. Record model paths and SHA-256 in `STATUS.md`.
2. Step 0.0 checks, then Phase 0 (0.1-0.9) on a branch; baseline on all folds.
3. End-term critical path (needs Phase 0.4 done first): split, `x*`, uniform, rough models, sensitivities, coreset, column experiment, Plots 1-5, speed-up table.
4. Adversarial review of every number against the CSVs.
5. Slides built only from the CSVs.
6. Phase 1 and later model improvements only after the end-term is submitted, each through the gate.

## 6. Decisions needed from the team

1. Test window: last 42 days (equals CV fold 4, disclose) or a different split that does not reuse a viewed fold?
2. Closed rows in train and test sets (they inflate R2): include or exclude, and state it on slide 1.
3. Meaning of "column sketch" in the rubric (feature columns vs sketching rows): confirm against the instructor's original text.
4. Whether the forecasting folder stays public. The course rules recorded in `SUMMARY.md` forbid public sharing of the competition work.

## 7. Outside reviews received (24 Nov) and how they were used

Four pasted reviews and two files were read; two files were checked by independent reader agents against the data. Adopted: the collision finding, the sum-versus-mean sensitivity fix, the `init_score` bug, hub-stratified sampling, noise-floor and per-fold checks, separation of column and row experiments. Rejected after testing: horizon-bucket offsets, "open-zero rows explain the gap", renormalising ridge weights. Held as unverified leads: hurdle models, peer-hub residuals, regime experts, hierarchical reconciliation, adaptation to conditional shift, coresets built from GBDT leaves, and several arXiv IDs given only by the reviews (including one that contradicts the plan's Chronos-2 ID). None of these citations has been opened by us yet; open each before citing.

## 8. All-inclusive gated plan (v3, 2026-11-24 evening)

Sources merged: the repo's long-form plan and Max-plan files, the two laptop copies, the four pasted outside reviews, two audit documents (checked by reader agents), the research agents' reports and our own measurements. Every item has a test route. "Harness" = `improve/exp_runner.py` (LightGBM screening, folds 2-4, fixed settings, one row per run in `experiments.csv`).

### 8.1 Gate rules (decide mechanically, never by the public score)
- **Champion** = the current best variant. Start: `base` (Phase-0 fixed LightGBM: cross-fitted HubID encoding, early stopping on a chronological holdout with purged origins).
- A variant **passes** only if all hold against the champion, on the same folds and seed: (1) mean RMSLE over folds improves by at least 0.002; (2) it wins on at least 2 of 3 development folds (3 of 5 once five folds exist); (3) the gain survives `rmsle_excl_open0`; (4) the gain is larger than twice the measured seed-to-seed spread of the champion; (5) per-hub and per-horizon error do not get materially worse (report, do not hide).
- The best passing variant becomes the new champion and the next variants are re-run against it (greedy forward selection). Variants that fail are logged, not deleted.
- Confirm each new champion on the full 3-model blend and on the season-matched fold before it is accepted. The final submission is chosen by CV, never by the public score.

### 8.2 Stages
| Stage | Content | Where | Done when |
|---|---|---|---|
| G0 trust | Noise floor (`base`, 2 seeds), `no_te` vs `base`, per-fold open-zero counts, season-matched fold 2014-06-19 built, fold 1 added, unit tests (helpers pass 6/6) | server B | noise floor and TE verdict logged |
| G1 level and bias (harness variants) | `level`, `level_off`, `offset`, `recency365/180`, `tweedie_init`, `colsample03` | server B (2 jobs) | each row in `experiments.csv`, champion chosen |
| G2 structure (needs small code) | `linear_tree` member; hurdle (classifier x positive regressor); month-ahead model for horizons 22-42 using only long-window features; per-format models; summer-2014 incomplete origins (needs a change in the origin builder) | server B / server A | gate verdicts |
| G3 features (needs feature code) | per-hub ridge trend (91/365 days), recent-window weekday x promo medians, closure and promo streak counters, region proxy, drop the 24 duplicate columns, permutation-importance pruning | server B | gate verdicts |
| G4 members and blend | CatBoost on the champion recipe, seeds, equal-weight log mean vs constrained weights (leave-one-fold-out), entity-embedding MLP (CPU/GPU), one Chronos-2 pilot, TabPFN-TS only after detrending | both | blend passes the gate on all folds |
| G5 final | retrain champion on all origins with fixed rounds, SAVE models and SHA-256, sanity checks (closed rows 0, no NaN, Id order, test-prediction distribution), tag and record in `STATUS.md` | server | models saved and hashed |

### 8.3 End-term track (runs in parallel, does not wait for G2-G4)
1. Decide the split (section 6, item 1). 2. Rebuild features at the split date. 3. `x*` LightGBM and ridge, hyperparameters tuned on a training-only part. 4. Uniform and hub-stratified sampling, leverage and sensitivity sampling, rough models, coreset, column experiment. 5. Plots 1-5, speed-up table, timings on an otherwise idle server. 6. Review agent checks every number, then slides.

### 8.4 Ideas from outside reviews: status and test route
| Idea | Source | Status | Test route |
|---|---|---|---|
| Level residual target, global offset | plan, reviews | in harness | `level`, `level_off`, `offset` |
| Horizon-bucket offsets | reviews | refuted by saved OOF | none |
| Open-zero rows explain the gap | reviews | refuted | none |
| Hurdle model (occurrence x size) | reviews 1, 3 | G2 | classifier on open rows, product with positive regressor; compare with `rmsle_excl_open0` |
| Month-ahead models for horizons 22-42 | review 1 (winner write-up not verified) | G2 | train on horizons 22-42 with only long-window features |
| Two-stage sessions x basket size | review 1 | G3 | out-of-fold session model; high leakage risk, gate strictly |
| Season-filtered member (May-Sep) | review 1, plan 1b.2 | G2 | needs incomplete origins |
| Closure and promo streak counters | reviews, plan 1b.7 | G3 | add columns, rerun gate |
| Entity-embedding MLP | reviews, plan 3.3 | G4 | CPU first, blend by out-of-fold weights |
| Foundation models | reviews disagree | G4 pilot only | one Chronos-2 pilot; no verified head-to-head on this data |
| Peer-hub residual features | review 3 | waiting list | residuals available only up to the origin, so it acts as a level feature; test after G3 |
| Regime experts, mixture of experts | review 3 | waiting list | regime indicator as a feature first |
| Hierarchical reconciliation | review 3 | waiting list | sums are on the raw scale while the loss is on log scale; test small |
| Conditional-shift adaptation, uncertainty calibration | review 3 | waiting list | low priority |
| Purged diagnostic fold, adversarial validation | review 4 | G0 / G3 | extra fold; feature screening |
| Coresets from GBDT leaves (PVLDB 2024) | review 4 | end-term extra | verify source first |
| Multi-seed bagging | review 4 | G4 | seeds in the blend |

## 9. Run log, 2026-11-24 evening
- Data rebuild fidelity (VERIFIED): both servers rebuilt 37 origins and 4 folds from the raw files; the two servers agree on all 37 file hashes. Against the original run's output the headers (347 columns) and row counts are identical and values differ only in the last digit (about 1e-16 relative, library-build rounding in the standard-deviation features), so the rebuilt data is numerically equal but not bit-identical.
- Harness smoke test (LightGBM, fold 2, 50-round cap, 8 threads, server under load): RMSLE 0.1301 (0.1237 without the one open-zero row). Loading a fold took most of the 809 s, so the harness now loads each fold once and runs several variants on it.
- Two sweeps running on the second server: A = `base,no_te,level,level_off,offset` (seed 20260807), B = `base,recency365,colsample03,tweedie_init` (seed 20260808). The two `base` rows give the noise floor.
- Control rerun of the original 3-model pipeline: fitting CatBoost, then LightGBM, then Tweedie; models and `submission_control.csv` appear in its output folder when done.

## 10. Results, 2026-11-25 (VPN restored)

### Control rerun (current best, server A) — DONE
- Weights recovered by rebuilding: `catboost.cbm`, `lightgbm.txt`, `lightgbm_tweedie.txt` plus the three `test_pred_*.npy`, all in `~/Kush/shadowhub_endterm/out_control/` on server A.
- SHA-256: catboost `03a248b2…ce06189`, lightgbm `572a90e2…06487a0`, tweedie `e50e4245…bb9ae1`; submission `443aa57e…fd0b8ba86` (full hashes in `control_manifest.json`).
- Compared with the committed `submission_ensemble.csv`: 46,830 rows, log-space RMS difference 3e-5, max 7e-4. The old pipeline is reproduced. The other committed `submission.csv` differs by RMS 0.063, so it is a different artefact.
- Best iterations: CatBoost 1499/1500, LightGBM 3000/3000, Tweedie 3000/3000 (no early stopping triggered). Fit times: 1.5 ks, 17.8 ks, 13.3 ks on a loaded server.

### Screening rounds 1–2 (dev folds 2,3,4; mean RMSLE all rows; lower is better)
Base noise: two seeds of `base` give 0.1221 and 0.1233 (spread 0.0012). Gate: gain ≥ 0.0024 vs the current champion.

| Variant | mean | fold2 | fold3 | fold4 | note |
|---|---|---|---|---|---|
| base | 0.1221 / 0.1233 | 0.126 | 0.132 / 0.137 | 0.108 | two seeds |
| recency365 (fixed helper) | 0.1208 | | | | within noise |
| feature fraction 0.5 | 0.1174 | | | | |
| feature fraction 0.3 | 0.1125 | 0.1205 | 0.1134 | 0.1036 | |
| **cs02** (fraction 0.2 + bagging) | **0.1127 / 0.1128** | 0.1215 | 0.1129 | 0.1039 | **champion, −0.0094 vs base, beyond noise** |
| cs02 without target encoding | 0.1112 | 0.1201 | 0.1122 | 0.1014 | −0.0015: below gate, stack candidate |
| cs02 lambda 10 | 0.1116 | 0.1212 | 0.1118 | 0.1019 | −0.0011: below gate |
| cs02 127 leaves | 0.1117 | 0.1183 | 0.1120 | 0.1048 | −0.0010: below gate |
| cs02 recency365 | 0.1145 / 0.1119 | | | | seeds disagree: noise |
| cs015 / cs01 / nobag / leaves31 / mcs200 / extra | 0.1129 / 0.1139 / 0.1126 / 0.1146 / 0.1142 / 0.1174 | | | | no gain |
| level target, level + offset, Tweedie with init score | 0.192 / 0.198 / 0.138 | | | | clearly worse; dropped |
| offset only | 0.1247 | | | | worse |

Reading: feature subsampling is the only change that passes the gate. Nothing in round 2 passes it alone. Three small gains (no target encoding, lambda 10, 127 leaves) are each about the size of the seed noise, so round 3 tests them stacked and on a third seed.

### Round 3 (launched 25 Nov ~12:17 on server B)
- F: stacked variants `cs02_no_te_l10`, `cs02_no_te_l10_127`, `cs02_l10_127` (seed 20260807, folds 2–4).
- G: `cs02_no_te`, `cs02_lambda10`, `cs02_leaves127` on seed 20260809 (noise check).
- H: `base`, `cs02`, `cs02_no_te` on fold 1 (a fold the champion was not picked on).
- Pass rule: a stack becomes champion only if it beats cs02 by ≥ 0.0024 on ≥ 2 of 3 folds on both seeds. Otherwise cs02 stays the recipe carried into CatBoost and the blend check.
