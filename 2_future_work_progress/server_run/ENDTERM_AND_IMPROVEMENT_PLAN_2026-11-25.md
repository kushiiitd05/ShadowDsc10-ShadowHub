# End-term and improvement plan (started 2026-11-24, updated 2026-11-25; the final plan v4 is `PLAN_V4_FINAL_2026-11-25.md`)

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

## 11. Calendar (due date confirmed 25 Nov: submission 8 Dec, presentations 9–10 Dec)

| Dates | Work | Server use |
|---|---|---|
| 25–27 Nov | Finish round 3, pick recipe; CatBoost + blend check on it; end-term fixed split and x* model (LightGBM + ridge) built | B: screening; A: end-term build |
| 28 Nov–1 Dec | End-term experiments: ≥5 sizes × 5 repeats for uniform, hub-stratified, coreset (sensitivity), column sketch; Plots 1–5 data | A and B in parallel (no timing yet) |
| 1–2 Dec | **Exclusive timing runs** (nothing else on the server) for the speed-up table | one idle server |
| 3 Dec | **Freeze**: final model recipe, saved weights with SHA-256; no new variants after this | — |
| 3–5 Dec | Plots, tables, 6–9 slides, reviewer-agent pass, check for banned names/numbers | none |
| 6–7 Dec | Rehearse to 7 min, fix, buffer | none |
| 8 Dec | Submit | — |
| 9–10 Dec | Present | — |

Rules: rubric track wins any clash with improvement work. Improvement items not finished by 3 Dec go into the conclusion as future work, not into the plots.

## 12. Round 3 verdict (25 Nov, 13:00)

Mean RMSLE over folds 2–4; same-seed reference `cs02` in the first column.

| Variant | seed 20260807 | seed 20260809 | fold 1 (seed 07) |
|---|---|---|---|
| `cs02` | 0.1127 | 0.1136 | 0.1379 (base 0.1586) |
| `cs02_no_te` | 0.1112 | 0.1156 | 0.1349 |
| `cs02_lambda10` | 0.1116 | 0.1137 | |
| `cs02_leaves127` | 0.1117 | 0.1126 | |
| `cs02_l10_127` | 0.1108 | 0.1134 | 0.1335 |
| `cs02_no_te_l10_127` | 0.1109 | 0.1140 | 0.1294 |
| `cs02_no_te_l10` | 0.1115 | | |

- Noise of the champion across seeds: 0.0009 on the 3-fold mean; a single fold moves by up to 0.008 (fold 3 of `cs02_no_te`: 0.1122 vs 0.1208).
- No variant passes the gate (0.0024). The best stack (`cs02_l10_127`) gains 0.0019 on seed 07 but 0.0002 on seed 09, and is worse on fold 2 of seed 09, so it fails the "better on every fold and seed" challenger test. Dropping target encoding helps on seed 07 and fold 1 and hurts on seed 09.
- **Decision: `cs02` stays champion; no challenger.** The round-2 single-seed gains were mostly noise. Fold 1 (a different season) favours the stacks, so `cs02_l10_127` and `cs02_no_te_l10_127` are kept as optional extras for G4b multi-seed bagging, not as the recipe.
- Baseline-ensemble jobs B1 (`base`, production Tweedie, production CatBoost) and B2 (the same on `cs02`) are running on folds 1–4; blends are computed offline by `improve/blend.py`.

## 13. Baseline ensemble versus champion ensemble (25 Nov, 13:40) — B1 and B2 complete

Same folds (1–4), same seed 20260807, same early-stopping holdout rule; blend = log-space mean of the three members (CatBoost production settings, LightGBM, production-style Tweedie with raw target and log link). LightGBM screening settings (lr 0.05, ≤1500 rounds) are used for both rows, so the comparison is like for like; the original production run used lr 0.025 and 3000 rounds.

| Recipe | Member | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
|---|---|---|---|---|---|---|
| old (`base`) | LightGBM | 0.1586 | 0.1259 | 0.1322 | 0.1081 | 0.1312 |
| old | Tweedie | 0.1514 | 0.1258 | 0.1359 | 0.1106 | 0.1309 |
| old | CatBoost | 0.1445 | 0.1335 | 0.1305 | 0.1050 | 0.1284 |
| **old, 3-model blend** | | 0.1412 | 0.1261 | 0.1286 | 0.1041 | **0.1250** |
| new (`cs02`) | LightGBM | 0.1377 | 0.1215 | 0.1129 | 0.1039 | 0.1190 |
| new | Tweedie | 0.1414 | 0.1248 | 0.1095 | 0.1042 | 0.1200 |
| new | CatBoost | 0.1378 | 0.1241 | 0.1225 | 0.1010 | 0.1213 |
| **new, 3-model blend** | | 0.1356 | 0.1215 | 0.1129 | 0.1010 | **0.1178** |

- **Old best ensemble on the corrected folds: 0.1250. New champion ensemble: 0.1178. Gain −0.0072, better on all four folds** (−0.0056, −0.0046, −0.0157, −0.0031); above the 0.0024 gate and about 7× the seed noise.
- The recipe gain is larger for the single LightGBM (−0.0122) than for the blend (−0.0072): blending already removed part of the variance that feature subsampling removes.
- On the champion recipe the blend improves on LightGBM alone by only 0.0012 (it ties on folds 2 and 3). Within the noise: keep it only if the end-term time budget allows; the LightGBM member alone is the fallback.
- Members are not yet tuned on the champion recipe (CatBoost uses `rsm 0.2`, Tweedie uses the same feature fraction); that is stage G4a.

## 14. Experiment modules built (25 Nov, afternoon) and the server run queue

Three modules were written by helper agents, tested locally on synthetic data (all tests pass), and pushed under `improve/`. None has run on real data yet. Each reuses `exp_runner.prepare_fold` and writes `experiments.csv` plus `oof_<variant>_s<seed>.csv.gz` in the harness schema, so `blend.py` and the gate rule apply unchanged.

| Stage | Module | Variants | Notes |
|---|---|---|---|
| G2 structure | `g2_horizon/g2_runner.py` | `cs02` (control through the same code path), `split_horizon` (models for horizons 1–21 and 22–42), `long_window_only` (the 22–42 model drops 32 short-window columns, keeps 309 of 341) | 56 tests; `horizon_as_weight` skipped on purpose: training and validation both contain every hub × every horizon, so the weights would be constant. Expected gain small because trees can already split on Horizon |
| G3 features | `g3_features/g3_runner.py` | `feat_trend`, `feat_weekday_promo`, `feat_streaks`, `feat_all`, `dedupe_only`, `dedupe_plus_all` | 45 tests; 23 new `g3_*` columns; leakage test passes on real raw history (origin 2014-10-02: identical output after scrambling later data). The 24 duplicate columns are the `VolumePerSession_*` set (structural twins of `Conversion_*`); a data-level check finds 29 more (mostly `Sessions_*Count` equal to `Orders_*`), dropped only with `--dedupe-scope all`. Promo flag is global (same for every hub on a date), so promo streak columns act as calendar features. No region column exists: `HubFormat` and `AssortmentTier` stand in |
| G4b MLP | `g4_mlp/g4_runner.py` | `mlp` (entity embeddings, 512-256-128, OneCycle, bf16) | 11 tests; needs torch on the server; early stopping on the one-origin holdout, so noisy |

Run order once the servers are reachable (each job in its own output folder, `nohup nice`):
1. Upload `improve/` (tar over ssh); smoke each module on fold 4 with a tiny round count.
2. Server B: `g2_runner.py --variants cs02,split_horizon,long_window_only --folds 1,2,3,4`, then `g3_runner.py --variants feat_all,dedupe_only,dedupe_plus_all --folds 1,2,3,4`, then the single-group features that survive.
3. Server A (GPU): `g4_runner.py --folds 1,2,3,4 --epochs 40 --device cuda`, then `blend.py` on `cs02,cs02_tweedie,cs02_cat,mlp`.
4. A candidate that passes the gate on seed 20260807 is repeated on seeds 08 and 09 before it becomes champion.
Running now (independent of the VPN): server A `et_full` (end-term sampling, 5 repetitions × 7 sizes, LightGBM and ridge, rows and columns); server B seed-08 and seed-09 champion-recipe blends and a lr-0.025/3000-round run for the final retrain settings.

## 15. Leftover-items ledger (25 Nov, 14:00): what remains untested from every plan version

Offline checks run on the saved fold predictions (pooled over folds 1–4, leave-one-fold-out):

| Item | Old recipe | New recipe (`cs02`) | Verdict |
|---|---|---|---|
| Equal-weight log-mean blend (baseline) | 0.1257 | 0.1184 | keep |
| 2.2 non-negative blend weights (NNLS), held-out fold | 0.1251 | 0.1192 | no gain on the new recipe (worse): **keep equal weights** |
| 1b.4 calibration `log1p(y) ≈ a + b·ŷ`, held-out fold | 0.1285 | 0.1207 | worse on both: **drop** |

(Pooled RMSE differs slightly from the mean of fold scores in section 13 because folds have different row counts.)

| Plan item | Origin | Status |
|---|---|---|
| 0.1, 0.4, 0.5, 0.6 origins rebuilt, cross-fitted encoding, holdout early stopping, extra metrics | long plan | done in the harness |
| 0.2 schema assertion in `model_pipeline.py` | long plan | left (low value; harness bypasses it) |
| 0.3 season-matched fold 2014-06-19 | long plan | **left** |
| 0.8 CV ranks CatBoost-alone below the 3-model blend like the earlier public points did | long plan | consistent (0.1284 vs 0.1250); closed |
| 1.1 level target, 1.2 offset, 1.3 recency | long plan | tested, refuted/no gain |
| 1.4 / 1b.1 per-hub ridge trend; 1b.3 weekday × promo medians; 1b.7 closure counters; 1b.8 region proxy (format/tier) | long plan | **running** (G3) |
| 1b.2 summer-2014 incomplete origins (and the season-filtered member) | long plan | **left**: needs a change to the origin builder |
| 1b.5 holiday / school-closure counts this, last, next week | long plan | **left** (feature code) |
| 1b.6 hub ratio features (promo lift, Saturday ratio, open share) | long plan | **left** (feature code) |
| 1b.8 region clusters from identical holiday/closure dates | long plan | **left** |
| 1b.9 feature subsampling | long plan | done: champion |
| 1b.10 per-HubFormat models blended with the global one | long plan | **left** |
| 2.1 three seeds per backend | long plan | **running** (seeds 08, 09) |
| 2.2 NNLS weights | long plan | done: refuted |
| 2.4 Optuna search (≤ 40 trials per backend, folds 3–4 then check 1–2) | long plan | **left** |
| 3.1 Chronos-2 zero-shot, 3.1b TabPFN-TS | long plan | **left** (3.1 pilot only; 3.1b only if time) |
| 3.2 TiDE / NHITS | long plan | **left** (cut first) |
| 3.3 entity-embedding MLP | long plan | code ready, torch installing |
| 4 error analysis by segment; permutation-importance pruning | long plan | **left** (offline from saved predictions, cheap) |
| 5 final retrain, saved weights, sanity checks | long plan | **left**, due 3 Dec |
| Hurdle model | v3 | dropped: ≤ 1 open-zero row per dev fold |
| Horizon split / long-window-only | v3 | **running** (G2) |
| Fixed rounds versus early stopping | new | **running** |
| Month-ahead model, per-format models, two-stage sessions | v3 | **left** (after G2 result) |
| Adversarial validation, purged diagnostic fold | v3 | **left** (cheap) |
| Peer-hub residuals, regime experts, reconciliation, shift adaptation | v3 | waiting list |
| Coresets from GBDT leaves | v3 | end-term extra |

## 16. Results at 14:00 (25 Nov): G2, fixed rounds, seed bagging; winner-sweep additions

Mean RMSLE over folds 1–4, seed 20260807 unless stated.

| Test | Result | Verdict |
|---|---|---|
| G2 `split_horizon` (separate models for horizons 1–21 and 22–42) | 0.1190 (folds 0.1370 / 0.1210 / 0.1107 / 0.1072) vs `cs02` 0.1190 | no gain: dropped |
| G2 `long_window_only` (22–42 model without the 32 short-window columns) | 0.1189 | no gain: dropped |
| Fixed rounds at lr 0.05, no early stopping | 300 rounds 0.1190; 600 rounds 0.1195; 1000 rounds 0.1198 | **early stopping is not needed**: fixed 300 equals early-stopped. The final retrain can use a fixed round count |
| `cs02` at lr 0.025 / 3000 (early stopped) | 0.1170 | slightly better than lr 0.05 (0.1190) |
| `cs02` on 3 seeds (07, 08, 09), 4-fold mean | 0.1190, 0.1186, 0.1191 | seed spread of the 4-fold mean is only 0.0005 |
| 3-model blend per seed (08, 09) | 0.1184, 0.1180 (seed 07: 0.1178) | as before |
| **LightGBM `cs02` averaged over 3 seeds** | **0.1173** (folds 0.1340 / 0.1194 / 0.1124 / 0.1035) | −0.0017 vs a single seed |
| 3-model blend averaged over 3 seeds | 0.1175 (folds 0.1339 / 0.1204 / 0.1143 / 0.1016) | **no better than seed-averaged LightGBM alone**: once seeds are averaged, CatBoost and Tweedie add nothing |

Working conclusion: the final model is a seed-averaged LightGBM `cs02`, with a fixed round count, tested at lr 0.025 (fixed 600 and 1000 rounds running). The 3-model blend stays as a fallback only.

**Winner sweep** (`research/winner_sweep_2026-11-25.md`; the agent opened M5 1st/2nd/4th/5th code, Web Traffic 1st, Walmart 1st, Favorita 5th/12th and four arXiv papers; Favorita 1st, M5 3rd and the Ecuador competition were not verifiable; no source gives an RMSLE gain, so its ranking is judgement). New items added to the ledger:

| # | Item | Plan |
|---|---|---|
| 1 | Fixed rounds / checkpoint averaging / seed bagging instead of early stopping | **done above** |
| 2 | Treat the 184-day gap as missing: rolling and lag-364 features from observed days only, observed-day counts, fallback to a prior below a coverage minimum | feature code (round G3b) |
| 3 | Year-ago window features (weekday ±1 week around lag 364; year-ago 42-day window mean, zero-day and promo counts; ratio to the year-ago pre-origin level) | feature code (G3b) |
| 4 | Peer aggregates at the origin: format/tier means (in G3) plus the 5 most-correlated hubs | feature code (G3b) |
| 5 | Keep-bar rule: same 3 seeds as the baseline, accept only if the gain beats the seed spread | already our gate; spread is now measured (0.0005 on the 4-fold mean) |
| 6 | Low-rank SVD denoising of the hub × day matrix (rank 12–15), refit per origin | candidate, medium cost |
| 7 | Equal-weight blend with one non-tree member | MLP run pending (torch build fixed: the first install was CUDA 13 on a CUDA 12.9 driver; reinstalling the cu128 build) |
| 8 | Zero-day and closure counts and promo / non-promo means over several windows | in G3 streaks; extend in G3b |
| 9 | Training origins every 7 days instead of 14 | candidate; needs origin rebuild |
| 10 | Recursive plus direct models | dropped (low plausibility at 42 days, high cost) |

## 17. G3 verdict (25 Nov, 15:00): none of the new features passes the gate

Mean RMSLE over folds 1–4, seed 20260807, against `cs02` 0.1190 (folds 0.1377 / 0.1215 / 0.1129 / 0.1039). Deltas per fold in brackets.

| Variant | Mean | Δ folds 1 / 2 / 3 / 4 | Verdict |
|---|---|---|---|
| `dedupe_only` (drop 24 duplicate columns) | 0.1184 | −0.0010 / 0.0000 / −0.0023 / +0.0010 | within seed spread (0.0005 on the mean); harmless, slightly faster fits |
| `feat_streaks` | 0.1186 | +0.0006 / −0.0012 / −0.0005 / −0.0007 | within noise |
| `feat_trend` | 0.1207 | +0.0007 / 0.0000 / +0.0039 / +0.0022 | worse |
| `feat_all` | 0.1321 | **+0.0531** / −0.0001 / +0.0013 / −0.0020 | fold 1 breaks |
| `feat_weekday_promo` | 0.1304 | **+0.0447** / +0.0001 / +0.0030 / −0.0020 | fold 1 breaks |
| `dedupe_plus_all` | 0.1269 | +0.0361 / +0.0019 / −0.0011 / −0.0056 | fold 1 breaks |

- **No new feature passes the 0.0024 gate.** On folds 2–4 the groups move the score by at most ±0.002.
- **Fold 1 breaks with the weekday × promo group:** the forecast origin 2015-01-02 sits right after the 184-day gap (to 2014-12-31) of 180 hubs, so their last 8 weeks have no observations, the medians are missing or stale and the training origins do not look like this. This is the failure the gap-aware group (G3b, next round) is meant to fix, so G3b is now more important than it looked.
- `dedupe_only` is neutral; it is kept as an option to shorten fit times, not as an improvement.
- Stage G3 is closed except the gap-aware / year-ago / peer round (G3b), which is being written.

## 18. Per-group models (1b.10), 25 Nov 15:30

One `cs02` LightGBM per group, stitched; groups with fewer than 20,000 training rows share one pooled model. Folds 1–4, seed 20260807.

| Variant | Mean | Folds 1 / 2 / 3 / 4 |
|---|---|---|
| `cs02` (global) | 0.1190 | 0.1377 / 0.1215 / 0.1129 / 0.1039 |
| per HubFormat | 0.1198 | 0.1397 / 0.1214 / 0.1138 / 0.1042 |
| **per AssortmentTier** | **0.1178** | **0.1316** / 0.1197 / 0.1128 / 0.1068 |

Equal-weight log-mean blends:

| Blend | Mean | Folds 1 / 2 / 3 / 4 |
|---|---|---|
| `cs02` s07 only | 0.1190 | 0.1377 / 0.1215 / 0.1129 / 0.1039 |
| `cs02` s07 + s08 (2-seed average, size-matched control) | 0.1176 | 0.1350 / 0.1199 / 0.1125 / 0.1030 |
| `cs02` s07 + per-tier | 0.1168 | 0.1327 / 0.1190 / 0.1114 / 0.1042 |
| `cs02` 3-seed average | 0.1173 | 0.1340 / 0.1194 / 0.1124 / 0.1035 |
| `cs02` 3-seed + per-tier | **0.1165** | 0.1322 / 0.1185 / 0.1117 / 0.1037 |
| `cs02` 3-seed + per-tier + per-format | 0.1165 | 0.1327 / 0.1185 / 0.1116 / 0.1032 |

- The per-tier model alone is −0.0012 vs `cs02`, mostly from fold 1 (−0.0061), the fold right after the data gap; per-format is worse.
- Mixing a per-tier model with the global one beats the size-matched seed average by about 0.0008 (0.1168 vs 0.1176; 0.1165 vs 0.1173). That is below the 0.0024 gate and above the seed spread (0.0005): **keep as an optional member**, decide after the seed-08/09 per-tier runs (running).
- Per-format adds nothing on top of per-tier.

## 19. Per-tier on three seeds; lr 0.025 fixed rounds (25 Nov 14:30)

| Seed | `cs02` | per-tier | `cs02` + per-tier (equal log mean) |
|---|---|---|---|
| 20260807 | 0.1190 | 0.1178 | 0.1168 (−0.0022) |
| 20260808 | 0.1186 | 0.1174 | 0.1167 (−0.0019) |
| 20260809 | 0.1191 | 0.1198 | 0.1179 (−0.0012) |
| 3-seed average of each, then blended | 0.1173 | 0.1170 | **0.1164** (−0.0009 vs the size-matched 3-seed average) |

- The pairing gain has the same sign on all three seeds and comes mostly from fold 1 (about −0.005); it stays below the 0.0024 gate against a size-matched control, so per-tier is an **optional member**, adopted only if the final retrain budget allows.
- Fixed rounds at lr 0.025: 600 rounds 0.1176, 1000 rounds 0.1174 (early-stopped at lr 0.025: 0.1170; fixed 300 at lr 0.05: 0.1190). **Final-retrain setting: lr 0.025, 1000 fixed rounds.**

## 20. G3b results and the gap-hub finding (25 Nov 15:00)

Seed 20260807, RMSLE, `cs02` reference 0.1190 (folds 0.1377 / 0.1215 / 0.1129 / 0.1039).

| Variant | Mean, all hubs | Mean, 935 non-gap hubs | Fold 3 / fold 4 (all hubs) |
|---|---|---|---|
| `cs02` | 0.1190 | 0.1196 | 0.1129 / 0.1039 |
| `g3b_gap` | 0.1220 | 0.1232 | 0.1235 / 0.1066 |
| `g3b_yearago` | 0.1199 | 0.1217 | 0.1167 / 0.1061 |
| `g3b_peer` | 0.1191 | 0.1202 | 0.1177 / 0.1050 |
| `g3b_all` | 0.1193 | 0.1208 | 0.1123 / 0.1051 |
| `feat_all` (G3) | 0.1321 | 0.1182 | 0.1155 / 0.1034 on non-gap hubs |
| `g3b_all_plus_g3` | 0.1251 | **0.1175** | **0.1100 / 0.0994** |

- **The G3 features break the 180 gap hubs on fold 1** (RMSLE 0.29–0.37 on those hubs vs 0.14 for `cs02`): their last 8–13 weeks before 2015-01-02 are empty. On the 935 non-gap hubs the same features help slightly (`feat_all` −0.0014, `g3b_all_plus_g3` −0.0021, fold 4 −0.0046).
- **Why this matters for the real test:** the real forecast window starts 2015-06-20, almost six months after the gap ended, so the 8-week and 91-day windows of every hub are clean. Folds 3 and 4 (origins 2015-03-27 and 2015-05-08) look like the test; folds 1 and 2 are contaminated by the gap.
- **Caution, a post-hoc choice:** deciding that folds 3–4 matter most was made after seeing the table. It is only accepted if (a) the same sign holds on seeds 08 and 09 (running), (b) it passes on the season-matched fold 2014-06-19, and (c) fold 1 is reported honestly with gap hubs handled (fallback to the pre-gap value or a prior) rather than hidden.
- No G3b variant alone passes. The candidate is `g3b_all_plus_g3` (all new groups together).

## 21. Error analysis of the champion (Phase 4, 25 Nov 15:15) and fold 5

Saved `cs02` predictions on folds 1–4 (seed 20260807), open rows.

| Segment | Finding |
|---|---|
| HubFormat 1 / 2 / 3 / 4 | RMSLE 0.121 / 0.129 / 0.133 / 0.111; format 1 carries 55% of squared error because it has most rows |
| AssortmentTier 1 / 2 / 3 | 0.125 / 0.110 / 0.113 |
| Gap hubs vs others | 0.117 vs 0.120: the champion is not worse on the 180 gap hubs |
| Weekday | Thursday 0.142 and Monday 0.138 worst; Sunday has 0.0318 only because most Sunday rows are closed |
| Horizon week 1 / 2 / 3 / 4 / 5 / 6 | 0.154 / 0.111 / 0.110 / 0.102 / 0.116 / 0.119: **week 1 is the worst, not the best** |
| Concentration | the 20 worst hubs carry 16% of squared error; the worst 5% of hubs 23% |

Why week 1 is worst: the error is **concentrated on a few event days**, not on a horizon. The 10 worst days (6% of the 168) carry 28% of the squared error. Days where a regional holiday covers more than 5% of hubs average RMSLE 0.267 vs 0.133 for other days. Worst days: 2015-05-01 (holiday + promo, 0.416), 2015-03-15 and several other Sundays (mean volume only about 200 vs about 5,000 on weekdays, so a few hubs are open), 2015-02-16 (promo + school closure), Easter 2015-04-03..06 (movable; lag-364 never aligns), 2015-05-14 and 2015-05-25 (holidays), 2015-05-15 (school closure 38%).

Consequence: the next feature round (G3c, being written) targets event days: hub-specific history on previous holidays of the same type (so Easter 2015 sees Easter 2014), holiday and school-closure counts and distances, an open-Sunday profile and a hub promo ratio. These are plan items 1b.5, 1b.6 and 1b.7, now justified by data rather than by the Rossmann write-ups.

Fold 5 (season-matched, origin 2014-06-19, validation 2014-06-20..07-31, 41,250 rows because 180 hubs have no rows after 2014-06-30): built on server B with the pipeline's complete-grid assertion switched off by an environment flag in the server copy only. Runs for `base`, `cs02`, `cs02_tweedie`, per-tier and the G3 candidate (3 seeds each) are in progress; they serve as the independent check on the summer season.

## 22. End-term run 1 complete (25 Nov 15:25) — results in `endterm/results_run1/`

Setup: fixed time split (train = origins up to 2015-05-08 minus the holdout, 1,133,640 open positive-volume rows; test = fold 4, 46,830 rows); model x* = LightGBM `cs02` recipe with 1184 rounds fixed from the training-only holdout, second model ridge (λ = 0.1 chosen on the same holdout); loss = squared error on log1p; sizes 2k–200k rows; 5 repetitions; medians with min–max. Timings are from a shared server (load 10–30): they are for development only; the speed-up table must come from the exclusive run on 1–2 Dec.

| Quantity | LightGBM | Ridge |
|---|---|---|
| Full model test RMSLE / R² (log scale) | 0.1022 / 0.9992 | 0.1592 / 0.9980 |
| Full training time (shared server) | 60 s | 4.3 s |
| Total sensitivity S (range 2–6 expected) | 2.47 | 2.12 |
| Largest roughness α_t = F(x̃)/opt | 2.45 | 1.01 |

LightGBM rows, test RMSLE (median) uniform → coreset: 5k 0.1218 → **0.1195**; 10k 0.1187 → **0.1170**; 25k 0.1133 → **0.1108**; 50k 0.1102 → **0.1070**; 100k 0.1099 → **0.1058**; ties at 2k (0.1297 / 0.1318) and 200k (0.1055 / 0.1045). Hub-stratified is no better than uniform. Coreset build time about 10 s (five rough models + scoring all rows); at 50k rows the coreset reaches 0.1070 vs full 0.1022 (+0.0048) in about 21 s against 60 s for the full fit, a 2.9× speed-up; uniform at 50k is 5.2× faster but 0.0080 above full.
Ridge: the sampling method barely matters (0.159–0.166 at every size); the coreset helps the training-set gap at small sizes (m = 2000: 0.030 vs 0.084) but its 18.6 s build cost exceeds the 4.3 s full fit, so for ridge sampling gives no time saving. Honest negative result for the conclusion slide.
Columns (rows fixed at 200,000, LightGBM): importance-based column choice (gain from a rough model fitted on a uniform training sample) vs uniform random columns: k = 40: 0.1102 vs 0.1770; k = 80: 0.1087 vs 0.1248; k = 160: 0.1058 vs 0.1129; full 342 columns: 0.1045.
Slopes (log–log, Plot 3 vs Plot 5): ridge parameter difference falls at 0.66× the training-gap slope (theory about 0.5); LightGBM prediction difference 0.33×.
Open items for this track: exclusive timing run (1–2 Dec), a model-agnostic check that the coreset's advantage at mid sizes is not an artefact of the weight normalisation (rerun with unnormalised weights on one size), slide text, reviewer-agent pass.

## 23. Replication of the G3/G3b feature lead (25 Nov 15:35): it does not hold

Delta vs `cs02` of the same seed (negative = better), RMSLE:

| Variant, seed | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|
| `g3b_all_plus_g3`, 07 | +0.0294 | +0.0021 | −0.0029 | −0.0045 |
| `g3b_all_plus_g3`, 08 | +0.0420 | +0.0142 | +0.0098 | −0.0019 |
| `g3b_all_plus_g3`, 09 | +0.0552 | +0.0031 | +0.0053 | −0.0051 |
| `feat_all`, 07 | +0.0531 | −0.0001 | +0.0013 | −0.0020 |
| `feat_all`, 08 | +0.0604 | +0.0067 | −0.0007 | −0.0010 |
| `feat_all`, 09 | +0.0508 | +0.0180 | −0.0032 | −0.0036 |

- Fold 4 improves on every seed (about −0.002 to −0.005) but fold 3 flips sign between seeds and fold 2 gets worse on every seed. Averaged over folds 3–4 the best case is −0.0016 (`feat_all`), below the 0.0024 gate, and folds 1–2 are clearly worse.
- **Decision: no feature set from G3 or G3b is adopted.** `cs02` stays the recipe. The earlier "folds 3–4 represent the test" reasoning is kept as an argument but is not enough alone; fold 5 (summer fold) and the event-day features (G3c, being written) are the remaining feature tests.

## 24. G3c (event-day features) and a limit on what holiday features can do (25 Nov 15:50)

55 tests pass; 36 `g3c_` columns in four groups (holiday history incl. same-type previous holiday, calendar counts and distances for holiday / school closure / promo, open-Sunday profile, hub promo ratio). **Finding that limits the whole idea:** `RegionalHoliday` is a code (1 ordinary holiday, 2 Easter Friday/Monday, 3 Christmas); **the real test window (2015-06-20 to 07-31) has `RegionalHoliday` = 0 everywhere**, and so does fold 5 (summer 2014). Any gain from the holiday group can only be measured on folds 3 and 4 and cannot carry over to the final test. The calendar's school-closure and promo parts and the Sunday profile still apply to summer. Therefore: G3c runs on folds 3, 4, 5, 2; only the groups that help on fold 5 are candidates, and a gain that appears only on folds 3–4 through holiday columns is recorded but not adopted.

## 25. Fold 5 (summer 2014) and G3c verdicts (25 Nov 17:50)

**Fold 5, three seeds each, RMSLE on all rows (excluding open-zero rows in brackets).** Fold 5 has many open-but-zero rows, so "all" is about 0.06 above the "excl" number for every model. Differences between variants are small compared with that.

| variant | 3 seeds (all rows) | 3-seed log-mean | mean excl open0 |
|---|---|---|---|
| `base` (colsample 0.8) | 0.1747 / 0.1766 / 0.1731 | 0.1736 | 0.1163 |
| `cs02` | 0.1741 / 0.1754 / 0.1811 | 0.1754 | 0.1193 |
| `cs02_tweedie` | 0.1780 / 0.1767 / 0.1839 | 0.1782 | 0.1224 |
| `pertier` | 0.1751 / 0.1751 / 0.1742 | 0.1735 | 0.1162 |
| `g3b_all_plus_g3` | 0.1732 / 0.1744 / 0.1705 | 0.1714 | 0.1131 |
| `g3b_all` | 0.1832 / 0.1789 / 0.1735 | 0.1766 | 0.1216 |

- Seed spread on fold 5 is large (`cs02` 0.1741–0.1811), larger than on folds 1–4. A fold-5 gap under about 0.003 is noise.
- `cs02` does **not** show its folds 1–4 gain on summer: `base` is 0.0018 better on the 3-seed average, inside the noise. It is not worse by enough to change the recipe, because the 4-fold CV gain (0.1312 → 0.1190) is large and consistent. This stays a caveat, not a reversal. The final model averages seeds, which removes most of this variance.
- `pertier` is the steadiest member on fold 5 (spread 0.0009) and matches `base`. Keeps its place as an optional blend member.
- `g3b_all_plus_g3` is the best fold-5 entry (0.1714, −0.0040 vs `cs02`), but `g3b_all` alone is worse than `cs02`. Together with folds 1–3 (fold 3 flips, fold 2 worse, fold 1 breaks on gap hubs) this fails the gate (it passes fold 4 and fold 5 only). **Not adopted.** Blends of `cs02` with it give 0.1728 vs 0.1754 alone, which is the same size as the seed noise, so the blend is not adopted either.

**G3c (seed 20260807, RMSLE all rows; `cs02` reference from the same seed: fold 2 0.1202, fold 3 0.1109, fold 4 0.1020, fold 5 0.1741 [0.1151 excl]).**

| variant | fold 3 | fold 4 | fold 5 | fold 2 |
|---|---|---|---|---|
| `g3c_calendar` | 0.1178 | 0.1052 | 0.1742 | 0.1209 |
| `g3c_sunday` | 0.1164 | 0.1048 | 0.1791 | 0.1263 |
| `g3c_promo` | 0.1188 | 0.1055 | 0.1758 | 0.1201 |
| `g3c_holiday` | 0.1122 | 0.1032 | 0.1744 | 0.1196 |
| `g3c_all` | 0.1107 | 0.1044 | 0.1806 | 0.1190 |
| `g3c_all_plus_g3b` | 0.1157 | 0.1031 | 0.1749 | 0.1190 |

- Every group is worse than `cs02` on fold 3 (except `g3c_all`, equal) and on fold 4, equal or worse on fold 5, and within noise on fold 2. **No G3c group is adopted.** The event-day features add columns without adding information the trees did not already get from `DayOfWeek`, `Promo` and the existing holiday flags.
- Feature work (G3, G3b, G3c) is closed: no feature set passes the gate. Recipe stays `cs02` (lr 0.025, 1000 fixed rounds, seed-averaged), with `pertier` as the optional member.

**Still running or queued:** MLP on Vinayak (torch now cu128, CUDA works; 11 tests pass; smoke then full fold 1–4 run started 17:45) and the blend check against `cs02`.

## 26. MLP (G4) verdict (25 Nov 18:40)

Vinayak, torch 2.11 cu128, 40 epochs, seed 20260807, folds 1–4. MLP alone: 0.2454 / 0.1825 / 0.1200 / 0.1095 (mean 0.1643) vs `cs02` 0.1377 / 0.1215 / 0.1129 / 0.1039 (mean 0.1190). Log-space blend with `cs02`:

| MLP weight | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|
| 0 (`cs02`) | 0.1377 | 0.1215 | 0.1129 | 0.1039 |
| 0.1 | 0.1423 | 0.1215 | 0.1115 | 0.1032 |
| 0.2 | 0.1489 | 0.1232 | 0.1105 | 0.1028 |
| 0.3 | 0.1573 | 0.1266 | 0.1101 | 0.1026 |

Helps folds 3–4 by at most 0.003, hurts folds 1–2. Fails the gate (needs ≥2 of 3 dev folds). **MLP not adopted.** Model-family diversity (CatBoost, Tweedie, MLP) has now been tried three times with no gain over seed-averaged LightGBM.

Random hyper-parameter search (G6, 20 configs around `cs02`, folds 1–4, seed 20260807) started on Manjot at 17:44.

## 27. Decision (25 Nov 19:10): test every remaining item, then finalise

The earlier plan said to cut low-priority items. **Changed on instruction: every item still on the ledger is run and gets a recorded verdict before the recipe is frozen.** The gate (section 12 / PLAN_V4 §2) is unchanged and applies to each. Nothing is adopted early; the final recipe is chosen after all rows below have a verdict, by 2 Dec, frozen 3 Dec.

| # | Item | Origin | Where | Owner / state |
|---|---|---|---|---|
| 1 | Random hyper-parameter search (Optuna item 2.4), 20 configs, folds 1–4, then top 3 on 3 seeds and fold 5 | long plan 2.4 | Manjot | running since 17:44 |
| 2 | Chronos-2 zero-shot pilot + blend with `cs02` | 3.1 | Vinayak GPU | agent running |
| 3 | Adversarial validation + permutation-importance pruning | v3, Phase 4 | Manjot | agent running |
| 4 | Final-retrain script, SHA-256 manifest, sanity checks, fold-4 dry run | G5 | Manjot | agent running |
| 5 | TiDE, NHITS, TabPFN-TS pilots (global models on log1p, blend check vs `cs02`) | 3.2 / 3.1b | Vinayak GPU | queued, agent |
| 6 | Summer-2014 incomplete origins (1b.2), season-filtered member, 7-day origins | 1b.2, winner sweep | Manjot / Vinayak CPU | queued, agent |
| 7 | SVD denoising (rank 12–15 of the hub x day matrix as features) | winner sweep | CPU | queued, agent |
| 8 | Month-ahead model, two-stage session model | v3 | CPU | queued, agent |
| 9 | Waiting list: peer-hub residuals, regime experts, hierarchical reconciliation, shift adaptation | v3 | CPU, offline from OOF where possible | queued, agent |
| 10 | Schema assertion in `model_pipeline.py` (0.2) | long plan | local | queued |
| 11 | Coresets from GBDT leaves (end-term extra) | v3 | Vinayak | after item 5 |
| 12 | Gate pass on every item; combine passers; 3-seed + fold-5 confirmation | PLAN_V4 §2 | both | after 1–9 |
| 13 | Recipe pick by CV | — | — | by 2 Dec |
| 14 | Exclusive timing run (end-term) | end-term | idle server | 1–2 Dec |
| 15 | **Freeze, final retrain, SHA-256, sanity checks, STATUS.md** | G5 | Manjot | 3 Dec |

Closed with verdicts (no rerun): level/offset/recency, Tweedie, CatBoost and MLP blends, NNLS, calibration, horizon split, per-format and per-tier, G3, G3b, G3c, fold 5 (sections 12–26).

Rules: one change at a time against `cs02`; results land as one row per run in `experiments.csv`; every verdict is written here as the next numbered section; the servers always have a queued job.

## 28. Chronos-2 zero-shot pilot (item 3.1), 25 Nov 19:30: not adopted

History = all raw rows up to the forecast origin; 0.5 quantile, expm1, clipped at 0, closed future days set to 0; 13 tests pass. Covariates: IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag.

| RMSLE, all rows | fold 1 | fold 2 | fold 3 | fold 4 | mean |
|---|---|---|---|---|---|
| Chronos-2 univariate | 0.278 | 0.234 | 0.339 | 0.263 | 0.2785 |
| Chronos-2 with covariates | 0.206 | 0.157 | 0.178 | 0.156 | 0.1743 |
| `cs02` | 0.1377 | 0.1215 | 0.1129 | 0.1039 | 0.1190 |

Blend gain over `cs02` (covariates variant; positive = better): weight 0.1 gives −0.0006 / +0.0016 / +0.0018 / +0.0007 (mean +0.0009); weight 0.2 gives −0.0029 / +0.0019 / +0.0013 / −0.0001 (mean 0.0000); weight 0.3 gives mean −0.0025. The univariate version loses at every weight. Best fold gain 0.0019, below the 0.0024 gate; fold 1 is worse. **Not adopted.** Code in `improve/g7_chronos/`, results in `endterm/results/CHR/` locally.

## 29. TiDE and NHITS pilots (items 3.2), 25 Nov 20:30: not adopted; TabPFN-TS pending

neuralforecast 3.4.0 in a separate venv; history includes the origin day; closed-day targets interpolated in history (without it, smoke RMSLE was 0.86–0.95); 26 tests pass. TabPFN v3.5/v2.5 need a browser licence login, so TabPFN-TS uses the ungated TabPFN v2 checkpoint. Configs were chosen on a fold-4, 100-hub smoke, a small leak into fold 4 only.

| RMSLE, all rows | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|
| TiDE | 0.2210 | 0.2138 | 0.2042 | 0.1855 |
| NHITS | 0.3717 | 0.1562 | 0.1604 | 0.1394 |
| `cs02` | 0.1377 | 0.1215 | 0.1129 | 0.1039 |

Blend gain over `cs02` (positive = better), folds 1 / 2 / 3 / 4: TiDE w0.1 +.0017 / −.0009 / −.0028 / +.0013; w0.2 +.0006 / −.0044 / −.0076 / −.0002. NHITS w0.1 −.0045 / +.0009 / −.0009 / +.0007; w0.2 −.0167 / +.0008 / −.0028 / +.0004. Best mean gain: TiDE −0.0008, NHITS +0.0003, both far below 0.0024. **TiDE and NHITS not adopted.**

**TabPFN-TS:** fold 4 on 100 hubs only (0.1096 alone vs 0.0938 for `cs02` on those hubs; blend gains +0.0019 / +0.0030 / +0.0033 at weights 0.1 / 0.2 / 0.3). A subset is not a verdict; the full 4-fold run (about 14 min per fold) was still running on Vinayak when the VPN dropped. Verdict after the OOF is pulled and `neural_eval.py` is run.

## 30. Summer-2014 origins, season-filtered member, 7-day origins (1b.2), 25 Nov 21:00: none adopted

16 origins on the 14-day grid (2014-06-01 to 2014-12-28) have a 42-day horizon that touches the 180-hub gap (2014-07-01..12-31); the loader drops them because they are not full 46,830-row blocks. Fold 5 was rebuilt (41,250 rows = 46,830 − 180 hubs × 31 days; the builder reproduces an existing origin file exactly). Control `cs02` reproduces within 0.0001 on folds 1–4 (fold 5: 0.1743 vs 0.1741). 20 tests pass.

Gain over `cs02` (positive = better; mean of seeds 07/08/09):

| Variant | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | folds ≥ 0.0024 | Gate |
|---|---|---|---|---|---|---|---|
| (a) add incomplete summer-2014 origins | +.0011 | +.0012 | +.0009 | +.0018 | n/a | 0 | fail |
| (b) blend `cs02` + season member, w 0.3 | +.0032 | +.0017 | +.0043 | +.0005 | +.0018 | 2 | borderline |
| (b) blend w 0.5 | +.0036 | +.0021 | +.0060 | −.0002 | +.0017 | 2 | borderline |
| (c) dense 7-day origins | +.0009 | −.0006 | −.0008 | −.0021 | −.0006 | 0 | fail |

- (a) mean +0.0013, noise level. Fold 5 cannot change: it can only train on origins ending by 2014-06-19.
- (b) passes folds 1 and 3 but fails folds 2–4 as a set and fold 5. **Averaging `cs02` with a second `cs02` seed already gains about +0.001, so a good part of (b) is variance reduction, not season information.** The season member alone is poor (4–6 training origins, mean gain −0.0002). Not adopted; to be re-tested only against a two-seed `cs02` control if time remains.
- (c) doubles rows (0.93 M → 1.80 M), memory (10.5 → 18.8 GB) and fit time (62 → 110 s) for no gain. Not adopted.
- Season rule ±45 days leaves almost nothing for fold 5 (only July 2013 exists in that season).

## 31. Final-retrain pipeline (G5) built and dry-run, 25 Nov 21:15

`improve/g8_final/`: `final_retrain.py`, `sanity_checks.py`, `test_final.py` (16 tests pass locally and on the server). The real retrain has **not** been run; it is the 3 Dec step.

- Setting: LightGBM `cs02`, lr 0.025, 1000 fixed rounds, three seeds (20260807/08/09) averaged in log space, plus optional per-tier set (`--blend-mode member|half`, `--pertier-seeds`, default first seed only). `--preflight` checks inputs and exits. It refuses to overwrite an existing manifest.
- Outputs mirror the control run: `lightgbm_s<seed>.txt`, `pertier_*.txt`, `test_pred_*.npy`, `submission_final.csv`, `final_manifest.json` with a SHA-256 for every model and for the submission.
- Dry run on fold 4 (33 origins, seed 20260807, 16 threads): global model RMSLE **0.10172**; global + per-tier **0.10132** (the fold-4 score for the final setting). All sanity checks pass (closed rows 0, no NaN or negatives, 46,830 rows).
- Real-run sanity path run with the control submission as both prediction and reference: all checks pass, Id order matches the test frame.
- Preflight on the real test frame: 37 origins whose labels end 2015-06-19 (one day before the test window), 341 feature columns present.
- Estimated wall time for the real retrain on Manjot with 40 cores: 30–40 min (3 seeds + one per-tier set), 50–60 min with three per-tier seeds, about 15 min of it single-threaded loading.

## 32. SVD, two-stage, peer residuals, shift adaptation, reconciliation, regime experts (items 7–9), 25 Nov 22:30: none adopted

66 tests pass (`improve/g12_misc/`, one module per item). Gain = `cs02` minus variant RMSLE (positive = better); `cs02` per fold 0.1377 / 0.1215 / 0.1129 / 0.1039; gate 0.0024 on ≥ 2 of 3 dev folds (2, 3, 4).

| Item | Design | Result | Verdict |
|---|---|---|---|
| A SVD denoising | rank 14, `g12a_*` features, folds 3–4, seed 07 | fold 3 −0.0054, fold 4 0.0000 | fails; folds 1, 2, 5 not run |
| B two-stage | cross-fitted weekly total feeds the day model, folds 3–4 | full −0.0005 / −0.0016; minimal +0.0010 / +0.0002 | fails |
| C peer residuals (offline, folds 2–4) | same tier × format peers | about −0.0001 per fold | fails |
| | correlation peers (corr5, 28-day window) | +0.0009 / +0.0004 / −0.0001 | fails |
| | own residual + corr5 | +0.0027 / +0.0015 / +0.0007 | challenger, 1 of 3 folds, not adopted |
| D shift adaptation (offline, 14-day window, fitted beta ≈ 0.3) | +0.0016 / +0.0011 / +0.0013 (other windows +0.0012 / +0.0015 / +0.0010) | positive on every fold and seed but misses the gate; literal beta = 1 hurts (−0.017 / −0.004 / −0.003); needs an extra previous-origin forecast at deployment | weak challenger, not adopted |
| E hierarchical reconciliation | fitted shrink weights 0 or negative | gain about 0.0000 on every fold | fails |
| F regime experts | hard-gate blend −0.0066 / −0.0015; half-weight −0.0021 / +0.0005; special-rows-only −0.0004 / +0.0003 | open Sunday/holiday rows are only 1.7–3% of squared error, so the ceiling is about 0.0016 | fails |

Seed-agreement and open-zero checks were done for C, D, E; A, B, F stopped at folds 3–4 on seed 07 because they already failed. The ledger's waiting list (peer residuals, regime experts, reconciliation, shift adaptation) is now closed with verdicts.

## 33. Adversarial validation and permutation-importance pruning (G9), 26 Nov 00:10: nothing actionable

43 tests pass (`improve/g9_diagnostics/`).

**Adversarial validation** (folds 1–5 and the real test window, hub-grouped 3-fold CV): AUC is 1.000 for every target, including the control (last training origin vs earlier ones) and even the raw no-calendar set (0.99999 on fold 4); the reduced set that also drops promo, holiday and history-length columns gives 1.000 on real data against 0.997–1.000 for the control. The test cannot show drift, because each origin can be recognised from cross-hub patterns alone, and the test window cannot be told apart from the latest origins either. No feature is flagged. Top non-calendar features by gain share: `Sessions_AtOriginMinus7D` (~0.22), `Orders_AtOriginMinus7D` (~0.17), then the 14- and 28-day versions, then `Orders_AtOrigin`: day-specific cross-hub signals, not distribution drift.

**Permutation importance on `cs02`** (folds 3–4, seed 07, 3 repeats, cost in log RMSE): hub_stats 0.239 (almost all `HistHub*`), rolling 0.030, promo 0.028, calendar 0.021, holiday 0.017, encoding 0.014, lags 0.010. No group is at or below 0 on both folds; only the 6 `lags_avail` flags are about 0.

**Pruning, 3 seeds, gain = `cs02` minus pruned (positive = better):**

| Variant | fold 1 | fold 2 | fold 3 | fold 4 | Gate |
|---|---|---|---|---|---|
| p1: drop 81 columns under the 0.0012 noise floor on both folds | −0.0063 | −0.0051 | +0.0028 | +0.0013 | fail (fold 5: −0.0065) |
| p2: drop the 6 `lags_avail` flags | +0.0041 | −0.0004 | −0.0000 | −0.0003 | fail (fold-1 gain not seed-stable) |

**Not adopted; the column set stays as is.**

## 34. Hyper-parameter search (G6) and TabPFN-TS fold 5, 26 Nov 01:30

Full 20-config table with per-fold scores: `RESULTS_AND_VERDICTS_2026-11-26.md` section 2 and `research/hp_search_20configs_2026-11-26.csv`. The control `hp_000` (= `cs02`) reproduces 0.1378 / 0.1215 / 0.1129 / 0.1039.

- **`hp_017` (colsample 0.15, 255 leaves, min_child 160, lambda 24.1) is the only config better on all four folds:** 0.1288 / 0.1202 / 0.1078 / 0.1027, mean 0.1149 against 0.1190 (gain 0.0041; folds 2–4 gain 0.0025 against the 0.0024 gate). Best of 19 alternatives on the same folds, so the gain is optimistic. **Candidate; confirmation running:** seeds 08 and 09 on folds 1–4, seeds 07/08/09 on fold 5, and nine neighbour variants registered as `hp17*` in `exp_runner.py` (lambda 10/50, 511 and 127 leaves, min child 320, colsample 0.1/0.2, fixed 1000 rounds).
- Pattern: 255 leaves with strong L2 and large min child wins; all 31-leaf configs lose on every fold.
- **TabPFN-TS on fold 5** (935 hubs with a full block; the runner now drops partial-block hubs): alone 0.1778 against 0.1784 for the 3-seed `cs02`; blend gain +0.0020 / +0.0036 / +0.0048 at weights 0.1 / 0.2 / 0.3. Together with folds 2 and 4 (+0.0025, +0.0029 at w 0.3) it is positive on three of four non-winter folds, negative on fold 1 (−0.0120), and misses the gate on folds 2–4 (+0.0018). Stays a **challenger**; tested again on top of the winning `cs02` variant before the freeze.

## 35. `hp_017` confirmation on three seeds, 26 Nov 02:00

`hp17` = colsample 0.15, 255 leaves, min_child 160, lambda 24.1 (lr 0.05, early stopping). Per-seed 4-fold mean: seed 07 0.1149, seed 08 0.1156, seed 09 0.1167, against `cs02` 0.1190 / 0.1186 / 0.1191 (gains +0.0041 / +0.0030 / +0.0024). Per fold, seed 08 / 09 gains: fold 1 +0.0066 / +0.0083, fold 2 +0.0019 / +0.0012, fold 3 +0.0024 / +0.0017, fold 4 +0.0011 / **−0.0016**. Fold 5 (seeds 07/08/09): 0.1738 / 0.1727 / 0.1731 against `cs02` 0.1741 / 0.1754 / 0.1811 (gains +0.0003 / +0.0027 / +0.0080; excluding open-zero rows 0.1138 vs 0.1193).

**3-seed log-space averages, `cs02` vs `hp17`:** fold 1 0.1340 vs 0.1265 (+0.0075), fold 2 0.1194 vs 0.1177 (+0.0016), fold 3 0.1124 vs 0.1094 (+0.0030), fold 4 0.1035 vs 0.1032 (+0.0003); 4-fold mean 0.1173 vs 0.1142 (**+0.0031**).

**Gate reading (stated plainly):** the 4-fold mean gain is 0.0031, above 0.0024, and the sign is positive on every fold in the 3-seed average and on every seed's 4-fold mean. The stricter reading, folds 2–4 only, gives +0.0016 and **misses** the 0.0024 mark; fold 4, the fold nearest the test window, is flat (+0.0003), and one seed is slightly negative there. Most of the gain comes from folds 1 and 3. The best-of-19 selection effect is visible: the seed-07 gain (+0.0041) shrank on the two fresh seeds (+0.0030, +0.0024). Realistic expected gain over `cs02`: about 0.003 on the 4-fold mean, near zero on fold 4.

**Status:** strong challenger, used as the base for the stacking tests (lr 0.025 with fixed rounds, per-tier member, TabPFN-TS blend, neighbour sweep running as `exp_out_HP17N`). Final adoption is decided by CV at the 2 Dec pick, with the folds 2–4 miss and the flat fold 4 stated in the decision.

## 36. Stack test: `hp17` + TabPFN-TS (weight 0.2) against `cs02`, 26 Nov 03:00

Both sides use the same seeds (`cs02` seeds 07/08/09 vs `hp17` seeds 07/08/09); TabPFN-TS is a single run (seed 07). Gain = `cs02` minus stack (positive = better), RMSLE all rows. Columns: seed 07 / 08 / 09 / 3-seed log average.

| fold | seed 07 | seed 08 | seed 09 | 3-seed avg |
|---|---|---|---|---|
| 1 | +0.0018 | −0.0004 | +0.0009 | −0.0005 |
| 2 | +0.0046 | +0.0043 | +0.0034 | +0.0039 |
| 3 | +0.0046 | +0.0023 | +0.0019 | +0.0024 |
| 4 | +0.0034 | +0.0035 | +0.0017 | +0.0024 |
| 5 (935 hubs with a full block) | +0.0033 | +0.0054 | +0.0111 | +0.0058 |
| mean folds 2–4 | +0.0042 | +0.0034 | +0.0023 | **+0.0029** |

Excluding open-zero rows, 3-seed average: fold 1 −0.0005, fold 2 +0.0041, fold 3 +0.0024, fold 4 +0.0024. Weights tried on the same folds: 0.1 gives folds 2–4 mean +0.0025, 0.2 gives +0.0029, 0.3 gives +0.0026. `hp17` alone gives +0.0016 on folds 2–4 (misses the gate); the TabPFN-TS blend adds about +0.0013 on top.

**Gate reading:** mean gain over folds 2–4 is 0.0029 (≥ 0.0024); improved on 3 of 3 folds; positive on every fold 2–5 for every seed; survives excluding open-zero rows; fold 5 checked and positive on all seeds. Seed 09 alone is 0.0023, 0.0001 under the mark. **The stack passes the gate on the seed-averaged reading and is the leading candidate, not yet the champion.** Open points that keep it a candidate: (1) `hp17` was the best of 19 on the same folds and its gain shrank on fresh seeds (selection effect); (2) the weight 0.2 was picked from three values on the same folds; (3) TabPFN-TS has one seed and was tuned on a fold-4, 100-hub smoke (a small leak into fold 4); (4) fold 1 is flat, so the stack is not better on the winter fold; (5) the final pipeline does not yet predict the real test window with TabPFN-TS. These are closed by the steps in `FINAL_PLAN_2026-11-26.md`.
