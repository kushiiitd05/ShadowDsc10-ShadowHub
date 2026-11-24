# End-term and improvement plan v2 (2026-11-24)

Supersedes the "Improvement work (later)" notes in `STATUS.md`. `improvement_plan/IMPROVEMENT_PLAN.md` stays the long-form phase list; this file records what changed after the 24 Nov verification round and what is running.
Everything marked VERIFIED below was recomputed locally from the raw files, the committed metrics or the code. Anything else is labelled as a claim from an outside review and is not yet confirmed.

## 1. State

| Item | State |
|---|---|
| Best earlier model | CatBoost + LightGBM + LightGBM-Tweedie, log-space mean, seed `20260807`, earlier public score 0.06976 (RMSLE) |
| Model weights | **Not saved by any script and not found anywhere** (repo, Kaggle kernel output, Mac, both GPU servers). The 4.5 h run must be redone to get weights |
| Control rerun | Running on a GPU server (CPU GBDT). Same code, seed and pins (Python 3.11, numpy 1.25.2, pandas 2.3.3, scikit-learn 1.7.2, CatBoost 1.2.10, LightGBM 4.7.0; pyarrow pinned to 17.0.0). Rebuilds all 37 origins and 4 folds, then fits 3 models, saves each model with a SHA-256, timings and a control submission. Script: `endterm/control_rerun.py` |
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
