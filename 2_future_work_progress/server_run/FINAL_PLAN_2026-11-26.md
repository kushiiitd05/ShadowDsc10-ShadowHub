# Final plan: what is decided, what is left, and the order (26 Nov 2026)

Supersedes the queue in `PLAN_V4_FINAL_2026-11-25.md` section 4. The gate and rules are unchanged. Every result is in `ENDTERM_AND_IMPROVEMENT_PLAN_2026-11-25.md` (§10–§36) and `RESULTS_AND_VERDICTS_2026-11-26.md`. Dates: model freeze **3 Dec**, submission **8 Dec**, presentations **9–10 Dec**.

## 1. Where we stand (evidence snapshot)

| Role | Recipe | Status |
|---|---|---|
| Confirmed champion | `cs02`, lr 0.025, 1000 fixed rounds, 3 seeds averaged in log space; optional per-tier member | 4-fold mean 0.1173 (3-seed average); fold-4 dry run 0.10172 (0.10132 with per-tier) |
| Leading candidate | `hp17` (colsample 0.15, 255 leaves, min child 160, lambda 24.1), 3 seeds, blended with TabPFN-TS at weight 0.2 | passes the gate on the seed-averaged reading (§36): folds 2–4 gain +0.0029, 3 of 3 folds, fold 5 positive on all seeds |
| Challenger | `hp17` alone | 4-fold gain +0.0031, folds 2–4 +0.0016 (misses the gate); fold 4 flat |
| Closed, not adopted | all others in `RESULTS_AND_VERDICTS` section 3 | final |

## 2. Remaining work, in order (nothing else is added)

| # | Step | Where | Output / pass condition | Date |
|---|---|---|---|---|
| 1 | Read the neighbour sweep around `hp17` (lambda 10/50, 511 and 127 leaves, min child 320, colsample 0.1/0.2, fixed 1000 rounds); folds 1–4, seed 07 | Manjot (running) | pick at most one better corner; it needs seeds 08/09 + fold 5 before use | 26 Nov |
| 2 | `hp17` with lr 0.025 and fixed rounds (the final-retrain setting), 3 seeds, folds 1–4 and 5 | Manjot | must not lose to `hp17` at lr 0.05 by more than noise (0.0012) | 26–27 Nov |
| 3 | Per-tier member built from the `hp17` recipe, 3 seeds | Manjot | counts only if it adds ≥ 0.0012 on folds 2–4 on top of the stack | 27 Nov |
| 4 | TabPFN-TS: seed 08 (and 09 if the model is stochastic), and weight chosen on folds 2–4 only, then fixed | Vinayak GPU | the 0.2 weight must stay within 0.0012 of the best weight; report fold-1 loss | 27 Nov |
| 5 | TabPFN-TS on the **real test window** (history to 2015-06-19, 42 forecast days, same covariates) added to `g10_neural`, with a test, and run on the GPU (about 13–15 min) | Vinayak | prediction file with no NaN, closed days 0, Ids aligned to the test frame | 27–28 Nov |
| 6 | Extend `g8_final/final_retrain.py`: recipe = the chosen `hp17` variant; blend the TabPFN-TS test predictions at the fixed weight; manifest hashes both; refuse to run if the recipe differs from the registered variant | local + Manjot | tests pass; dry run on fold 4 reproduces the stack's fold-4 score within 0.001 | 28–29 Nov |
| 7 | Stack gate table, final: stack vs `cs02`, seeds 07/08/09, folds 2–5, with and without open-zero rows, written as plan §37 | local | **adopt** only if mean gain on folds 2–4 ≥ 0.0024, improves ≥ 2 folds, same sign on all seeds, fold 5 positive; otherwise keep `cs02` (+ per-tier) | 29–30 Nov |
| 8 | Evidence audit (items A–E below) written as plan §38 | local | each item marked done / limitation | 30 Nov–1 Dec |
| 9 | Exclusive timing run for the end-term on an idle server, nothing else running on it | Manjot or Vinayak | speed-up table from exclusive timings only | 1–2 Dec |
| 10 | Recipe pick by CV, written decision | local | one line in `STATUS.md`: recipe, gate evidence, limitations | by 2 Dec |
| 11 | **Freeze and real retrain** (30–40 min), saved models + SHA-256, `sanity_checks.py` on the submission, record in `STATUS.md` | Manjot | all checks pass; submission hash recorded | 3 Dec |
| 12 | Report (8 sections), evidence audit, 7-minute presentation brief, from `REPORT_PROMPT_2026-11-26.md`; slides 6–9 | Mac | reviewed against the plan sections, banned-name and number scan (no outside scores, no server or key details) | 3–7 Dec |
| 13 | Rehearsal, submission | Mac | submit | 6–8 Dec |

If step 7 fails, the final model is `cs02` (3 seeds, lr 0.025, 1000 rounds) with the per-tier member if it passed its own check; step 5 and the TabPFN-TS part of step 6 are then dropped. The retrain script already supports that path.

## 3. Audit items (from the external review) and how each is handled

| Item | Finding so far | To do |
|---|---|---|
| A. End-term test = dev fold 4; fold-4 labels were used for model selection | True: the end-term split uses the last 42 days (fold 4) as test, and `cs02`, lr and feature decisions were chosen using fold-4 scores. Training rows end before the fold-4 origin, so there is no feature or label leakage into the fit. The end-term models (default LightGBM and ridge, tuned on a training-only part) were not chosen from fold-4 labels, but the recipe family was | state it as a limitation; call the number a dev-fold result, not an untouched holdout; no replacement score invented |
| B. Bagging really active? | Yes: `subsample` 0.85 is always passed with `subsample_freq = cfg.subsample_freq`, and `cs02` and `hp17` set `subsample_freq = 1`; the old `base` recipe has `subsample_freq = 0`, so bagging is off there (`exp_runner.py` line ~204, `final_retrain.py` line ~67) | one fitted-model parameter print on the server before the freeze (step 8) |
| B. Mixed settings | the table in `RESULTS_AND_VERDICTS` §1 keeps seeds, learning rate and evaluation protocol per row; `hp17`'s 0.1149 is single seed, lr 0.05, early stopping | keep the labels; never compare across rows without them |
| C. Best of 19 | stated in §35 and §36; `hp17` stays a candidate until step 7 | done in text |
| D. Fold 5 | full fold has 41,250 rows (180 gap hubs only have 11 of 42 days); TabPFN-TS and stack comparisons use the 935 hubs with full blocks; fold-5 error is larger (about 0.173 vs 0.115 without open-zero rows) because it has more open-zero rows | state both populations in every fold-5 number |
| E. End-term conventions | coreset: with replacement, p = s/Σs, weight 1/(m·p); duplicates are kept as repeated rows each carrying its own weight (equivalent to aggregation); uniform baseline without replacement; 5 repetitions per size, medians; R² is on the log1p scale with the closed-day rows included (their error is 0 and they raise the variance, so R² is higher than on open days only); the ridge coreset build costs more than a full ridge fit; column-importance subset (keep the k columns with the highest importance) is a different method from a random linear projection; timings from the shared server are not final | re-run timings exclusively (step 9); define both column methods in the slides; report R² with its scale |
