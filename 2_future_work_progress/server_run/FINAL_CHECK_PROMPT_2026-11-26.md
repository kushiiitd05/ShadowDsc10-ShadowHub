# Final check prompt: verify the freeze candidate before the real retrain

Attach: `FINAL_PLAN_2026-11-26.md`, `RESULTS_AND_VERDICTS_2026-11-26.md`, `ENDTERM_AND_IMPROVEMENT_PLAN_2026-11-25.md` (sections 35–41), `REPORT_PROMPT_2026-11-26.md`, `improve/g8_final/final_retrain.py`, `improve/g8_final/sanity_checks.py`, and the saved tables `research/g6c_pooled_80_trials_2026-11-26.csv` and `research/tabpfn_context_weight_grid_2026-11-26.csv`. If you can read the saved predictions and manifests, use them; mark every statement as "checked in artifacts" or "checked in the markdown only".

You are an independent reviewer. Do not run new model searches and do not change the adoption gate. Decide whether the candidate below may be frozen on the stated evidence, and list what must be fixed first.

## The candidate (notation)
For hub h and forecast day t, with ŷ the predicted order volume and ℓ(x) = log1p(x):

F(h,t) = 0 if the hub is closed on t; otherwise
F(h,t) = expm1( 0.8 · (1/3) · Σ_{s ∈ {20260807, 20260808, 20260809}} ℓ(ŷ^{hp17}_s) + 0.2 · ½ · ( ℓ(ŷ^{TabPFN-TS, 365 d}) + ℓ(ŷ^{TabPFN-TS, 730 d}) ) ), clipped at 0.

Per-component weights: each LightGBM seed 0.8/3 ≈ 0.267; TabPFN-TS 365-day context 0.10; TabPFN-TS 730-day context 0.10.
- `hp17`: LightGBM, colsample 0.15, subsample 0.85 with bagging every iteration, 255 leaves, min child 160, lambda 24.1, log1p target, hub target-encoding cross-fitted, learning rate 0.025, 1000 fixed rounds (no early stopping), trained on all 37 origins whose labels end 2015-06-19.
- TabPFN-TS: TabPFN v2 backbone, history up to the forecast origin only, closed days interpolated in the history, known-future covariates (open, promo, holiday, school-closure flags, calendar), closed future days forced to 0, deterministic.
- Fallbacks kept: `cs02` fixed-rounds 3 seeds; `hp17` + 365-day TabPFN-TS (rehearsal 1).

## Evidence to re-check (numbers must match the files)
- Like-for-like baseline: `cs02`, lr 0.025, 1000 fixed rounds, 3 seeds. 4-fold mean of the 3-seed average 0.1170.
- Dev-fold RMSLE (all rows; fold 5 = the 935 complete-block hubs), 3-seed average: `cs02` 0.1333 / 0.1186 / 0.1127 / 0.1033 / 0.1794; `hp17` alone 0.1275 / 0.1180 / 0.1083 / 0.1026 / 0.1756; candidate 0.1302 / 0.1141 / 0.1073 / 0.0999 / 0.1737 on folds 1 / 2 / 3 / 4 / 5. Excluding open-zero rows the candidate is 0.1231 / 0.1065 / 0.1073 / 0.0909 / 0.1105.
- Candidate vs `cs02` on folds 2–4: mean gain about +0.0045 (per seed 0.0045–0.0047), all three folds improve on every seed; fold 5 positive on every seed; fold 1 +0.0031 but 0.0027 worse than `hp17` alone.
- Pooled Optuna (80 trials, folds 2–3 objective, blend at 0.2): best trial 0.1105 vs `hp17` 0.1109: inside the 0.0012 noise margin; no replacement.
- Context/weight grid: flat within about 0.001–0.002; weight kept 0.2 (weight 0.3 improves folds 2–4 by only 0.0010 over 0.2; weights 0.4–0.5 lower fold 5).
- Rehearsal retrain 1 (365-day input) and 2 (mean-of-contexts input): all 12 sanity checks pass.
- The candidate's recorded evidence is on dev folds only. **There are no labels for the real test window in this project; no real test score exists or will be reported.** The only scores are the dev-fold numbers above.

## Questions to answer (each PASS / FAIL / CONDITION with the source)
1. Does the formula above match `final_retrain.py` (recipe `hp17`, blend in log1p space, closed rows 0, clip at 0, weights 0.8/0.2, TabPFN input = mean of the two contexts)? Is the 365/730 mix implemented as a pre-averaged file, and is that file reproducible from the two TabPFN files?
2. Leakage: do history windows for LightGBM features and for TabPFN-TS stop at the forecast origin (2015-06-19 for the real window)? Are the cross-fitted target encoding and the training-origin purge correct? Is anything in the blend file derived from the test labels (none exist)?
3. Gate: recompute the gate for the candidate against the like-for-like baseline from the saved OOF files (mean gain on folds 2–4 at least 0.0024, at least two folds improve, same sign on all three seeds, holds without open-zero rows, fold 5 checked). Report the numbers you get.
4. Selection bias: count the experiments that fed this choice (20 random-search configs, 8 neighbours, 80 Optuna trials, three TabPFN context lengths, a weight grid, one post-hoc mix of contexts). State what is optimistic: `hp17` best of 19 (gain shrank on new seeds), the mix defined after seeing 365 and 730, weight picked among three values on folds 2–4 including fold 4, TabPFN-TS configured on a fold-4 100-hub smoke, fold 4 also being the end-term test window.
5. Fold 1: the candidate is worse than `hp17` alone on fold 1 (0.1302 vs 0.1275) while better than `cs02`. Is this acceptable given a summer test window with no regional holidays? Is the claim "TabPFN-TS hurts winter and year-end" supported?
6. Final-retrain readiness: seeds, manifest hashes, saved models and configuration, exact command and software versions recorded, closed rows 0, 46,830 rows, exact Id order, finite non-negative values, fallback kept, nothing overwritten. What is still missing?
7. Wording: confirm that no document calls the model "state of the art", compares it with any outside score or ranking, or calls folds 1, 4 or 5 untouched holdouts.

## Output
A table of the seven checks (PASS / FAIL / CONDITION), a list of fixes required before the real retrain, a list of limits to state in the report, and one line: GO or NO-GO for freezing the candidate. Do not invent numbers; mark unverified items as unverified.
