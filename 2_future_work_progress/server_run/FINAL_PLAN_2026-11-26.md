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

---

# Update, 26 Nov evening (supersedes the table in section 2 where they differ)

## A. Verdict as of now
- **Best validated internal model:** `hp17` (colsample 0.15, 255 leaves, min child 160, lambda 24.1; lr 0.025, 1000 fixed rounds; 3 seeds averaged in log space) blended with TabPFN-TS at weight 0.2 (log space). In the final-retrain setting its gain over `cs02` (early stopping, same seed) on folds 2–4 is +0.0034 (3-seed average; each seed 0.0039–0.0044), positive on every fold 2–5 and on every seed, and it holds without open-zero rows (plan §37). Not yet declared champion: the like-for-like `cs02` baseline (lr 0.025, 1000 fixed rounds, 3 seeds) is still running, and the gate verdict is plan §38. `hp17` alone misses the gate (folds 2–4 +0.0016 to +0.0021).
- **Fallback** (kept ready): `cs02`, lr 0.025, 1000 fixed rounds, 3 seeds, optional per-tier member.
- **Closed:** per-tier on the `hp17` recipe (seed 07, folds 2–4: 0.1192 / 0.1124 / 0.1060 against 0.1188 / 0.1085 / 0.1023 for `hp17`) looks worse alone; offline blends on top of the stack add nothing: stack 0.0034 (folds 2–4 mean gain over `cs02`); + Chronos-2 at 0.1 gives 0.0035; + per-tier `cs02` at 0.2 gives 0.0034; mixing `cs02` and `hp17` 50/50 before the blend gives 0.0031. The recipe is saturated on these inputs; the only lever left that has moved the score is TabPFN-TS itself.

## B. What the external review proposed, and what we do with it
| Proposal | Decision |
|---|---|
| Phase 0: confirm baselines, bagging active, same schema/origins/rounds, per-fold with and without open-zero rows, gate status | mostly done: bagging verified (`subsample` 0.85 with `subsample_freq` 1 for `cs02`/`hp17`); per-fold tables with and without open-zero rows are in §36–§37; the like-for-like baseline finishes tonight; gate verdict = §38 |
| Phase 1: one bounded Optuna study (TPE, multivariate, seed fixed, 10 startup, 40 trials, SQLite storage, `cs02` and `hp17` enqueued as controls, lr 0.025 / 1000 rounds, space: colsample 0.10–0.35, leaves {63,127,255,383,511}, min child 60–400, lambda 1–100 log; objective on folds 2 and 3 only) | **accepted as an optional, time-boxed lane** (started 26 Nov). Expected gain is small: the 20-config search and the 8-variant neighbour sweep both found a plateau (all within 0.002 of `hp17`), and more tries add selection bias. It must not delay anything: it stops after 40 trials or at 1 Dec 12:00, whichever is first |
| Phase 2: top 5 finalists on folds 1–4, 3 seeds, fold 5, gate against `cs02`, also shown against `hp17`; bias disclosed | accepted; the gate is not weakened. A finalist replaces `hp17` only if it passes against `cs02` AND beats `hp17` by more than the noise (0.0012) on folds 2–4 |
| Phase 3: TabPFN-TS blend only on the validated recipe; no weight tuned on fold 4; end-to-end rehearsal of the real `final_retrain.py` path with `--recipe`, `--blend-file`, `--blend-weight` | accepted. Weights 0.1/0.2/0.3 were compared on folds 2–4 including fold 4 (gain 0.0031–0.0034, flat), so 0.2 is the middle value of a flat curve and is disclosed as such; no further weights are explored. Unit tests plus a dry run exist; the **full rehearsal on the real path is added as step R below** |
| Phase 4: compare A `cs02`, B `hp17`, C Optuna winner, D best recipe + TabPFN-TS, E per-tier; by CV only; no "state of the art" wording | accepted as the decision table in §38; wording rule adopted |
| Phase 5: freeze 3 Dec; all hashes, 46,830 rows, closed rows 0, finite non-negative, features only from data before the origin, exact command and software versions, keep a fallback, never overwrite | accepted; most already coded in `final_retrain.py` / `sanity_checks.py`; versions and exact command added to the manifest in step R |

## C. Extra levers worth the remaining time (ranked by expected gain)
1. **TabPFN-TS settings** (largest incremental contributor, weak on the winter fold): context length 730 and 180 days against 365 on folds 2–4 (running on the GPU). Adopt a change only if the stack gain on folds 2–4 improves by more than 0.0012 and fold 1 does not get worse.
2. **Five seeds instead of three** in the final LightGBM average (about 0.0003–0.0005, almost free; retrain time grows by about 4 minutes).
3. Optuna lane (above), expected below 0.001.
Not worth more time: more model families, more feature sets, more blend members (all measured, section A).

## D. Revised schedule
| Date | What |
|---|---|
| 26 Nov (tonight) | like-for-like `cs02` baseline finishes → §38 gate table; per-tier on `hp17` finishes; TabPFN context sweep; Optuna starts |
| 27–28 Nov | Optuna runs; TabPFN context sweep read; if a better TabPFN setting is found, regenerate the test-window file |
| 29 Nov | **Step R: full rehearsal retrain on the real path** (`--recipe hp17 --blend-file <TabPFN-TS test file> --blend-weight 0.2`, 3 seeds, own output folder, nothing overwritten) + `sanity_checks.py`; the output is also the confirmed fallback artifact |
| 30 Nov | Optuna finalists: seeds 08/09 + fold 5 + gate; evidence audit §39 |
| 1 Dec | Optuna stops (12:00); exclusive end-term timing run starts on an idle server |
| 2 Dec | recipe decision written (CV only): one of `cs02` / `hp17` / Optuna winner, with or without TabPFN-TS |
| 3 Dec | **freeze, real retrain (30–40 min), hashes, sanity checks, `STATUS.md`** |
| 3–7 Dec | report, evidence audit, presentation brief, slides 6–9, banned-term scan, rehearsal |
| 8 Dec | submit; 9–10 Dec presentations |

## E. Change at 26 Nov 05:30: Optuna now tunes the LightGBM settings and the blend weight together
Reason: the best candidate is `hp17` + TabPFN-TS, so the search objective is the blended score. Study `g6c_blend`: same LightGBM space as before (colsample 0.10–0.35, leaves {63,127,255,383,511}, min child 60–400, lambda 1–100) plus the weight w in 0.00–0.40 (step 0.05); objective = RMSLE of the log-space blend, mean over folds 2 and 3 only; 40 trials, TPE seed 20260807; controls enqueued: (`cs02`, w 0), (`cs02`, w 0.2), (`hp17`, w 0), (`hp17`, w 0.2). Stops at 40 trials or 28 Nov 18:00. Top 5 are then run on folds 1, 4, 5 and compared with `hp17` + 0.2; seeds 08/09 and the gate (§39) follow on 29–30 Nov. Selection bias (best of 40, weight tuned on two folds) is disclosed in the verdict. A winner replaces `hp17` + 0.2 only if it passes the gate against the like-for-like `cs02` baseline AND beats `hp17` + 0.2 by more than 0.0012 on folds 2–4; otherwise the freeze candidate stays `hp17` + 0.2.

## F. Compressed schedule: model decided and frozen by end of tomorrow (27 Nov)

New finding that changes the blend input: TabPFN-TS with a 730-day context scores fold 4 at **0.1041** alone (365 days: 0.1125; `cs02` 0.1035). A better TabPFN-TS member is the largest lever left, so the blend weight range is widened to 0.00–0.60 and Optuna uses the 730-day OOF.

| When | Manjot (CPU) | Vinayak (GPU + CPU) |
|---|---|---|
| tonight | Optuna study A (`g6c_blend`, TPE seed 20260807); rehearsal retrain on the real path (`hp17` + 365-day TabPFN-TS test file at 0.2, 3 seeds; checks the whole path end to end and is the first confirmed fallback artifact) | TabPFN-TS 730-day context: folds 4,3,2 (about 40 min), then folds 1 and 5; then the 730-day test-window file (about 13 min); Optuna study B (same script, TPE seed 20260808, own storage) on CPU |
| tomorrow 06:00–09:00 | read the 730-day OOF: stack gate with 730 (§39); if better than 365 by more than 0.0012 on folds 2–4, the 730 file replaces it everywhere | same |
| tomorrow 10:00 | both studies stop (40 trials or 10:00, whichever first); merge trial tables; top 5 overall (excluding controls) | finalists: folds 1, 4, 5 at seed 07 (Manjot and Vinayak split the five); top 2 also seeds 08/09 on folds 2–5 |
| tomorrow 15:00 | gate table (§40): finalist + TabPFN-TS vs like-for-like `cs02` baseline AND vs `hp17` + 0.2 (must beat it by > 0.0012 on folds 2–4) | evidence audit written |
| tomorrow 17:00 | **recipe decision written (CV only), `STATUS.md`** | |
| tomorrow evening | **real retrain on the decided recipe (30–40 min), hashes, sanity checks, fallback kept** | |
| 28 Nov – 7 Dec | no model changes (3 Dec stays as the last-resort date only); report, audit, presentation brief, slides 6–9; **exclusive timing run 1–2 Dec on an idle server**; banned-term scan; rehearsal; submit 8 Dec | |

Decision rule if nothing finishes in time: the freeze candidate is `hp17` (3 seeds, lr 0.025, 1000 fixed rounds) + TabPFN-TS 0.2, already passed the gate (§38); fallback `cs02` (same setting). Optuna and the 730-day input are accepted only on evidence, never on schedule pressure.

## G. Reviewer adjustments adopted (26 Nov, overrides sections E and F where they differ)
1. **One pooled search.** Optuna A (Manjot, sampler seed 20260807) and B (Vinayak, sampler seed 20260808) are one combined search of up to 80 trials; complete histories are pooled and the best-of-80 selection bias is disclosed. The LightGBM training seed is a separate constant (20260807) in both studies; sampler seeds and model seeds are never mixed.
2. **Blend weight fixed at 0.2** in the primary search (objective = folds 2–3 mean RMSLE of the log-space blend at 0.2; LightGBM-only RMSLE logged too). Other weights are a separate, explicitly logged ablation after the best LightGBM recipe is chosen. This replaces the weight-in-the-search design of section E.
3. **Top-3 average is not automatic.** It is compared with an equal-compute multi-seed ensemble of the best single configuration; the simpler one stays unless the gain exceeds the noise (0.0012).
4. **Gate for every promoted model:** original gate with matched settings (lr 0.025, 1000 fixed rounds), all three seeds, open-zero check, fold-5 diagnostic. The extra 0.0012 margin over `hp17` + TabPFN-TS is an additional condition, not a replacement.
5. **Bagging** is active (`subsample` 0.85 with `subsample_freq` 1 in `cs02`, `hp17` and all Optuna trials; the old `base` recipe had `subsample_freq` 0, so bagging was off only there). Baseline and candidates use the same setting. A fitted-model parameter dump is saved with the final artifacts.
6. **TabPFN-TS 365 vs 730** is compared on identical rows (same HubID/Date keys per fold), history cut at the forecast origin, closed days interpolated the same way, and on fold 5 on the 935 complete-block hubs (the 180 gap hubs are reported separately). 730 is promoted only if the stack gain improves on folds 2–4 by more than 0.0012 AND fold 1 and fold 5 do not get worse; never on fold 4 alone.
7. **Artifacts preserved:** the `hp17` + 0.2 rehearsal output (real blend path, running on Manjot now) and the `cs02` fallback are never overwritten; the rehearsal is the end-to-end real-path test.
8. **Resource control:** while the rehearsal runs on Manjot, Optuna A uses at most 24 threads (rehearsal 16); memory is watched; exclusive timing runs for the end-term happen only with the server idle (1–2 Dec).
9. **Wording:** folds 1, 4 and 5 are confirmation diagnostics, not untouched holdouts (all were used in earlier selection); fold 4 is also the end-term test window.
10. **At the decision point** the report contains: the exact gate table, per-fold and per-seed results, the experiment count (all trials + variants), compute cost, the blend-weight ablation and the recommendation. The search is not extended if a candidate narrowly misses. If `hp17` + TabPFN-TS 0.2 has passed every condition (it did against the like-for-like baseline, §38, with the limits stated there), it remains the preferred freeze candidate; anything else stays provisional until its full evidence exists.
