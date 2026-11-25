# Plan v4 (final), 25 Nov 2026

Supersedes v3 (section 8 of `ENDTERM_AND_IMPROVEMENT_PLAN_2026-11-25.md`). Nothing from v3 is dropped: every stage G0–G5, every outside-review idea (section 8.4) and the end-term track stay. This file adds measured results, baselines, a fixed decision rule, a per-server queue and the calendar. Submission 8 Dec; presentations 9–10 Dec; **model freeze 3 Dec**.

## 1. Baselines (what we have, what we lack)

| Baseline | Value | Status |
|---|---|---|
| Old best pipeline (3-model blend: CatBoost + LightGBM + Tweedie) | weights recovered; test predictions match the committed `submission_ensemble.csv` to RMS 3e-5 (log space) | **DONE.** Models and SHA-256 in `out_control/` on server A |
| Old best, CV score on the corrected folds | **not measured** | Control run has predictions on test rows only, no labels. Job B1 below measures it |
| LightGBM `base` (Phase-0 fixed, single model), RMSLE all rows | fold 1 0.1586; folds 2/3/4 0.1259 / 0.1322 / 0.1081; mean of 2–4 0.1221 (seed 2: 0.1233) | measured |
| Seed-to-seed noise of `base` | 0.0012 on the 3-fold mean | measured |
| LightGBM `cs02` (feature fraction 0.2 + bagging) | fold 1 0.1376; folds 2/3/4 0.1215 / 0.1129 / 0.1039; mean 0.1127 (2 seeds: 0.1127, 0.1128) | measured; **current champion** |
| Round-3 stacks vs `cs02` | −0.0015 to −0.0019 on the 3-fold mean, better on all 3 folds | awaiting seed 3 (jobs G, V1) |

Rules for baselines: the old committed CV numbers (flawed origins and early stopping) are never quoted. The earlier public score is not used for any decision and does not appear in any deliverable. The earlier "4-hour model" is the control run above.

## 2. Gate and decision rule (fixed now, applied mechanically)

1. Pass = mean gain ≥ 0.0024 (that is 2× the measured noise 0.0012, which also meets v3's ≥ 0.002 floor), on ≥ 2 of 3 development folds, and the sign holds in `rmsle_excl_open0`. Per-hub and per-horizon error is reported.
2. Seeds: a candidate must show the same sign on seeds 20260807, 08 and 09.
3. **Stack rule for round 3:** a stack passes only if it meets rule 1 against `cs02`. If it misses by less than the noise but is better on every fold and every seed, it becomes the **challenger**: it is carried into CatBoost and the blend check at no extra cost, and replaces `cs02` only if the 3-model blend gain is ≥ 0.0024 or it wins on fold 1 as well.
4. Final model choice is by CV and never by the public score.

## 3. Stages (all of v3's G0–G5, with status)

| Stage | Status 25 Nov | Next |
|---|---|---|
| G0 trust (noise floor, `no_te`, open-zero counts, fold 1) | done (noise 0.0012; fold 1 added) | season-matched fold 2014-06-19 built if time allows |
| G1 level and bias | done: `level`, `level_off`, `offset`, `tweedie_init`, `recency` lose or tie; `colsample` wins | none |
| G1b feature subsampling and regularisation | done: `cs02` champion; round 3 stacks pending | pick champion / challenger by section 2 |
| **B1 baseline ensemble** (new) | not started | Run `base`-recipe CatBoost, LightGBM, Tweedie (vp 1.1) with out-of-fold predictions saved on folds 1–4; log-mean blend. This is the true CV score of the old best |
| G4a members on the champion | not started | Same three models on the champion recipe, same folds, predictions saved; blend gain vs B1 |
| G2 structure (hurdle, month-ahead, season-filtered member, per-format, summer-2014 origins) | not started | only after G4a; needs small code |
| G3 features (ridge trend, streak counters, region proxy, dedupe, pruning, adversarial check) | not started | only after G4a |
| G4b extras (MLP, one Chronos-2 pilot, multi-seed, constrained blend weights) | not started | MLP on the idle GPU while CPU jobs run |
| G5 final | not started | **by 3 Dec:** retrain on all origins with fixed rounds, save models and SHA-256, sanity checks, record in `STATUS.md` |

Harness change needed first (small): save per-fold prediction files (`.npy` with row ids) so blends, per-hub and per-horizon checks run offline without refitting; add CatBoost and Tweedie model types.

## 4. Queue per server

| When | Server B (Manjot) | Server A (Vinayak) |
|---|---|---|
| now–26 Nov | round 3 G; then harness change; B1 baseline ensemble on folds 1–4 | round 3 V1/V2; then end-term fixed split + x* model |
| 26–28 Nov | G4a on champion (and challenger if any) | end-term sampling experiments (uniform, hub-stratified, sensitivity, column sketch): ≥ 5 sizes × 5 repeats |
| 28 Nov–2 Dec | G2/G3 one change at a time, only what is cheap; MLP on GPU | Plot 1–5 data; **exclusive timing runs on 1–2 Dec** (server B takes no job from A during them) |
| by 3 Dec | **G5 final retrain and save** | final sampling table, speed-up table |
| 3–7 Dec | idle | idle; slides, review, rehearsal on the Mac |

Full-use rule: every server always has a queued job; a stage that finishes starts the next without waiting for a message from the user. Jobs run under `nohup`, one row per run in `experiments.csv`, results copied to `endterm/results/` and into plan section 10 at each stage close.

## 5. End-term track (rubric first)
Fixed time-ordered split (last 42 days = fold 4 unless the team decides otherwise); x* = LightGBM plus ridge tuned on a training-only part; uniform vs hub-stratified vs sensitivity-sampling coreset vs column sketch; ≥ 5 sizes × 5 repeats; Plots 1–5; speed-up table from exclusive timing runs; conclusion; 6–9 slides, 7 minutes. The improvement work supplies the x* recipe (feature fraction 0.2 is the one proven change) and the "what we would do next" line. Decisions in force until the team says otherwise: test window = last 42 days; closed rows kept and RMSLE reported with and without open-zero rows; column sketch = random projection of the features for ridge; repo made public only after a name and server scan.

## 6. Ideas from the outside reviews
Unchanged from v3 section 8.4. Each stays in its stage (hurdle, month-ahead, streaks, MLP, Chronos pilot, multi-seed in G2–G4; peer-hub residuals, regime experts, reconciliation, shift adaptation on the waiting list; coresets from GBDT leaves as an end-term extra after the source is opened). Rejected items stay rejected: horizon-bucket offsets, "open-zero rows explain the gap", renormalised ridge weights.

## 7. What is still open
1. The CV-versus-public-score gap is unexplained; B1 will show whether the old ensemble scores differently from single LightGBM on the corrected folds.
2. Team confirmation of the four end-term defaults in section 5.
3. Citations from the reviews are unopened; open each before citing.
