# Experiment ledger: every idea tried on the forecaster, with the measured result and the reason

All numbers are RMSLE on our own time-ordered development folds (fold 1 = 2015-01-02, fold 2 = 2015-02-13, fold 3 = 2015-03-27,
fold 4 = 2015-05-08, 46,830 rows each; fold 5 = 2014-06-19, season-matched to the forecast window). "Gain" = baseline minus
candidate (positive = better). Gate: mean gain over folds 2-4 >= 0.0024 (2 x measured noise 0.0012), >= 2 of 3 folds improve,
same sign on seeds 20260807/08/09, holds without open-but-zero rows, fold 5 checked. A "challenger" is better on most folds but
misses the gate; it is not adopted.

## A. Baselines and the path to the frozen model

| Step | RMSLE (mean folds 1-4) | Note |
|---|---|---|
| Earlier 3-model blend (CatBoost + LightGBM + Tweedie), re-scored on corrected folds | 0.1250 | corrected folds: no early stopping on the evaluated fold, cross-fitted hub encoding |
| LightGBM `base` (feature fraction 0.8, no bagging) | 0.1312 | single model |
| `cs02` = feature fraction 0.2 + row bagging 0.85 every iteration | 0.1190 (seeds 0.1190 / 0.1186 / 0.1191) | **adopted** (-0.0094 on folds 2-4, -0.021 on fold 1) |
| `cs02`, 3 seeds averaged in log space | 0.1173 | **adopted** (free variance reduction) |
| `cs02`, lr 0.025, 1000 fixed rounds, 3 seeds | 0.1170 | the like-for-like baseline of the final gate |
| `hp17` (feature fraction 0.15, 255 leaves, min child 160, L2 24.1), same setting, 3 seeds | 0.1141 | challenger: folds 2-4 gain 0.0019 < 0.0024 |
| `hp17` + TabPFN-TS (equal log mix of 365- and 730-day contexts), weight 0.2 | 0.1129 | **frozen**: folds 2-4 gain 0.0045, every seed >= 0.0046, all gate conditions pass |

## B. Ideas that did not pass, grouped by family

| Family | Idea | Measured | Verdict | Why (our reading of the evidence) |
|---|---|---|---|---|
| Target / level | predict relative to recent hub level; level + offset; one global log offset; recency weights (half-life 180/365 d) | lose or tie on folds 2-4 | refuted | the rolling medians already give the level; one offset cannot fix hub-specific bias, and recency weights throw away seasonal information |
| Target | Tweedie objective with init score; production Tweedie | worse than log1p target | refuted | RMSLE is a log-scale loss, so fitting the log directly matches the metric |
| Models | CatBoost; 3-model blend on the champion recipe | blend 0.1178 vs seed-averaged LightGBM 0.1173 | no gain | members are highly correlated with LightGBM; seed averaging already removes most variance |
| Blending | NNLS blend weights per fold | 0.1192 vs equal weights 0.1184 | refuted | weights fitted on few folds over-fit |
| Post-hoc | calibration a + b*y_hat | worse | refuted | the slope is unstable across seasons |
| Post-hoc | horizon-bucket offsets | lose to one global offset | refuted | the bias does not depend on the horizon in a stable way |
| Structure | separate models per horizon block; long-window-only features | 0.1190 / 0.1189 | null | smaller training set per model cancels any gain |
| Structure | per-HubFormat models | worse | refuted | splitting the data costs more than format-specific fits gain |
| Structure | per-AssortmentTier member | `cs02` pair 0.1164 vs 0.1173; on `hp17` worse than plain | not adopted | helps the small 63-leaf trees, hurts the 255-leaf trees that already separate tiers |
| Features (G3, G3b) | ridge trend, streak counters, weekday x promo, region proxy, gap-aware lags, year-ago, peer aggregates | fold 4 -0.002..-0.005 on all seeds; fold 3 flips sign; fold 2 worse | not adopted | first lead did not replicate on new seeds; the signal is already in existing rolling columns |
| Features (G3c) | event-day features (6 groups) | worse or equal on folds 3, 4, 5 | not adopted | the forecast window has no regional holidays, so holiday features cannot help there |
| Neural | entity-embedding MLP | alone 0.2454 / 0.1825 / 0.1200 / 0.1095; blend +<=0.003 on folds 3-4, -0.011 on fold 1 | not adopted | weak alone; blend gain below the gate and negative in winter |
| Foundation | Chronos-2 zero-shot with covariates | alone 0.1743; best blend +0.0019 on one fold | not adopted | below gate |
| Neural | TiDE; NHITS | blend -0.0008; +0.0003 | not adopted | no gain |
| Foundation | TabPFN-TS, 365-day context, on `cs02` | folds 2-4 +0.0018 at w 0.3; fold 1 -0.0120 | challenger | positive in summer, negative in winter; became useful only with `hp17` and the long-context mix |
| Origins | summer-2014 incomplete origins | +0.0013 | not adopted | noise |
| Origins | season-filtered member | +0.0017..+0.0043 folds 1-3, fails folds 2-4 and fold 5 | not adopted | part of the gain is plain two-model averaging |
| Origins | dense 7-day origins | worse on folds 2-5; 2x memory and time | not adopted | heavily overlapping label windows add correlated rows, not information |
| Denoising | SVD of the hub x day matrix, rank 14 | fold 3 -0.0054 | not adopted | removes real hub-specific events |
| Structure | two-stage (weekly total, then day share) | -0.0005 / -0.0016 (full); +0.0010 / +0.0002 (minimal) | not adopted | below noise |
| Residuals | peer-hub residual correction | best +0.0027 / +0.0015 / +0.0007 | challenger | 1 of 3 folds above gate |
| Adaptation | shift adaptation (14-day window, beta ~ 0.3) | +0.0016 / +0.0011 / +0.0013 on all folds and seeds | weak challenger | consistent but below gate; needs an extra forecast from the previous origin |
| Reconciliation | hierarchical (hub vs total) | fitted weights 0 | no gain | the total is already consistent |
| Experts | regime experts (Sunday/holiday vs ordinary days) | -0.0066..+0.0005 | not adopted | open Sunday/holiday rows are only 1.7-3% of squared error |
| Diagnostics | adversarial validation (train vs fold) | AUC 1.000 for every target incl. the control | uninformative | date-like columns separate any two periods |
| Pruning | drop 81 low-importance columns | -0.0063 / -0.0051 / +0.0028 / +0.0013, fold 5 -0.0065 | not adopted | low-importance columns still matter on some folds |
| Pruning | drop 6 flag columns | ~0 | not adopted | no effect |
| Hurdle | zero / non-zero first stage | <= 1 open-zero row per development fold | dropped | nothing to model |

## C. Searches on the LightGBM recipe (selection bias applies to every "best")

| Search | Size | Result |
|---|---|---|
| Random search around `cs02` (lr 0.05, early stopping, seed 07, folds 1-4) | 20 configs | `hp_017` only config better on all 4 folds: 0.1149 vs 0.1190; best of 19 |
| `hp17` on fresh seeds | 2 seeds | gains +0.0030, +0.0024 (seed 07 had +0.0041): selection effect visible |
| Neighbours of `hp17` (L2 10/50, 127/511 leaves, min child 320, feature fraction 0.1/0.2, fixed rounds) | 8 variants | all within 0.002 of `hp17`: a plateau |
| Pooled Optuna (TPE, sampler seeds 20260807/08, 80 trials, objective = folds 2-3 blend RMSLE at w 0.2) | 80 trials | best 0.1105 vs `hp17` 0.1109 vs `cs02` 0.1123; top 10 span 0.0005; no replacement |
| TabPFN-TS context length | 365 / 540 / 730 days and mixes | 730 fixes winter fold 1, loses fold 5; equal 365/730 mix is the only input without a loss on folds 1 and 5 |
| Blend weight | 0.1 .. 0.5 | flat within 0.001-0.002; 0.3 adds 0.0010 over 0.2 on folds 2-4 (below noise) and lowers fold 5 |
| Optuna finalists on all five folds (seed 07, lr 0.025, 1000 fixed rounds) | 5 trials | vs `hp17` on the same seed: folds 2-4 +0.0001 .. +0.0007, fold 1 -0.0057 .. +0.0013, fold 5 -0.0028 .. +0.0006; none near the gate, `hp17` kept (`g6c_finalists_5folds.csv`) |

Total recipe variants scored on the same folds: about 125 (20 + 8 + 80 + 5 finalists + contexts + weights + about 40 ideas above).

## D. Error analysis of `cs02`
The 10 worst of 168 forecast days carry 28% of the squared error; they are event days (holidays, Easter, Sundays, promotion
and school-closure days). Holiday days score 0.267 against 0.133 on ordinary days. The forecast window has no regional holidays.

## E. Leakage check: every decision re-taken without fold 4
Fold 4 (8 May 2015 origin) is also the end-term test window. `code/leakage_check.py` re-takes each decision that used fold-4
scores with fold 4 removed (`leakage_check_without_fold4.csv`). None changes: `cs02` over `base` (+0.0119 on folds 2-3, fold 1
+0.0210); `hp17` alone still fails (seed 08 loses fold 2); the frozen blend still passes (+0.0050 on folds 2-3, every seed
positive, fold 5 positive); weight 0.2 kept (best rival 0.3, +0.0006, below noise); the 365/730 mix kept (730 alone +0.0010,
below noise). Rounds, ridge lambda, the end-term learning rate, sensitivities and column choice never used fold 4.
