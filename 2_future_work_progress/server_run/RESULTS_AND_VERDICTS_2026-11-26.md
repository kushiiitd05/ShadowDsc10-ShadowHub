# Results and verdicts: every experiment run (26 Nov 2026, 01:30)

Task: forecast daily orders for 1,115 hubs, 42 days ahead. Metric: RMSLE (RMSE of log1p). All numbers are on our own time-ordered dev folds, not comparable to any outside leaderboard.
Dev folds (forecast origin): fold 1 = 2015-01-02, fold 2 = 2015-02-13, fold 3 = 2015-03-27, fold 4 = 2015-05-08 (46,830 rows each); fold 5 = 2014-06-19, season-matched to the real test window that starts 2015-06-20 (41,250 rows; 180 "gap" hubs lack data 2014-07-01..12-31). Real test window has no regional holidays; fold 5 has none either.
Gate for adoption (fixed in advance): mean gain over `cs02` ≥ 0.0024 (2× the measured noise 0.0012), on at least 2 of the 3 dev folds 2–4, same sign on seeds 20260807/08/09, still true excluding open-but-zero rows, and fold 5 checked. A "challenger" is better on most folds but misses the gate. Final choice is by CV only. Full detail per experiment: `ENDTERM_AND_IMPROVEMENT_PLAN_2026-11-25.md` sections 10–33.

## 1. Baselines and the current recipe (RMSLE, mean of folds 1–4 unless stated)

| Model | RMSLE |
|---|---|
| Old recipe, 3-model blend (CatBoost + LightGBM + Tweedie), re-scored on corrected folds | 0.1250 |
| LightGBM `base` (feature fraction 0.8) | 0.1312 |
| `cs02` (feature fraction 0.2, bagging 0.85, 63 leaves, min child 80, log1p target, hub target-encoding cross-fitted), one seed | 0.1190 (seeds 0.1190 / 0.1186 / 0.1191) |
| `cs02`, 3 seeds averaged in log space | 0.1173 |
| `cs02`, lr 0.025, 1000 fixed rounds (final-retrain setting) | 0.1174 (1 seed) |
| `cs02` + per-tier member, 3-seed pair vs control | 0.1164 vs 0.1173 |
| Final-retrain dry run on fold 4 (origins before 2015-05-08, 1 seed) | 0.10172; with per-tier member 0.10132 |

## 2. 20-config random search around `cs02` (seed 20260807, lr 0.05, early stopping, folds 1–4)
Gain = hp_000 minus config (positive = better). `hp_000` is `cs02` itself and reproduces 0.1378 / 0.1215 / 0.1129 / 0.1039.

| config | colsample | leaves | min_child | lambda | extra_trees | fold 1 | fold 2 | fold 3 | fold 4 | 4-fold mean | gain (mean) | gain folds 2–4 | folds 2–4 improving |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **hp_017** | 0.15 | 255 | 160 | 24.1 | no | 0.1288 | 0.1202 | 0.1078 | 0.1027 | **0.1149** | **+0.0041** | **+0.0025** | **3/3** |
| hp_006 | 0.1 | 63 | 320 | 22.67 | no | 0.1318 | 0.1183 | 0.1151 | 0.1041 | 0.1174 | +0.0017 | +0.0002 | 1/3 |
| hp_014 | 0.15 | 255 | 160 | 2.86 | yes | 0.1310 | 0.1196 | 0.1148 | 0.1043 | 0.1175 | +0.0016 | −0.0002 | 1/3 |
| hp_008 | 0.1 | 255 | 40 | 0.71 | no | 0.1350 | 0.1181 | 0.1133 | 0.1047 | 0.1178 | +0.0013 | +0.0007 | 1/3 |
| hp_012 | 0.3 | 255 | 80 | 1.63 | no | 0.1323 | 0.1198 | 0.1154 | 0.1042 | 0.1179 | +0.0011 | −0.0004 | 1/3 |
| hp_007 | 0.2 | 63 | 80 | 9.98 | no | 0.1386 | 0.1210 | 0.1103 | 0.1023 | 0.1181 | +0.0010 | +0.0015 | 3/3 |
| hp_011 | 0.3 | 255 | 40 | 2.74 | no | 0.1338 | 0.1197 | 0.1140 | 0.1055 | 0.1183 | +0.0007 | −0.0003 | 1/3 |
| hp_003 | 0.3 | 255 | 40 | 7.39 | no | 0.1346 | 0.1208 | 0.1134 | 0.1049 | 0.1184 | +0.0006 | −0.0003 | 1/3 |
| hp_000 (= `cs02`) | 0.2 | 63 | 80 | 1.0 | no | 0.1378 | 0.1215 | 0.1129 | 0.1039 | 0.1190 | 0 | 0 | 0/3 |
| hp_010 | 0.4 | 255 | 80 | 1.21 | no | 0.1340 | 0.1199 | 0.1180 | 0.1045 | 0.1191 | −0.0001 | −0.0014 | 1/3 |
| hp_002 | 0.3 | 255 | 80 | 0.46 | yes | 0.1303 | 0.1189 | 0.1216 | 0.1066 | 0.1194 | −0.0003 | −0.0029 | 1/3 |
| hp_018 | 0.3 | 127 | 40 | 1.76 | no | 0.1389 | 0.1218 | 0.1177 | 0.1053 | 0.1209 | −0.0019 | −0.0022 | 0/3 |
| hp_019 | 0.4 | 63 | 80 | 2.13 | yes | 0.1314 | 0.1219 | 0.1270 | 0.1046 | 0.1212 | −0.0022 | −0.0051 | 0/3 |
| hp_015 | 0.1 | 127 | 320 | 5.91 | yes | 0.1325 | 0.1178 | 0.1246 | 0.1128 | 0.1219 | −0.0029 | −0.0057 | 1/3 |
| hp_001 | 0.15 | 31 | 160 | 0.52 | no | 0.1461 | 0.1231 | 0.1132 | 0.1070 | 0.1224 | −0.0033 | −0.0017 | 0/3 |
| hp_009 | 0.2 | 31 | 160 | 18.55 | no | 0.1407 | 0.1256 | 0.1152 | 0.1095 | 0.1227 | −0.0037 | −0.0040 | 0/3 |
| hp_013 | 0.2 | 31 | 320 | 6.38 | yes | 0.1336 | 0.1218 | 0.1264 | 0.1099 | 0.1229 | −0.0039 | −0.0066 | 0/3 |
| hp_016 | 0.15 | 31 | 80 | 3.6 | no | 0.1476 | 0.1252 | 0.1154 | 0.1056 | 0.1234 | −0.0044 | −0.0027 | 0/3 |
| hp_004 | 0.15 | 31 | 160 | 0.35 | yes | 0.1388 | 0.1242 | 0.1265 | 0.1071 | 0.1242 | −0.0052 | −0.0066 | 0/3 |
| hp_005 | 0.4 | 31 | 320 | 7.08 | no | 0.1443 | 0.1256 | 0.1253 | 0.1055 | 0.1252 | −0.0062 | −0.0060 | 0/3 |

Reading: **hp_017 is the only config that improves all four folds** (−0.0090, −0.0013, −0.0051, −0.0012) with a 4-fold mean gain of 0.0041 and a folds 2–4 mean gain of 0.0025 (gate 0.0024). Pattern: large trees (255 leaves) with strong L2 (lambda ≈ 24) and a large min child (160) win; 31-leaf trees lose everywhere. It was picked as the best of 19 alternatives on the same folds, so the gain is optimistic until it is confirmed on seeds 08 and 09 and on fold 5 (**running**, see section 5). Verdict: **candidate, awaiting confirmation**.

## 3. Verdicts of everything else (gain over `cs02`; positive = better)

| Item | Result | Verdict |
|---|---|---|
| Level target, level + offset, global offset, recency weights | lose or tie | refuted |
| Tweedie with init score; production Tweedie; CatBoost; blends of the three | 3-model blend 0.1178 vs seed-averaged LightGBM 0.1173 | no gain |
| Early stopping vs fixed rounds | lr 0.05: 300 rounds 0.1190, 600 0.1195, 1000 0.1198; lr 0.025: 600 0.1176, 1000 0.1174 | fixed rounds fine; lr 0.025 / 1000 chosen |
| Pooled NNLS blend weights | 0.1192 vs equal 0.1184 | refuted |
| Calibration `a + b·ŷ` | worse | refuted |
| G2 horizon split / long-window-only | 0.1190 / 0.1189 | null |
| Per-HubFormat models | worse | refuted |
| Per-AssortmentTier models | 0.1178 alone; with `cs02` 3-seed pair 0.1164 vs 0.1173 | optional blend member |
| G3 / G3b features (ridge trend, streaks, weekday×promo, region proxy, gap-aware, year-ago, peer aggregates) | fold 4 −0.002..−0.005 on all seeds; fold 3 flips sign; fold 2 worse; fold 1 breaks on gap hubs | not adopted (replication failed) |
| G3c event-day features (6 groups) | every group worse or equal on folds 3, 4, 5 | not adopted (and holiday features cannot help: test window has none) |
| MLP (entity embedding) | alone 0.2454 / 0.1825 / 0.1200 / 0.1095; blend w 0.2 gains on folds 3–4 ≤ 0.003, loses 0.011 on fold 1 | not adopted |
| Chronos-2 zero-shot (with covariates) | alone mean 0.1743; best blend +0.0019 (one fold) | not adopted |
| TiDE | blend best mean −0.0008 | not adopted |
| NHITS | blend best mean +0.0003 | not adopted |
| **TabPFN-TS** (TabPFN v2 backbone, 365-day context) | see section 4 | **challenger** |
| Summer-2014 incomplete origins | +0.0013 mean (noise) | not adopted |
| Season-filtered member blended | +0.0017..+0.0043 on folds 1–3, fails folds 2–4 and fold 5; part is plain two-seed averaging | not adopted |
| Dense 7-day origins | worse on folds 2–5, doubles memory and time | not adopted |
| SVD denoising, rank 14 | fold 3 −0.0054 | not adopted |
| Two-stage (weekly total → day) | −0.0005 / −0.0016 (full), +0.0010 / +0.0002 (minimal) | not adopted |
| Peer-hub residuals | best +0.0027 / +0.0015 / +0.0007 (1 of 3 folds) | challenger, not adopted |
| Shift adaptation (14-day window, β≈0.3) | +0.0016 / +0.0011 / +0.0013, all folds and seeds, below gate; needs extra previous-origin forecast | weak challenger, not adopted |
| Hierarchical reconciliation | fitted weights 0 | no gain |
| Regime experts (Sunday/holiday vs ordinary) | −0.0066..+0.0005; open Sunday/holiday rows are 1.7–3% of squared error | not adopted |
| Adversarial validation | AUC 1.000 for every target including control | uninformative; no drift flagged |
| Permutation-importance pruning | drop 81 low-importance columns: −0.0063 / −0.0051 / +0.0028 / +0.0013 (fold 5 −0.0065); drop 6 flags: ≈ 0 | not adopted |
| Hurdle model | ≤ 1 open-zero row per dev fold | dropped |

Error analysis of `cs02`: errors concentrate on event days (holidays, Easter, Sundays, promo/school closure); the top 10 of 168 days carry 28% of squared error; holiday days RMSLE 0.267 vs 0.133. The real test window has no regional holidays.

## 4. TabPFN-TS blend in detail (gain over the 3-seed `cs02` average; positive = better; weight w on TabPFN-TS in log space)

| fold | `cs02` avg3 | TabPFN-TS alone | w 0.1 | w 0.2 | w 0.3 |
|---|---|---|---|---|---|
| 1 | 0.1340 | 0.2143 | −0.0024 | −0.0065 | −0.0120 |
| 2 | 0.1194 | 0.1321 | +0.0014 | +0.0023 | +0.0025 |
| 3 | 0.1124 | 0.1336 | +0.0006 | +0.0006 | −0.0001 |
| 4 | 0.1035 | 0.1125 | +0.0015 | +0.0025 | +0.0029 |
| 5 (935 hubs with a full 42-day block) | 0.1784 | 0.1778 | +0.0020 | +0.0036 | +0.0048 |

Folds 2–4 mean gain at w 0.3: +0.0018 (gate 0.0024): **misses the gate**, but it is positive on folds 2, 4 and 5 (fold 5, the summer-matched fold, shows the largest gain) and clearly negative on fold 1 (winter origin; the loss is worse on non-gap hubs, so it is not a gap-hub artifact). Single seed (the model is close to deterministic). Status: **challenger**, to be tested on top of the winning `cs02` variant before the freeze.

## 5. Still running at time of writing
Confirmation of hp_017: seeds 20260808 and 20260809 on folds 1–4; seeds 07/08/09 on fold 5; neighbours of hp_017 (lambda 10/50, 511 leaves, min child 320, colsample 0.1/0.2, 127 leaves, fixed 1000 rounds). Then: combine winners, confirm on 3 seeds + fold 5, pick recipe by CV by 2 Dec, freeze 3 Dec (retrain script `improve/g8_final/final_retrain.py`, SHA-256 manifest, sanity checks).

## 6. End-term track (separate from the model gate)
Fixed split (train = earlier part, test = last 42 days = fold 4), same model/settings on full and every summary, 5 repetitions per size, medians. Full-data LightGBM RMSLE 0.1022 (R² 0.9992, fit 60 s); ridge 0.1592 (4.3 s). Sensitivity-sampling coreset beats uniform at mid sizes (25k 0.1108 vs 0.1133; 50k 0.1070 vs 0.1102; 100k 0.1058 vs 0.1099), ties at 2k and 200k; hub-stratified ≈ uniform; sensitivity build ≈ 10 s; column-importance subset k = 40: 0.1102 vs uniform 0.1770 (rows fixed at 200k). The ridge coreset build (18.6 s) costs more than a full ridge fit, so it saves no time there. Timings must be re-run exclusively on an idle server on 1–2 Dec. Due 8 Dec; presentations 9–10 Dec.
