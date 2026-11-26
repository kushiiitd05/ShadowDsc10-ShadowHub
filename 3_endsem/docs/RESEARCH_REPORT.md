# Research report: improving the 42-day hub-demand forecaster (concise)

Every number below comes from `results/` or `docs/FREEZE_AUDIT.md`. All scores are RMSLE on our own time-ordered development
folds. They are not comparable to any score outside this project, and the forecast window has no labels here.

## 1. Problem, data, split, metric
1,115 hubs, daily orders, 42 days ahead; rows = (origin, hub, day); 342 columns from history up to the origin. Rolling
development folds start on 2 Jan, 13 Feb, 27 Mar and 8 May 2015 (46,830 rows each); fold 5 starts on 19 Jun 2014 and is
season-matched to the forecast window (935 complete-block hubs in comparisons). Metric: RMSLE on all rows, plus RMSLE without
open-but-zero rows.

## 2. Starting point and final recipe
| Recipe | Mean RMSLE folds 1-4 |
|---|---|
| Earlier 3-model blend, re-scored on corrected folds | 0.1250 |
| `cs02` (feature fraction 0.2 + bagging), 1 seed / 3 seeds | 0.1190 / 0.1173 |
| `cs02`, lr 0.025, 1000 fixed rounds, 3 seeds (like-for-like baseline) | 0.1170 |
| **`hp17` (3 seeds) + TabPFN-TS (365/730 mix) at 0.2 (frozen)** | **0.1129** |

## 3. What worked, ranked by gain
1. **Feature fraction 0.2 + bagging** (0.1312 -> 0.1190). Many correlated rolling columns: sampling 20% per tree decorrelates the trees.
2. **Seed averaging in log space** (-0.0017). Variance reduction at no modelling cost.
3. **TabPFN-TS member at weight 0.2** (+0.0019 on top of `cs02` with the 365-day input; the long-context mix adds about +0.0013 more). The errors differ from the trees' because the member reads each hub's recent level directly.
4. **`hp17`** (255 leaves, strong L2, large min child): +0.0019 on folds 2-4 alone (misses the gate), +0.0013 inside the blend.

## 4. What did not work
More than 40 ideas, grouped by family in `results/experiment_ledger.md`, section B. The pattern: the error sits on event
days (top 10 of 168 days = 28% of squared error), so ideas aimed at ordinary days gain at most 0.002. The forecast window has no
regional holidays, so holiday features cannot help there.

## 5. The searches
20-config random search: `hp_017` is the only config better on all four folds (best of 19). 8 neighbours sit within 0.002 (a
plateau). 80 pooled Optuna trials: best 0.1105 vs `hp17` 0.1109, a third of the noise. The 5 finalists, re-run on all five folds, gain +0.0001 to
+0.0007 over `hp17` on folds 2-4. TabPFN-TS context/weight grid: flat
within 0.001-0.002. Conclusion: the recipe is saturated on these inputs.

## 6. Risks and limits
Selection bias (best of many on the same folds). Post-hoc context mix. Weight chosen on folds that include fold 4. Fold 4 is
also the end-term test window. Fold 1 (winter) is 0.0027 worse than `hp17` alone. Fold 5 covers 935 hubs. Timings come from a
shared server. The forecast window has no labels, so no real test score exists.

## 7. Recommendation and final retrain
Freeze `hp17` (3 seeds, lr 0.025, 1000 rounds) + TabPFN-TS (equal log mix of 365- and 730-day contexts) at weight 0.2. This
was done on 26 Nov 2026 with `final_retrain.py` on the real path. Saved boosters, SHA-256 manifest, exact command, code
hashes and hard checks are in `final/`. Fallbacks kept: `hp17` + TabPFN-TS 365 (rehearsal 1) and `cs02` (same setting).
No further model search.

## 8. End-term connection
The end-term x* is the `cs02` recipe (fixed before the improvement work finished). Its sensitivity coreset reaches 0.1070 at
50k rows against 0.1022 for the full model. The improvement work shows the model itself is saturated. So the remaining lever is
cost, which is the subject of the end-term: a 50k-row coreset trains in a third of the time for +0.0048 RMSLE.

## Five examiner questions
1. *Why is the gate 0.0024?* Twice the measured seed-to-seed noise (0.0012) of the same recipe. A gain below that cannot be told apart from reseeding.
2. *Is the gain real or selection bias?* Partly selection: `hp17`'s gain shrank on fresh seeds. The frozen blend still clears the gate on each of three seeds separately and on fold 5, which was not used to pick the weight.
3. *Why not use fold 4 as a test set?* It shaped earlier choices, so we call it a development fold and report it as such. Re-taking every decision without fold 4 changes none of them (`results/leakage_check_without_fold4.csv`).
4. *Why does fold 5 have larger error?* It has 9 open-but-zero rows. Without them the frozen model scores 0.1105, not 0.1737.
5. *Why accept the fold-1 loss against `hp17` alone?* The forecast window is summer with no holidays, and the season-matched fold 5 improves on every seed. The fold-1 loss is stated, not hidden.
