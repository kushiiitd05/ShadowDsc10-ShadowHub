# Freeze audit of the forecaster (26 Nov 2026)

**Candidate.** Three-seed LightGBM `hp17` (80%) + equal log-space average of TabPFN-TS 365-day and 730-day contexts (20%).
For hub h and day t, with l(x) = log1p(x):

    L = 0.8 * (1/3) * sum_{s in 20260807, 20260808, 20260809} l(yhat_hp17_s) + 0.2 * (1/2) * ( l(yhat_TabPFN_365) + l(yhat_TabPFN_730) )
    F(h,t) = max(0, expm1(L)) if the hub is open on t, else 0

`hp17` = LightGBM, feature fraction 0.15, bagging fraction 0.85 every iteration, 255 leaves, min child 160, L2 24.1, log1p
target, cross-fitted hub target encoding, learning rate 0.025, 1000 fixed rounds, trained on the 37 origins whose label windows
end on or before 2015-06-19.

**Method.** Every statement below was checked in the artifacts, not in the markdown: the saved out-of-fold prediction files,
the TabPFN-TS prediction files and manifests, the saved boosters, the rehearsal outputs and the final retrain outputs.
Script: `code/freeze_audit.py` (output `results/audit_summary.json`, `results/audit_gate_gains.csv`,
`results/audit_fold_scores.csv`). No new search, weight grid or model-selection experiment was run.

## 1. Seven-row audit

| # | Check | Result | Evidence (checked in artifacts) |
|---|---|---|---|
| 1 | Formula = `final_retrain.py` | **PASS** | `blend_external`: `expm1((1-w)*log1p(base) + w*log1p(external))`, closed rows 0, clip at 0; `base` = `expm1(mean log1p(seeds))`, so the blend equals the formula exactly. `final_config("hp17")` refuses to run unless the config equals the registered `hp17_fix1000` variant. The rehearsal-2 forecast rebuilt from its saved per-seed predictions and the mix file differs from the saved file by max abs log difference **0.0**. |
| 2 | Leakage | **PASS** | Training origins: `target_end <= forecast origin` (37 origins, last label 2015-06-19, first target 2015-06-20). TabPFN-TS: `history_last_date` = `forecast_origin` = 2015-06-19 in both test-window manifests, enforced by `assert_history_before_window`. Hub encoding: `TargetEncoder.fit_transform` (internal cross-fitting) on training rows; the forecast frame only gets `transform`. No labels exist for the forecast window, so nothing in the blend can derive from them. Minor note: origins are two weeks apart with 42-day label windows, so a day's label appears in up to 3 origins and the random cross-fitting folds do not separate those copies. This affects only how strongly the model trusts the encoding column during training, not evaluation; the development folds were scored with exactly the same scheme. |
| 3 | Gate from saved OOF | **PASS** | Table in section 2. Candidate: folds 2-4 mean gain **0.0045** (3-seed), **0.0047 / 0.0047 / 0.0046** per seed (all >= 0.0024); 3 of 3 folds improve on every seed; without open-zero rows **0.0046**; fold 5 positive on every seed (+0.0035 / +0.0075 / +0.0067). Matches the markdown (0.0045; per seed 0.0045-0.0047) within 0.0001 rounding. |
| 4 | 365/730 mix | **PASS** | Rebuilt in log1p space from the two files: max abs log difference **0.0** against the pre-averaged file on 187,320 development rows (folds 1-4) and 46,830 forecast rows. Ids identical and in the same order in the 365, 730 and mix files. SHA-256: mix `6d5eb125...cab3` = the hash in the rehearsal-2 manifest; 365 `3c2f6c40...058e` and 730 `460f2225...b29fb6` equal their own manifests. Both contexts: forecast origin and history end 2015-06-19. |
| 5 | Implementation of `hp17` and bagging | **PASS** | All three saved boosters print `feature_fraction 0.15`, `bagging_fraction 0.85`, `bagging_freq 1`, `num_leaves 255`, `min_data_in_leaf 160`, `lambda_l2 24.1`, `learning_rate 0.025`, `num_iterations 1000`, with 1000 trees each. |
| 6 | Production-path retrain | **CONDITION** | Real path, own folder, nothing overwritten; all 13 sanity checks and all local hard checks pass (section 4). Condition: a second run of the identical recipe is **not bit-identical** (forecast RMS log difference 0.0017, max 0.0097, correlation 0.99999); the frozen artifact is the saved files, not a re-run. |
| 7 | Wording | **PASS** | The report, README and notebook contain no "state of the art" claim, no outside score, ranking or platform name, and call folds 1, 4 and 5 confirmation diagnostics. They state that the forecast window has no labels in this project and no real test score exists. (Scan command in `docs/REPRODUCIBILITY.md`.) |

## 2. Recalculated gate table (gain = like-for-like `cs02` minus candidate; positive = better)

Baseline: `cs02`, lr 0.025, 1000 fixed rounds, same seed. Fold 5 = the 935 hubs with a full 42-day block (39,270 rows); folds 1-4
have 46,830 rows each.

| arm | candidate | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | mean folds 2-4 | folds improved 2-4 | mean 2-4 excl. open-0 | fold 5 excl. open-0 |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 07 | `hp17` alone | +0.0047 | +0.0013 | +0.0032 | +0.0015 | +0.0014 | 0.0020 | 3 | 0.0021 | +0.0022 |
| seed 07 | **frozen candidate** | +0.0024 | +0.0055 | +0.0044 | +0.0043 | +0.0035 | **0.0047** | 3 | 0.0049 | +0.0055 |
| seed 07 | fallback `hp17` + TabPFN 365 | -0.0032 | +0.0046 | +0.0024 | +0.0035 | +0.0042 | 0.0035 | 3 | 0.0036 | +0.0065 |
| seed 08 | `hp17` alone | +0.0063 | -0.0003 | +0.0051 | +0.0006 | +0.0054 | 0.0018 | 2 | 0.0019 | +0.0084 |
| seed 08 | **frozen candidate** | +0.0038 | +0.0043 | +0.0062 | +0.0034 | +0.0075 | **0.0047** | 3 | 0.0049 | +0.0118 |
| seed 08 | fallback `hp17` + TabPFN 365 | -0.0016 | +0.0035 | +0.0042 | +0.0027 | +0.0083 | 0.0034 | 3 | 0.0036 | +0.0128 |
| seed 09 | `hp17` alone | +0.0059 | +0.0002 | +0.0051 | -0.0002 | +0.0047 | 0.0017 | 2 | 0.0017 | +0.0073 |
| seed 09 | **frozen candidate** | +0.0039 | +0.0040 | +0.0065 | +0.0032 | +0.0067 | **0.0046** | 3 | 0.0047 | +0.0105 |
| seed 09 | fallback `hp17` + TabPFN 365 | -0.0014 | +0.0029 | +0.0046 | +0.0024 | +0.0071 | 0.0033 | 3 | 0.0034 | +0.0110 |
| 3-seed avg | `hp17` alone | +0.0058 | +0.0005 | +0.0045 | +0.0007 | +0.0038 | 0.0019 | 3 | 0.0019 | +0.0060 |
| 3-seed avg | **frozen candidate** | +0.0031 | +0.0045 | +0.0054 | +0.0035 | +0.0057 | **0.0045** | 3 | 0.0046 | +0.0089 |
| 3-seed avg | fallback `hp17` + TabPFN 365 | -0.0024 | +0.0035 | +0.0035 | +0.0027 | +0.0063 | 0.0032 | 3 | 0.0033 | +0.0098 |

Absolute RMSLE, 3-seed average (all rows): `cs02` 0.1334 / 0.1186 / 0.1128 / 0.1034 / 0.1794; `hp17` 0.1275 / 0.1180 /
0.1083 / 0.1026 / 0.1756; frozen candidate 0.1302 / 0.1141 / 0.1073 / 0.0998 / 0.1737 (folds 1 / 2 / 3 / 4 / 5). Without
open-zero rows the candidate scores 0.1231 / 0.1065 / 0.1073 / 0.0909 / 0.1105. Open-zero rows per fold: 1 / 1 / 0 / 1 / 9.

Gate verdicts: **candidate PASSES** every condition; fallback `hp17` + TabPFN 365 also passes; `hp17` alone **FAILS**
(folds 2-4 mean 0.0019 < 0.0024 and 2 of 3 folds on seeds 08 and 09).

## 3. Experiment-selection limits (stated, not removable)
- `hp17` was the best of 19 random-search configs on the same folds; its seed-07 gain (+0.0041) shrank on fresh seeds (+0.0030, +0.0024).
- The 365/730 mix was defined after both results were seen (one post-hoc variant).
- The weight 0.2 was the middle of {0.1, 0.2, 0.3} on folds 2-4, which include fold 4; the curve is flat (0.0031-0.0034 on the 365 input).
- TabPFN-TS was configured on a fold-4, 100-hub smoke run (a small leak of fold 4 into the configuration).
- Fold 4 shaped earlier recipe choices and is also the end-term test window. Fold 5 covers a different population (935 of 1,115 hubs, 9 open-zero rows).
- Experiments that fed the choice: 20 random-search configs, 8 neighbours, 80 Optuna trials, three TabPFN-TS context lengths (+ mixes), a weight grid, one post-hoc mix, and about 40 ideas in `results/experiment_ledger.md`.
- **Folds 1, 4 and 5 are confirmation diagnostics, not untouched test sets. There are no labels for the real forecast window in this project; no real test score exists or is reported.**
- Fold 1 (winter, year-end): the candidate is better than `cs02` (+0.0031) but 0.0027 worse than `hp17` alone. The forecast window is summer, has no regional holidays, and fold 5 (season-matched) improves on every seed. That is why we accept this. Whether TabPFN-TS generally hurts winter and year-end is only partly supported: the 365-day input loses on fold 1, the 730-day and mixed inputs do not.

## 4. Production-path retrain and final checks
Command (run from `code/model` on the server; full record with code hashes in `final/FREEZE_COMMAND.txt`):

    python g8_final/final_retrain.py <proj_root> <run_root>/final_freeze_2026-11-26 --recipe hp17 \
        --blend-file ../blend_in/test_pred_tabpfnts_mix.csv --blend-weight 0.2 --seeds 20260807,20260808,20260809 --n-jobs 16

Started 2026-11-26 11:35 IST, finished 12:01 (server load average 25-40 from other users: loading 597 s, fits 108 / 240 / 107 s).
Code SHA-256: `final_retrain.py` 886cf55b..., `exp_runner.py` cf8d4f77..., `helpers.py` 2c15ab38... (identical to `code/model/`).

| Item | Value |
|---|---|
| Training | 37 origins, labels end 2015-06-19, 1,434,050 open positive rows, 342 columns |
| Booster SHA-256 (seed 07 / 08 / 09) | `613aa09dedaa65b7...a816` / `d0a4af424ff9f300...dc4a` / `fa798ad0c64b95eb...6ba7` (files in `final/models/*.txt.gz`; hashes are of the unzipped text) |
| Blend file | `test_pred_tabpfnts_mix.csv`, SHA-256 `6d5eb125...cab3`, weight 0.2, log1p space |
| Forecast file | `final/submission_final.csv`, SHA-256 `7d44a05089c6174d20afb07fb6a72414b540794f6ca86e2ae94ed483c0fc4484` |
| Hard checks (recomputed locally, `final/freeze_checks.json`) | 46,830 rows; Id order equals the sample file and the forecast frame; 0 NaN/inf; 0 negative; 6,548 closed rows exactly 0; 0 open rows at 0; formula rebuilt from the saved member predictions with max abs log difference 0.0; SHA-256 matches the manifest |
| Server sanity checks (`final/sanity_report.json`) | 13 / 13 pass; mean log1p difference to the earlier control forecast +0.0036, correlation 0.9970 (plumbing check only, not accuracy) |
| Versions | Python 3.11.18, LightGBM 4.7.0, numpy 1.25.2, pandas 2.3.3, scikit-learn 1.7.2 |

**Run-to-run check against rehearsal 2** (same recipe, same inputs, same seeds; `results/audit_summary.json`): seed 08
predictions are identical; seeds 07 and 09 differ (RMS log difference 0.0056 and 0.0026; max 0.034 and 0.019); the blended
forecast differs by RMS 0.0017 in log space (max 0.0097, mean +0.0002, correlation 0.99999). The booster files differ in hash
even where predictions are identical. Likely cause (not verified): LightGBM is not run with `deterministic=True`; with 16
threads on a loaded server it chooses between row-wise and column-wise histogram building by timing, and floating-point sums
in a different order can change a split. Scale: two *different* seeds of the same recipe differ by RMS 0.019-0.020 in log space on the same rows, so the run-to-run
difference (0.0017 for the blended forecast) is about a tenth of the seed-to-seed difference. Because
|RMSLE(a) - RMSLE(b)| <= RMS(log1p a - log1p b), the forecast's score can move by at most 0.0017 between the two runs, and
in practice by much less, since the difference is unrelated to the error. It does not change any conclusion. It does mean that
"reproducible" here means *the saved files reproduce the forecast exactly*; a re-run reproduces it only approximately.

Fallbacks kept, not overwritten: rehearsal 2 (same recipe, forecast `0d9dd419...e962`), rehearsal 1 (`hp17` + TabPFN-TS 365,
`0037034b...6465`), and `cs02` (same setting) via `--recipe cs02`.

## 5. Unresolved issues
1. **Bit-level retrain reproducibility.** Not achieved (section 4). The fix for any future run is `deterministic=True,
   force_col_wise=True` in the LightGBM parameters. We did **not** apply it now, because it would change the implementation after
   the audit; the frozen artifact is the saved boosters and forecast file.
2. **Command inside the manifest.** `final_manifest.json` has versions but no command field; the exact command is recorded in
   the sidecar `final/FREEZE_COMMAND.txt` together with the code hashes.
3. **Hub target encoding and overlapping origins** (audit row 2): cross-fitting splits rows at random, and a day's label can
   appear in up to three origins. This is a training-side effect only and was identical in every development fold; not changed.
4. **Winter fold 1** is 0.0027 worse than `hp17` alone; unexplained, accepted for a summer forecast window.
5. **End-term timings** come from a shared server; the exclusive rerun (fixed cores, 16 threads, load logged) is still open.
6. **Coreset weight-normalisation check** (one size with unnormalised weights) was planned and not run.
7. **Member contributions** in the report header: filled for one member (26 Nov); the other four are still to be confirmed by each member.

## 6. Decision
**GO.** Every essential check passes: the formula matches the implementation, no leakage was found, the gate passes when
recomputed from the saved out-of-fold files (folds 2-4 +0.0045; every seed >= 0.0046; holds without open-zero rows; fold 5
positive on every seed), the 365/730 mix is reproduced exactly, bagging is active, and the forecast file passes every hard
check. One CONDITION is stated: a retrain is not bit-identical (RMS log difference 0.0017).

**Frozen:** recipe `hp17_fix1000` x 3 seeds + TabPFN-TS (equal log mix of 365- and 730-day contexts) at weight 0.2. Artifact:
`final/` (forecast SHA-256 `7d44a050...4484`). No further model search. From here on, the remaining effort goes to the end-term
presentation, source-backed comparisons and the exclusive timing run.

## 7. Post-freeze checks (26 Nov 2026, evening; nothing in `final/` was refitted)
1. **Optuna finalists.** The five best pooled-Optuna trials were re-run on all five folds (seed 07, lr 0.025, 1000 fixed
   rounds, 24 threads). Against `hp17` on the same seed they gained +0.0001 to +0.0007 on folds 2-4, between -0.0057 and
   +0.0013 on fold 1, and between -0.0028 and +0.0006 on fold 5 (`results/g6c_finalists_5folds.csv`). None comes near the
   gate, so the frozen recipe stays.
2. **Leakage check (fold 4 = end-term test window).** Every decision that used fold-4 scores was re-taken with fold 4 removed
   (`code/leakage_check.py`, `results/leakage_check_without_fold4.csv`). None changes. The frozen blend passes on folds 2-3
   alone with +0.0050, every seed positive and fold 5 positive. `hp17` alone still fails.
3. **Re-running the frozen models from the repository.** `code/model/g8_final/export_forecast_matrix.py` wrote the exact
   input matrix of the forecast window on the server (`final/forecast_matrix_open.parquet`, 40,282 open rows x 342 columns +
   Id). `code/model/predict_frozen.py` and `notebook/02_run_frozen_forecaster.ipynb` rebuild every seed's forecast and
   `final/submission_final.csv` from the gzipped boosters with a maximum log difference of 0.0, in about 10 s on a laptop.
4. **Feature use.** Split gain of the three boosters by group (`results/feature_group_importance.csv`): the hub x weekday x
   promotion profile is 46%, rolling order windows 22%, the promotion calendar 8% and the hub encoding 6%. The 115
   session/conversion columns are 2.5%.

The decision in section 6 is unchanged: **GO**, with the same single condition.
