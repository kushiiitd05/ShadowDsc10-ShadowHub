# End-term presentation brief: 7 minutes, 9 slides, 3 minutes of questions

Everything a slide needs is here: the claim (the slide title), the figure file, the on-slide text, the notation, what to say,
why we made each choice, and where the number comes from. **Every number comes from `results/` or `report/main.pdf`; do not add
numbers that are not there.** Figures are in `figures/` (PNG for slides, PDF for print).

## 0. Read this first: the confusions we cleared

1. **Two tracks, three model names.**
   - **Track A (the course rubric, slides 1-8):** the model $x^\*$ is LightGBM recipe `cs02` (63 leaves, feature fraction 0.2,
     row bagging 0.85, min child 80, learning rate 0.05, 1184 rounds). The second model is ridge. Every uniform,
     hub-stratified and coreset summary is fitted with exactly these settings.
   - **Track B (the forecaster we submit, slide 9):** `hp17` (the same LightGBM, tuned: 255 leaves, feature fraction 0.15, min
     child 160, L2 24.1, learning rate 0.025, 1000 rounds) x 3 seeds, blended with TabPFN-TS. Its output is
     `final/submission_final.csv`.
2. **The blend weights are 0.8 and 0.2, in log space.** 0.8 is the LightGBM average (3 seeds, about 0.267 each). 0.2 is
   TabPFN-TS, which is itself an equal log mix of a **365-day** and a **730-day** history window, so 0.10 each. The windows
   are 365 and 730 days, not 360, 720 or 740.
3. **"The test set" means two things.**
   - The **end-term test set** is 9 May-19 Jun 2015 (46,830 rows). It is labelled, and every rubric score comes from it.
   - The **submission window** is 20 Jun-31 Jul 2015. It has no labels in this project, so it has no score.
   - Never present a score for the submission window, and never compare with any outside score.
4. **Was it cheating that the end-term test window is also development fold 4?**
   - No test label entered any fit, sampling probability, column choice, number of rounds or ridge lambda.
   - Fold-4 *scores* were part of earlier recipe decisions. So we re-took every decision with fold 4 removed
     (`code/leakage_check.py`, `results/leakage_check_without_fold4.csv`), and **none changes**.
   - We still call 0.1022 a development-fold result, not an untouched holdout.
   - The coreset-vs-uniform comparison is unaffected either way, because both use the identical model.
5. **Why is the blend not coreset-ed?**
   - TabPFN-TS is pretrained and reads each hub's history at prediction time. It has no training loss F, no fitted x* and
     no rows to weight, so sensitivities do not exist for it.
   - The rubric's same-model rule would also mean refitting the whole blend for 5 repeats x 7 sizes x 3 methods.
   - The LightGBM part alone could be coreset-ed at several hours of server time. The rubric does not need it.
6. **Is the notebook needed?** The rubric asks for slides and plots, so the notebooks are supporting material.
   - `notebook/01_endterm_results.ipynb` rebuilds every table and figure.
   - `notebook/02_run_frozen_forecaster.ipynb` loads the frozen models and rebuilds the submitted file exactly, in about 10 s.
   - Both clone the repository themselves on Colab or a fresh machine.

## 1. Flow and timing (420 s)

| # | Rubric slot | Slide title (the claim) | Time |
|---|---|---|---|
| 1 | Data and problem | 1,115 hubs, 42 days ahead, one fixed time split | 50 s |
| 2 | Model | LightGBM is our x\*: 36% lower RMSLE than ridge, using known-ahead and history features | 50 s |
| 3 | Sampling + Plot 1 | A sensitivity coreset beats uniform at every size from 5k to 200k rows | 50 s |
| 4 | Time + speed-up | 50k coreset rows: 2.75x faster end to end, build included | 45 s |
| 5 | Coreset 1/3 | How the coreset is built: 5 rough models, sensitivities, S = 2.47 | 50 s |
| 6 | Coreset 2/3 | Plots 3-5: the coreset model stays closer to x\* | 50 s |
| 7 | Coreset 3/3 | Ridge checks the theory; where sampling does not help | 35 s |
| 8 | Conclusion | Use m = 50,000 rows: +0.0048 RMSLE, 59.7 s to 21.6 s, 0.0032 better than uniform | 45 s |
| 9 | Beyond the rubric | The forecaster we submit: tuned LightGBM + TabPFN-TS, +0.0045 on development folds | 45 s |

**Where slide 9 goes.** It goes after the conclusion.
- Slides 1-8 follow the rubric order exactly, with the conclusion as the last rubric slide.
- Slide 9 is clearly marked "beyond the rubric".
- Putting it earlier would mix up x\* (Track A) and the submitted model (Track B), which is the confusion section 0 removes.

If the team prefers to end on the conclusion, move slide 9 into the backup slides and say one line about it on slide 8.

Backup slides, shown only if asked:
- **B1** Leakage check table.
- **B2** Gate figure + TabPFN-TS grid.
- **B3** Ridge Plots 3-5.
- **B4** Experiment ledger (what failed and why).
- **B5** Optuna plateau + finalists.

## 2. Notation used on every slide

| Symbol | Meaning |
|---|---|
| $y_{h,t}$, $\hat y_{h,t}$ | actual and predicted orders of hub $h$ on day $t$ |
| $o$, $t=o+k$, $k=1..42$ | forecast origin and horizon |
| $f_i(x)=(\log(1+\hat y_i)-\log(1+y_i))^2$ | loss of training row $i$ under model $x$ |
| $F(P,w,x)=\sum_i w_i f_i(x)$ | weighted training loss; opt $=F(P,w,x^\*)$ |
| RMSLE $=\sqrt{\tfrac1N\sum_i f_i}$ | the metric (RMSE on the log scale) |
| $\tilde x_1..\tilde x_5$ | rough models (LightGBM on 20k uniform rows, 5 seeds); $\alpha_t=F(P,w,\tilde x_t)/\text{opt}$ |
| $s_i=\max_t[f_i(\tilde x_t)/F(P,w,\tilde x_t)+1/n]$ | sensitivity of row $i$; $S=\sum_i s_i$ |
| $p_i=s_i/S$, $w_i=1/(m\,p_i)$ | sampling probability and weight; $y^\*$ = model trained on the weighted coreset |
| $m$ | summary size (rows kept) |

## 3. Slide by slide

### Slide 1 - Data and problem
- **Title:** 1,115 hubs, 42 days ahead, one fixed time split.
- **Visual:** `figures/fig_pipeline.png`, cropped to the left half and Track A, or shown whole if it stays readable.
- **On the slide:**
  - Shadow Hub data: 1,115 hubs, daily orders, 1 Jan 2013-19 Jun 2015.
  - After encoding: 1,133,640 training rows x 342 columns = 387.7 M cells.
  - Split, fixed once: train = labels up to 8 May 2015; test = 9 May-19 Jun 2015 (46,830 rows).
  - Problem: predict $\log(1+y_{h,o+k})$ for $k=1..42$; metric RMSLE.
- **Say:**
  - "Each row is one hub on one future day, seen from a forecast origin."
  - "Features use history up to the origin only, plus flags the business knows in advance: open days, promotions, holidays."
  - "We split once by time and never move the split."
  - "The test window is also one of our development folds. We re-took every model decision without it, and none changed."
- **Why:**
  - Log target: volumes are skewed and errors are relative.
  - Time split: the task is forecasting, so a random split would leak the future.
  - Closed days are kept as 0, which inflates R², so we read RMSLE.
- **Source:** `report` section 1; `results/leakage_check_without_fold4.csv`.

### Slide 2 - Model
- **Title:** LightGBM is our x\*: 36% lower RMSLE than ridge.
- **Visual:** a table on the left, `figures/fig_feature_groups.png` on the right.
- **On the slide:**

  | Full model, test split | RMSLE | excl. open-0 | R² (log) | fit |
  |---|---|---|---|---|
  | LightGBM x\* | **0.1022** | 0.0932 | 0.9992 | 59.7 s |
  | Ridge | 0.1592 | 0.1539 | 0.9980 | 4.3 s |

- **Say:**
  - "Demand depends on interactions (weekday by promotion by hub format), and trees learn them without hand-made products."
  - "Ridge is our second model because coreset theory is exact for it, so it checks our implementation."
  - "On the right, which features the trees use: the hub-by-weekday-by-promotion profile carries 46% of the split gain, recent rolling levels 22%, the promotion calendar 8%. The 115 session columns add only 2.5%."
- **Why these features:**
  - Every column is known at the origin.
  - The data says a hub's day is mostly "this hub, this weekday, with or without a promotion", scaled by its recent level.
  - That is also why 40 well-chosen columns already score 0.1102 (slide 3).
- **Caveat to say if asked:**
  - Gain is not causal, and correlated columns share credit.
  - The importance is read from the frozen `hp17` boosters, which use the same 342 columns.
- **Source:** `results/endterm_run1_results.csv`, `results/feature_group_importance.csv`.

### Slide 3 - Sampling or sketching + Plot 1
- **Title:** A sensitivity coreset beats uniform at every size from 5k to 200k rows.
- **Visual:**
  - `figures/fig_plot1_rows.png`, LightGBM panel large.
  - Small inset: `figures/fig_plot1_columns.png`.
- **On the slide:**
  - **What:** rows (columns held at all 342). Methods: uniform (baseline), hub-stratified, sensitivity coreset.
  - **Sizes:** m = 2k, 5k, 10k, 25k, 50k, 100k, 200k; 5 repeats each; median with min-max band; dotted line = full model.
  - At 50k: coreset 0.1070, uniform 0.1102, full 0.1022.
  - Columns (rows held at 200k): 40 importance-chosen columns 0.1102 vs 40 random 0.1770.
- **Say:**
  - "We sample rows with probability proportional to a sensitivity: rows that cheap models fit badly (event days, hubs after a gap, promotion jumps) are drawn more often and down-weighted."
  - "That suits this data, because the error sits on few days: the 10 worst of 168 forecast days carry 28% of the squared error."
  - "At 2k rows it ties: a tree with min child 80 cannot use the weights at that size."
- **Why not a Gaussian sketch:** trees split on original columns, so projections do not apply. Columns are chosen by importance on training rows only.
- **Source:** report Table 2; `results/endterm_run1_results.csv`.

### Slide 4 - Time
- **Title:** 50k coreset rows: 2.75x faster end to end, build included.
- **Visual:** `figures/fig_plot2_time.png` (LightGBM panel) + the speed-up table.

  | m | T_build | T_train | RMSLE | speed-up | uniform |
  |---|---|---|---|---|---|
  | 10k | 10.2 s | 7.4 s | 0.1170 | 3.25x | 8.09x |
  | 25k | 10.2 | 9.5 | 0.1108 | 3.05x | 6.28x |
  | **50k** | 10.2 | 11.4 | **0.1070** | **2.75x** | 5.18x |
  | 100k | 10.2 | 14.0 | 0.1058 | 2.45x | 4.39x |
  | 200k | 10.2 | 21.8 | 0.1045 | 1.85x | 1.63x |

- **Say:**
  - "Speed-up = T_full / (T_build + T_train), with T_full = 59.7 s."
  - "Uniform is faster because it has no build, but it is less accurate: to match the coreset at 50k, uniform needs 100k-200k rows."
- **Limit, say it:** the timings come from a shared, busy server. An exclusive rerun is planned for 1-2 Dec. If it is done, replace these numbers and this sentence.
- **Source:** `results/endterm_run1_speedup.csv`.

### Slide 5 - Coreset 1/3: construction
- **Title:** How the coreset is built: 5 rough models, sensitivities, S = 2.47.
- **Visual:** the formula box + `figures/fig_hist_sensitivity.png` (log x-axis).
- **On the slide:**
  - $s_i=\max_{t}[f_i(\tilde x_t)/F(P,w,\tilde x_t)+1/n]$, with $n$ = 1,133,640 and no clusters, so $|G_{i,t}|=n$.
  - $p_i=s_i/S$, $w_i=1/(m\,p_i)$.
  - $S$ = 2.47 (LightGBM), 2.12 (ridge). Largest roughness $\alpha_{max}$ = 2.45 (LightGBM), 1.01 (ridge).
- **Say:**
  - "Five cheap LightGBMs on 20k uniform rows give five rough answers."
  - "Each row's sensitivity is its worst share of the loss across them, plus 1/n so every row stays reachable."
  - "Because we divide by the sum of losses, S must lie between 2 and 6, and a unit test enforces that."
  - "The histogram shows a long tail: a few rows are up to about 250 times more important than uniform."
- **Why:** this is the rubric's recipe. For trees there is no epsilon-guarantee, so we call it sensitivity-motivated importance sampling.
- **Source:** `results/endterm_run1_slopes_and_S.txt`.

### Slide 6 - Coreset 2/3: Plots 3, 4, 5 (LightGBM)
- **Title:** The coreset closes the training gap faster and its model stays closer to x\*.
- **Visual:** `figures/fig_plot345_lgbm.png`.
- **On the slide:**
  - **Plot 3** (train gap $F(P,w,y^\*)/opt-1$): coreset 0.373 vs uniform 0.557 at 50k.
  - **Plot 4** (test gap $|F(P_{test},w,y^\*)/F(P,w,x^\*)-1|$): 1.38 vs 1.52 at 50k; floor 1.17.
  - **Plot 5** (relative L2 distance of test predictions): 0.0049 vs 0.0056 at 50k.
  - **Slopes** (log-log): Plot 3 -0.59 vs -0.48; Plot 5 -0.19; ratio Plot 5/Plot 3 = 0.33.
- **Say:**
  - "Plot 4 does not go to zero, because its denominator is x\*'s training loss: the full model itself has a ratio of 1.17. Both methods approach that floor, and the coreset approaches it faster."
  - "Trees have no parameters to compare, so Plot 5 uses the test-prediction vector, as the rubric allows."
- **Source:** report section 5; `results/endterm_run1_results.csv`.

### Slide 7 - Coreset 3/3: ridge checks the theory; honest negatives
- **Title:** Where theory is exact, it matches; where sampling does not pay, we say so.
- **Visual:** `figures/fig_plot345_ridge.png` (small) + three bullets.
- **On the slide:**
  - Ridge: slope ratio Plot 5/Plot 3 = 0.66 against about 0.5 predicted (a loss gap epsilon gives a parameter gap of sqrt(epsilon)).
  - Ridge: test error the same for every method (0.159-0.166), and the build (18.6 s) is slower than the full fit (4.3 s). **No saving.**
  - LightGBM at 2k rows: coreset ties uniform.
- **Say:** "This is the check that our implementation is right, and the honest limit: for a cheap convex model, sampling saves nothing."
- **Source:** report section 5.

### Slide 8 - Conclusion
- **Title:** Use m = 50,000 rows with the sensitivity coreset.
- **On the slide** (four readouts):

  | Metric | Full model | Coreset at m = 50k | Change |
  |---|---|---|---|
  | Test RMSLE | 0.1022 | 0.1070 | +0.0048 |
  | R² (log) | 0.9992 | above 0.998 | |
  | End-to-end time | 59.7 s | 21.6 s | 2.75x faster |

  - vs uniform at 50k: **0.0032** better.
  - To match the coreset's error, uniform needs 2-4x as many rows.
- **Say:**
  - "4.4% of the rows, a third of the time, and the error rises by 0.0048."
  - "Uniform at the same size is faster but 0.0032 worse."
  - "For ridge we would not sample at all."
- **Source:** report section 10.

### Slide 9 - Beyond the rubric: the forecaster we submit
- **Title:** Our best forecaster: tuned LightGBM + TabPFN-TS, +0.0045 on development folds.
- **Visual:**
  - Left: `figures/fig_model_ladder.png`.
  - Right: formula box + `figures/fig_forecast_examples.png` (small).
  - Optional: `figures/fig_gate_gains.png`.
- **On the slide:**
  - $\hat y=\exp\big(0.8\cdot\tfrac13\sum_s\log(1+\hat y^{hp17}_s)+0.2\cdot\tfrac12[\log(1+\hat y^{TPF,365})+\log(1+\hat y^{TPF,730})]\big)-1$; closed days = 0.
  - Development folds (3-seed average), `cs02` to submitted model:

    | Fold | 1 | 2 | 3 | 4 | 5 |
    |---|---|---|---|---|---|
    | `cs02` → submitted | 0.1334 → 0.1302 | 0.1186 → 0.1141 | 0.1128 → 0.1073 | 0.1034 → 0.0998 | 0.1794 → 0.1737 |

  - Gate fixed in advance: a gain ≥ 0.0024 (2 x the noise). Measured +0.0045 on folds 2-4, every seed ≥ 0.0046, fold 5 positive.
  - It also passes with fold 4 removed: +0.0050.
  - `hp17` alone fails the gate (+0.0019).
- **Say:**
  - "Separately from the coreset study, we improved the forecaster we submit."
  - "We tried over 40 ideas and three searches. Only two things survived our gate: the tuned LightGBM, and a 20% blend with TabPFN-TS, a pretrained model that reads each hub's own history."
  - "Its errors differ from the trees', so the blend helps."
  - "80 Optuna trials found nothing better than noise. The five best, re-run on all folds, gained at most 0.0007."
  - "Its forecast window has no labels, so it has no test score. These are development-fold numbers."
  - "It is frozen with hashes, and notebook 2 rebuilds the submitted file exactly in 10 seconds."
- **Why not coreset it:** section 0, point 5.
- **Source:** `results/audit_fold_scores.csv`, `results/audit_gate_gains.csv`, `results/g6c_finalists_5folds.csv`, `docs/FREEZE_AUDIT.md`.

## 4. Question sheet (answers are in the report, section 8, and section 0 above)

| Question | Short answer |
|---|---|
| Why sensitivity sampling for trees? | No guarantee exists. Rows that rough models fit badly are the ones a small uniform sample misses. Plot 1 and Plot 5 show the gain empirically. |
| Why ridge? | The theory is exact for it, so it checks the code: the slope ratio is 0.66 against 0.5 predicted. |
| Noise floor? | 0.0012 seed-to-seed on the 3-fold mean; the gate is twice that. |
| Is fold 4 a clean test? | It is also a development fold. Without it no decision changes, but we report 0.1022 as a development-fold number. |
| Which model is submitted? | `hp17` x 3 seeds + TabPFN-TS at 0.2 (365/730 mix). It has no test score because there are no labels. |
| Why not coreset the submitted blend? | TabPFN-TS has no training loss or rows to weight. The rubric needs one trainable model refitted on every summary. |
| Why these features? | Hub x weekday x promotion profile 46% of gain, rolling levels 22%. All are known at the origin. |
| Why not tune more? | 80 Optuna trials gave +0.0004, a third of the noise; the 5 finalists gave +0.0001 to +0.0007. |
| Why is Plot 4 above 1? | Its denominator is x\*'s training loss; the full model's own ratio is 1.17. |
| Do timings hold? | They come from a shared server. Ratios are stable; the exclusive rerun is pending. |

## 5. Do not say or show
- No competition-platform names, leaderboard or "public" scores, rankings, "state of the art", or "beat the winner".
  - Our numbers are development-fold and end-term-split RMSLE.
  - They are not comparable to anything outside this project.
- Never call 0.1022 "unseen" or "a clean holdout", and never give a score for the 20 Jun-31 Jul window.
- No server names, IP addresses, key paths or personal names other than the group list.

## 6. Slide style
- The title is the claim, as a sentence. Use one figure per slide and at most three bullets.
- Put the source file in small grey text at the bottom.
- Colours match the figures: ultramarine for our method, grey for uniform or the baseline, amber for limits and losses.
- Numbers on slides use the same rounding as the report: 4 decimals for RMSLE, 2 for speed-ups.
