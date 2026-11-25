# Winner-solution sweep: techniques not yet in the plan (2026-11-25)

Scope: primary sources only (official solution repos, author-written summaries, arXiv full text or abstract). Every row cites a URL that was actually opened. Ranked items exclude everything already in the plan (Rossmann ideas, M5 store-level models, Chronos-2/TabPFN-TS pilots, hurdle, month-ahead, horizon-specific, reconciliation, regime experts).

## 1. Search protocol (reproducible)
- GitHub: `Mcompetitions/M5-methods` tree (A1, A2, A4, A5; A3 is absent), `Arturus/kaggle-web-traffic`, `davidthaler/Walmart_competition_code`, `LenzDu/Kaggle-Competition-Favorita`, `antklen/kaggle_favorita-grocery-sales`. GitHub repo search for "store-sales-time-series-forecasting" (sorted by stars).
- arXiv API queries: retail AND foundation AND forecasting; M5 AND LightGBM; Favorita; "global forecasting models" AND retail; foundation AND LightGBM AND sales; titles fev-bench / Chronos-2 / GIFT-Eval. Full HTML was read for 2610.25170, 2601.18919, 2510.15821 and 2509.26468. Only the abstract was read for 2505.00356, 2508.10063 and 2605.12200.
- Kaggle discussion/write-up pages render with JavaScript, and the archived copy (Wayback 2019) has no post body. So Kaggle-only write-ups are **UNVERIFIED** (Section 5).

## 2. Evidence extracted (what each source actually says)
| ID | Source (opened) | Technique | Evidence in source |
|---|---|---|---|
| S1 | github.com/Arturus/kaggle-web-traffic/blob/master/how_it_works.md (1st place, Web Traffic) | No early stopping. Fixed training region, 10 checkpoints (steps 10500-11500) x 3 seeds, average of all 30. Final model trained "blind" on all data | Author: validation correlated "very weakly" with future data. With the 30-checkpoint average, LB error was "roughly the same" as validation. No ablation number |
| S2 | same | Year-ago and quarter-ago datapoints as explicit inputs, smoothed `0.25*d364+0.5*d365+0.25*d366` | "considerably lowered prediction error". Lagged points alone slightly beat attention. No number |
| S3 | same | Hyperparameter comparisons made only on equal seeds (SMAC). "Best models had roughly the same performance" | Qualitative |
| S4 | github.com/Mcompetitions/M5-methods, A1 `2-1. nonrecursive_store_TRAIN.py`, `3-1. Final ensemble.py` (M5 1st) | Fixed 3000 rounds at lr 0.015, no `early_stopping_rounds`. Tweedie p=1.1, feature_fraction 0.5, bagging 0.5. Final = equal mean of 6 models: {recursive, non-recursive} x {store, store-cat, store-dept} | Code only. The win itself is the evidence. No ablation in repo |
| S5 | M5-methods A4 `M5_Accuracy_4th_Model_Summary...docx` (M5 4th) | Fixed 1400 rounds, untuned public params, no blending. "Validation scores varied significantly over the period" | Author says recognising CV instability was the key. No number |
| S6 | M5-methods A2 `Documentation_v01.pdf`, `m5-final-XX.py`, `m5-alignandsubmit.py` (M5 2nd) | Bottom-level LGBM with **all lag/rolling features removed** (calendar/price/id only). Asymmetric-MSE custom loss at multipliers 0.90-0.99, 5 runs averaged, aligned to an N-BEATS top level | Author: lag features gave "very unstable results"; the alignment was the "most important trick". No number |
| S7 | M5-methods A5 `Model Summary.pdf` (M5 5th) | Poisson LGBM per dept. Per store x dept multiplicative factor fitted on validation | Model ran 3-4% low on average. Factors = "biggest difference" (already covered by our bias work) |
| S8 | github.com/davidthaler/Walmart_competition_code README, `runAll.R`, `grouped.forecast.R` (Walmart 1st) | Per-dept matrix (weeks x stores) replaced by its rank-12 SVD reconstruction before forecasting. `stlf.nn`: average each store's forecast with up to k=5 stores correlated above 0.8 (all above 0.95) | Final LB WMAE: best single (stlf.svd ets) 2348. Simple-model average 2503. Full 6-model average ~2300 (2320 without fourier.arima) |
| S9 | github.com/LenzDu/Kaggle-Competition-Favorita `lgbm.py`, README (Favorita 5th) | 16 direct per-day models, 14 weekly origins stacked. Features: `last_year_mean/count0/promo` over the year-ago copy of the 16-day target window. Promo-conditioned means (`mean_i_haspromo/nopromo`), zero-day counts `count0_i`. Cross-sectional item-group and store-group means over windows 7-365. Fixed 700 rounds. Seed-averaged LGBM + CNN + seq2seq blend | Each model alone "top 1%". No ablation |
| S10 | arxiv.org/html/2601.18919v1 (VN2 winner report, CatBoost, 2026) | Censored periods masked into an "effective" series. All target features computed on it. 53-week scale only if at least 45 observed weeks, else warm-start expanding mean (or a higher-level prior). Two-level median imputation so trees do not split on structural NaNs | Won all 6 rounds as a pipeline. **No per-component ablation** |
| S11 | arxiv.org/html/2610.25170v3 (preprint, 2 authors, WRMSSE, M5/Favorita/Dominick's/Frat) | Equal-weight mean of tuned LightGBM + a dissimilar trained neural model. Changes kept only if the 3-seed mean beats the best by at least the seed spread ("keep bar") | Average lowest in 22/27 grain x dataset cases. Fitted weights beat equal weights in 14/39 and lost 7. LightGBM keep bars 0.002-0.010. Search-window gain vs held-out gain correlation **0.05 for LightGBM**. Tuned LGBM settings were worse on held-out than the starting settings on 5/9 datasets |
| S12 | arxiv.org/html/2509.26468v5 (fev-bench) and arxiv.org/html/2510.15821v1 (Chronos-2) | Benchmark: 100 tasks, 46 with covariates. LightGBM/CatBoost run **recursively via MLForecast with grid-searched lags only** | Chronos-2 avg win rate 87.2%. CatBoost 52.4% and LightGBM lower in the marginal tables. The GBDT baselines are not feature-engineered, so this says little about our direct, 340-feature GBDT |

## 3. Top 10 ranked techniques (new to our plan)
Score = G x P / C, each on a 1-3 scale. G = expected gain, P = plausibility on our hub panel, C = cost. The gain sizes are my prior judgements; none of the sources report an RMSLE number.

| # | Technique | Sources | How to apply to our setup | G | P | C | Score | Leakage risk |
|---|---|---|---|---|---|---|---|---|
| 1 | **Fixed rounds + iteration-band averaging + seed bagging**, replacing per-fold early stopping | S1, S4, S5, S9 | Pick one round count per model from the median best iteration across folds (hold lr fixed). Predict at about 10 iterations spread over a band (e.g. 0.8x-1.2x) and average them. Average 3-5 seeds. Costs no extra fits beyond the seeds (`num_iteration=`). Directly targets our 100-2500 early-stop spread and 0.008 seed swing | 2 | 3 | 1 | 6.0 | **Reduces** leakage: if early stopping now watches the fold being scored, the current CV is optimistic |
| 2 | **Gap-aware "effective series"** for the 180 gap hubs | S10 | Mark the 184 days as missing, never as 0 or interpolated. Compute rolling medians/means and lag-364 on observed days only. Add `n_observed_in_window` features. Below a coverage minimum (VN2 used 45/53), fall back to an expanding mean or a format-level prior. Fill structural NaNs with per-hub and then global medians of the scaled feature | 2 | 2 | 1 | 4.0 | Low. Imputation statistics must use only pre-origin data |
| 3 | **Year-ago target-window features**: smoothed and windowed, not a point lag | S2, S9 | For target day d: same-weekday mean of {d-371, d-364, d-357}. For the origin: mean and zero-count over the year-ago 42-day window, its ratio to the year-ago 28 days before the origin (seasonal shape), and the year-ago promo count. Since 364 > 42, every input is at or before the origin | 2 | 2 | 1 | 4.0 | Low. Breaks on the gap hubs, so combine with #2 |
| 4 | **Cross-sectional peer aggregates at the origin** | S8 (stlf.nn), S9 (store/item group means) | Format / assortment-tier / region-proxy means of 7-365-day log orders and growth ratios (28d/364d) at the origin. Plus k=5 most-correlated hubs (correlation computed on pre-origin history), pooled rolling medians, and the peer forecast mean as a feature | 2 | 2 | 1 | 4.0 | Medium: peers' windows and correlations must stop at the origin. Never use peers' future targets |
| 5 | **Seed-paired "keep bar" acceptance rule** for all experiments | S11, S3 | Run each candidate on the same 3 seeds as the baseline. Accept only if the 3-seed mean gain exceeds the baseline seed spread on the same folds. Also report a held-out final window that no tuning decision touched. S11 found LightGBM search gains barely predict held-out gains (r=0.05) | 2 | 3 | 2 | 3.0 | None. It guards against selection overfit |
| 6 | **Low-rank SVD denoising** of the hub x day matrix | S8 | Per origin and per format: take log1p orders over the trailing ~365 days, keep a rank-k reconstruction (Walmart used k=12-15), and compute the rolling and year-ago features on the reconstruction. Alternatively use hub loadings as static embedding features | 2 | 2 | 2 | 2.0 | Medium: refit the SVD per origin. A single SVD over the full panel leaks the future |
| 7 | **Equal-weight blend with one non-tree member**, instead of more GBDT variants | S11, S9 | Our blend is three GBDTs (low diversity). S11 shows the gain comes from a dissimilar model, and that fitted weights rarely beat equal weights (consistent with our finding). Priority: train the planned entity-embedding MLP on the same 340 features and blend at 1/2 or 1/3 in log space | 3 | 2 | 3 | 2.0 | Low if the NN uses the same origin-time features |
| 8 | **Zero/closure and promo-conditioned history counts** | S9 | `count0` over 7/14/28/91/364 days, days since the last zero day, and mean orders on promo vs non-promo days over the same windows (we have weekday x promo only) | 1 | 2 | 1 | 2.0 | Low |
| 9 | **Denser training origins** (every 7 days instead of 14) | S9 (14 weekly origins), S1 (random-start sampling) | Double the training origins while keeping the scoring origins. Purge any training origin whose 42-day target window overlaps the validation window | 1 | 2 | 2 | 1.0 | Medium: needs the purge gap |
| 10 | **Recursive + direct mixture** | S4 | Add a daily recursive LGBM (lag-1+ features, rolled forward 42 steps) and average it equally with the direct models. M5's horizon was 28. Error accumulates over 42 steps, so plausibility is low | 2 | 1 | 3 | 0.7 | Medium: easy to leak true lags into the roll-forward |

Considered, not ranked:
- **S6** (no-lag model) as a diversity member: hub orders are not intermittent, which was the author's stated reason for dropping lags.
- **S7** multipliers: same as our bias offsets, which did not help.
- **Asymmetric-loss ensembles (S6):** RMSLE is symmetric in log space.
- **Retraining frequency (arXiv 2505.00356, abstract only):** less frequent retraining kept accuracy on two retail datasets. It is a compute saver that could pay for #1's seeds, not an accuracy gain.
- **Forecast stability (arXiv 2508.10063, abstract only):** ensembles improved stability on M5/Favorita without hurting accuracy. Supports #1.
- **For the existing pilots (S12; 2605.12200 abstract):** run Chronos-2 in full cross-learning mode with sessions as a past-only covariate. TabPFN-TS fit simple covariate-target relations better than Chronos-2 at short horizons.

## 4. Critical appraisal
- Evidence strength: **weak to moderate overall.**
  - The competition sources are single-run wins with no ablations (S1, S4, S5, S6, S9, S10). Only S8 (Walmart LB numbers) and S11 (paired tests over 9 datasets) give quantitative comparisons.
  - S11 is a recent non-peer-reviewed preprint scored on WRMSSE, not RMSLE.
- Consensus across independent winners (S1, S4, S5, S9): **fixed boosting budgets and averaging over seeds/checkpoints, not early stopping.** Four separate competitions converge on this, so #1 is the best-supported item.
- Publication bias: write-ups describe what the winners kept. Techniques that failed for them are mostly unreported, so the absence of a technique here is not evidence against it.
- Transfer threat: M5 and Favorita are item-level and intermittent. Our 1,115 hubs are aggregate, dense series, so lag and level features are more informative for us than S6 suggests.

## 5. UNVERIFIED (not opened, not ranked)
- Favorita 1st-place write-up (kaggle.com/.../writeups/w-1st-place-solution): the page is JS-rendered and the archived copy has no body.
- M5 1st-place Kaggle discussion and Matthias's public write-up (discussion/164599): only the code and PDF in M5-methods were verified.
- M5 3rd place: no folder in M5-methods.
- M5 Accuracy results paper (IJF, sciencedirect S0169207021001874): HTTP 403.
- Walmart forum posts t/8125 and t/8028 (Christmas-shift rationale): linked from the README but not opened.
- Store Sales (Ecuador): no winner write-up found. GitHub hits are low-star tutorials, and the competition is a rolling playground.
