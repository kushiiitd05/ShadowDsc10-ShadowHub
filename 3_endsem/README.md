# CSE558 End-term, group DSC 10: coresets for 42-day hub demand, and an audited, frozen forecaster

Course: CSE558 Data Science, IIIT-Delhi, Monsoon 2026. Dataset: the Shadow Hub database (1,115 retail hubs, daily orders,
1 Jan 2013 to 19 Jun 2015). The raw files are in [`../1_midterm/data/raw`](../1_midterm/data/raw).

## Start here

| Want to | Open |
|---|---|
| Read the study | [`report/main.pdf`](report/main.pdf) |
| See every table and figure rebuilt (laptop or Colab, under a minute) | [`notebook/01_endterm_results.ipynb`](notebook/01_endterm_results.ipynb) |
| Re-run the frozen forecaster and rebuild the submitted file (about 10 s) | [`notebook/02_run_frozen_forecaster.ipynb`](notebook/02_run_frozen_forecaster.ipynb) |
| Build the slides | [`docs/PRESENTATION_BRIEF.md`](docs/PRESENTATION_BRIEF.md) (9 slides, slide by slide, with the reasoning) |
| Take over the remaining work | [`docs/TEAM_HANDOVER.md`](docs/TEAM_HANDOVER.md) |

Both notebooks start with a setup cell. When they are opened outside the repository (for example on Colab), it clones the
repository with a sparse checkout of `3_endsem/` and the raw data, and installs any missing package.

## Two tracks, three model names

| Name | What it is | Used for |
|---|---|---|
| `cs02` | LightGBM: 63 leaves, feature fraction 0.2, row bagging 0.85, min child 80, learning rate 0.05, 1184 rounds | **x\*** of the coreset study (Track A, the course rubric), with ridge as the second model |
| `hp17` | The same LightGBM, tuned: 255 leaves, feature fraction 0.15, min child 160, L2 24.1, learning rate 0.025, 1000 rounds | The trainable core of the submitted forecaster |
| Submitted forecaster | `expm1(0.8 * mean over 3 seeds of log1p(hp17) + 0.2 * log1p(TabPFN-TS))`. TabPFN-TS is the equal log mix of its 365- and 730-day contexts, 0.10 each. Closed days = 0 | `final/submission_final.csv` for 20 Jun-31 Jul 2015, a window with no labels, so it has development-fold scores only (Track B) |

The submitted blend is not coreset-ed. TabPFN-TS is pretrained and reads each hub's history at prediction time, so it has no
training loss and no rows to weight. The rubric also needs the same model refitted for every summary.

## Deliverables

| File | What it is |
|---|---|
| [`report/main.pdf`](report/main.pdf) | The end-term report (LaTeX on the course template; source `report/main.tex`) |
| [`notebook/`](notebook/) | `01_endterm_results.ipynb` (results, gate, leakage check) and `02_run_frozen_forecaster.ipynb` (frozen models), both executed |
| [`code/`](code/) | All code: end-term experiments, coreset, model training, TabPFN-TS member, final retrain, sanity checks, audit, leakage check, figures |
| [`final/`](final/) | Frozen forecaster: three LightGBM boosters (gzip), their exact input matrix (`forecast_matrix_open.parquet`), the forecast file, a manifest with SHA-256, the freeze record and the checks |
| [`docs/PRESENTATION_BRIEF.md`](docs/PRESENTATION_BRIEF.md) | 7-minute, 9-slide plan: the claim, figure, notation, script and reasoning per slide, plus a question sheet |
| [`docs/TEAM_HANDOVER.md`](docs/TEAM_HANDOVER.md) | What is done, what is left, who did what, and the files to read |
| [`docs/FREEZE_AUDIT.md`](docs/FREEZE_AUDIT.md) | Seven-row freeze audit, recomputed gate table, post-freeze checks, open issues, GO/NO-GO |
| [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) | Short research report of the model-improvement work |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | How to rerun everything, at three levels of cost |
| [`results/experiment_ledger.md`](results/experiment_ledger.md) | Every idea tried, the measured result and why it failed or passed |
| `MANIFEST.sha256` | SHA-256 of every file in this folder (`shasum -a 256 -c MANIFEST.sha256`) |

## Headline results (all from `results/`)

| Quantity | Value |
|---|---|
| Full LightGBM x\*: test RMSLE / R² (log) / fit time | 0.1022 / 0.9992 / 59.7 s (ridge 0.1592 / 0.9980 / 4.3 s) |
| Coreset at m = 50,000 rows (4.4%) vs uniform | 0.1070 vs 0.1102 test RMSLE; 2.75x vs 5.18x end-to-end speed-up |
| Coreset vs uniform, LightGBM | better at every size from 5k to 200k; tie at 2k |
| Total sensitivity S / largest roughness alpha | 2.47 / 2.45 (LightGBM), 2.12 / 1.01 (ridge) |
| Slope ratio Plot 5 / Plot 3 | 0.66 ridge (theory about 0.5), 0.33 LightGBM (no guarantee for trees) |
| Features the trees use (split gain) | hub x weekday x promotion profile 46%, rolling order windows 22%, promotion calendar 8% |
| Fold 4 = end-term test window | every model decision re-taken without fold 4: none changes (`results/leakage_check_without_fold4.csv`) |
| Submitted forecaster vs like-for-like baseline | +0.0045 RMSLE on development folds 2-4 (each seed 0.0046-0.0047); passes every gate condition, also without fold 4 |

## Frozen protocol (fixed before scoring)
- **Metric:** RMSLE (RMSE of log1p) and R² on the log scale. Closed days are kept with prediction 0, and RMSLE without open-but-zero rows is also reported.
- **Seeds:** 20260807, 20260808, 20260809. Noise floor: 0.0012 (seed-to-seed).
- **Gate:** folds 2-4 mean gain >= 0.0024; >= 2 of 3 folds improve; same sign on all seeds; holds without open-zero rows; fold 5 checked.
- **End-term:**
  - Identical model settings at every summary size.
  - 5 repeats with new seeds; medians with min-max bands; uniform on every plot.
  - Test rows are never used for fitting, sampling probabilities or column choice.

## Run
```bash
pip install -r requirements.txt
python code/model/predict_frozen.py            # rebuild the submitted file from the frozen models (exact)
python code/freeze_audit.py && python code/leakage_check.py && python code/feature_importance.py && python code/make_figures.py
python code/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebook/*.ipynb
python -m pytest -q code                        # unit tests
cd report && tectonic main.tex
```
The heavy steps (origin tables, 260 end-term fits, TabPFN-TS, final retrain) need the server; see `docs/REPRODUCIBILITY.md`.

## Folder map

| Folder | Contents |
|---|---|
| `code/endterm/` | `endterm_exp.py` (split, x\*, sampling, coreset, timing), `coreset.py`, `make_plots.py`, `test_coreset.py` |
| `code/model/` | `exp_runner.py` (recipes, folds), `helpers.py`, `blend.py`, `predict_frozen.py` (re-run the frozen forecaster), `g8_final/` (final retrain, input-matrix export, sanity checks, tests), `g10_neural/` (TabPFN-TS), `g6c_blend/` (Optuna study) |
| `code/src/` | feature pipeline used for every run (origin tables, folds, forecast frame) |
| `code/` | `freeze_audit.py`, `leakage_check.py`, `feature_importance.py`, `make_figures.py`, `build_notebook.py`, tests |
| `results/` | end-term results, speed-up, slopes, search tables (20 configs, 80 Optuna trials, 5 finalists, TabPFN grid), audit outputs, leakage check, feature importance, ledger |
| `artifacts/oof/` | out-of-fold predictions: `cs02` and `hp17` (3 seeds, folds 1-5), TabPFN-TS 365 and 730 days, `base` vs `cs02` screening runs |
| `artifacts/test_window/` | TabPFN-TS forecasts for the forecast window (365, 730, mix) with their manifests |
| `artifacts/rehearsal_*` | two rehearsal retrains on the real path (365-day input; mix input), kept as fallbacks |
| `final/` | the frozen forecaster |
| `figures/`, `report/`, `notebook/`, `docs/` | as named |

## Limits we state with every result
- **Fold 4 is also the end-term test window.** No decision changes without it, but the end-term numbers are development-fold results, not an untouched holdout.
- **Folds 1, 4 and 5 are diagnostics,** not untouched test sets.
- **No real test score exists.** The forecast window has no labels in this project.
- **Timings are not final.** They come from a shared server, and an exclusive rerun is still open.
- **No guarantee for trees.** The coreset's advantage for LightGBM is empirical.
