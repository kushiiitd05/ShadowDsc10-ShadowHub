# CSE558 End-term, group DSC 10: coresets for 42-day hub demand, and an audited, frozen forecaster

Course: CSE558 Data Science, IIIT-Delhi, Monsoon 2026. Dataset: the Shadow Hub database (1,115 retail hubs, daily orders,
1 Jan 2013 to 19 Jun 2015). The raw files are in [`../1_midterm/data/raw`](../1_midterm/data/raw).

## Deliverables

| File | What it is |
|---|---|
| [`report/main.pdf`](report/main.pdf) | The end-term report (LaTeX, built on the course template; source `report/main.tex`) |
| [`notebook/DSC10_endterm.ipynb`](notebook/DSC10_endterm.ipynb) | Executed notebook: rebuilds every table and figure and recomputes the model gate from saved predictions |
| [`code/`](code/) | All code: end-term experiments, coreset, model training, TabPFN-TS member, final retrain, sanity checks, audit, figures |
| [`final/`](final/) | Frozen forecaster: three LightGBM boosters (gzip), forecast file, manifest with SHA-256, freeze record, checks |
| [`docs/FREEZE_AUDIT.md`](docs/FREEZE_AUDIT.md) | Seven-row freeze audit, recomputed gate table, open issues, GO/NO-GO |
| [`docs/RESEARCH_REPORT.md`](docs/RESEARCH_REPORT.md) | Short research report of the model-improvement work |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | How to rerun everything, at three levels of cost |
| [`docs/PRESENTATION_BRIEF.md`](docs/PRESENTATION_BRIEF.md) | 7-minute, 8-slide plan with the numbers per slide and likely questions |
| [`results/experiment_ledger.md`](results/experiment_ledger.md) | Every idea tried, the measured result and why it failed or passed |
| `MANIFEST.sha256` | SHA-256 of every file in this folder (`shasum -a 256 -c MANIFEST.sha256`) |

## What the study does

1. **End-term track (the course rubric).** We fix one time-ordered split (train = origins whose labels end by 8 May 2015,
   test = the last 42 days, 46,830 rows) and train the full model x* (LightGBM, 1,133,640 rows x 342 columns) and ridge. We then
   train the same models on row summaries of 2k-200k rows: uniform, hub-stratified and a sensitivity-sampling coreset (5
   rough models, s_i = max_t[f_i/F_sum + 1/n], p_i = s_i/S, weight 1/(m p_i)). Each size is repeated 5 times. We also run a
   column experiment (importance-chosen vs random columns). Outputs: Plots 1-5, speed-up table, S, alpha, slopes.
2. **Model track.** We improve x* through more than 40 ideas and three searches under an adoption gate fixed in advance. Then
   we freeze the winner after an independent recomputation of the gate from saved out-of-fold predictions.

## Headline results (all from `results/`)

| Quantity | Value |
|---|---|
| Full LightGBM x*, test RMSLE / R² (log) / fit time | 0.1022 / 0.9992 / 59.7 s (ridge 0.1592 / 0.9980 / 4.3 s) |
| Coreset at m = 50,000 rows (4.4%) vs uniform | 0.1070 vs 0.1102 test RMSLE; 2.75x vs 5.18x end-to-end speed-up |
| Coreset vs uniform, LightGBM | better at every size from 5k to 200k; tie at 2k |
| Total sensitivity S / largest roughness alpha | 2.47 / 2.45 (LightGBM), 2.12 / 1.01 (ridge) |
| Slope ratio Plot 5 / Plot 3 | 0.66 ridge (theory about 0.5), 0.33 LightGBM (no guarantee for trees) |
| Frozen forecaster vs like-for-like baseline | +0.0045 RMSLE on folds 2-4 (each seed 0.0046-0.0047); passes every gate condition |

## Frozen protocol (fixed before scoring)
- Metric: RMSLE (RMSE of log1p); R² on the log scale; closed days kept with prediction 0, plus RMSLE without open-but-zero rows.
- Seeds: 20260807, 20260808, 20260809. Noise floor: 0.0012 (seed-to-seed). Gate: folds 2-4 mean gain >= 0.0024, >= 2 of 3
  folds improve, same sign on all seeds, holds without open-zero rows, fold 5 checked.
- End-term: identical model settings at every summary size; 5 repeats with new seeds; medians with min-max bands; uniform on
  every plot; test rows never used for fitting, sampling probabilities or column choice.

## Run
```bash
pip install -r requirements.txt
python code/freeze_audit.py && python code/make_figures.py
python code/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebook/DSC10_endterm.ipynb
cd report && tectonic main.tex
```
The heavy steps (origin tables, 260 end-term fits, TabPFN-TS, final retrain) need the server; see `docs/REPRODUCIBILITY.md`.

## Folder map

| Folder | Contents |
|---|---|
| `code/endterm/` | `endterm_exp.py` (split, x*, sampling, coreset, timing), `coreset.py`, `make_plots.py`, `test_coreset.py` |
| `code/model/` | `exp_runner.py` (recipes, folds), `helpers.py`, `blend.py`, `g8_final/` (final retrain, sanity checks, tests), `g10_neural/` (TabPFN-TS), `g6c_blend/` (Optuna study) |
| `code/src/` | feature pipeline used for every run (origin tables, folds, forecast frame) |
| `code/` | `freeze_audit.py`, `make_figures.py`, `build_notebook.py` |
| `results/` | end-term results, speed-up, slopes, search tables (20 configs, 80 Optuna trials, TabPFN grid), audit outputs, ledger |
| `artifacts/oof/` | out-of-fold predictions: `cs02` and `hp17` (3 seeds, folds 1-5), TabPFN-TS 365 and 730 days |
| `artifacts/test_window/` | TabPFN-TS forecasts for the forecast window (365, 730, mix) with their manifests |
| `artifacts/rehearsal_*` | two rehearsal retrains on the real path (365-day input; mix input), kept as fallbacks |
| `final/` | the frozen forecaster |
| `figures/`, `report/`, `notebook/`, `docs/` | as named |

## Limits we state with every result
Fold 4 shaped the recipe family and is also the end-term test window, so the end-term numbers are development-fold
results. Folds 1, 4 and 5 are diagnostics, not untouched test sets. The forecast window has no labels in this project, so no
real test score exists. All timings come from a shared server, and an exclusive rerun is still open. The coreset has no
guarantee for trees.
