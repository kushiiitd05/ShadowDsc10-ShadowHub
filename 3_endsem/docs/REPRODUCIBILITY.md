# Reproducibility guide

Four levels, from cheapest to full.

## Level 0: anywhere, including Colab (no setup)
Open `notebook/01_endterm_results.ipynb` or `notebook/02_run_frozen_forecaster.ipynb` (on Colab: File > Open notebook >
GitHub, repository `kushiiitd05/ShadowDsc10-ShadowHub`). The first cell finds the `3_endsem` folder. If the folder is
missing, the cell clones the repository with a sparse checkout (`3_endsem/` and `1_midterm/data/raw/`, about 200 MB including git history) and
installs any missing package (`lightgbm`, `pyarrow`). Notebook 1 rebuilds every table and figure from the saved outputs.
Notebook 2 loads the three frozen boosters and their exact input matrix and rebuilds `final/submission_final.csv`, with a
maximum log difference of 0.0.

## Level 1: rebuild the report's numbers and figures (laptop, under a minute)
Needs Python 3.11 or newer (`pip install -r requirements.txt`).

```bash
cd 3_endsem
python code/model/predict_frozen.py  # frozen boosters + input matrix + TabPFN-TS -> the submitted file, checked exactly
python code/freeze_audit.py          # gate, context mix and rehearsal checks -> results/audit_*.csv, audit_summary.json
python code/leakage_check.py         # every decision re-taken without fold 4 -> results/leakage_check_without_fold4.csv
python code/feature_importance.py    # split gain by feature group -> results/feature_group_importance.csv
python code/make_figures.py          # figures/*.pdf, *.png from results/*.csv
python code/build_notebook.py        # writes notebook/01_*.ipynb and 02_*.ipynb
jupyter nbconvert --to notebook --execute --inplace notebook/*.ipynb
python -m pytest -q code             # unit tests
cd report && tectonic main.tex       # report/main.pdf
```
Inputs:
- `results/`: the end-term run and the search tables.
- `artifacts/`: out-of-fold predictions, TabPFN-TS forecast-window predictions and the rehearsal outputs.
- `final/`: the frozen models, their input matrix and the forecast.
- `../1_midterm/data/raw/`: the raw data. The forecast frame has no labels.

## Level 2: check the frozen artifact
```bash
cd 3_endsem
shasum -a 256 -c MANIFEST.sha256                       # every file in the folder
gunzip -k final/models/lightgbm_s2026080*.txt.gz       # boosters; SHA-256 of the unzipped files is in final/final_manifest.json
```
`final/freeze_checks.json` records the hard checks: rows, Id order, NaN, negatives and closed rows exactly 0. The checks
were recomputed on this machine from the saved file.

`final/forecast_matrix_open.parquet` was written on the server by `code/model/g8_final/export_forecast_matrix.py`:
- It contains the same columns, the same hub encoding (first seed) and the same preparation as the final retrain.
- That script refused to save unless every booster reproduced its saved forecast from the file with a difference of 0.0.

## Level 3: rerun the experiments (server, about 1-2 hours of CPU plus a GPU for TabPFN-TS)
The code in `code/` is byte-identical to the code that produced the outputs (SHA-256 in `final/FREEZE_COMMAND.txt` and
`MANIFEST.sha256`).
1. **Feature tables.** `code/src/eda_pipeline.py` and `code/src/model_pipeline.py` build the 37 training origins, the
   development folds and the forecast frame from the raw files. Origin hashes are in `training_origin_manifest.csv`, which
   `final_retrain.py` checks.
2. **Development folds.** `code/model/exp_runner.py <variant>` writes out-of-fold predictions. The registered variants
   include `cs02_fix1000` and `hp17_fix1000`. Seeds are 20260807/08/09.
3. **TabPFN-TS.** `code/model/g10_neural/neural_runner.py` runs the folds. `neural_testwindow.py --context 365|730` runs the
   forecast window in about 13-18 minutes on one A100. It is deterministic: two seeds gave identical predictions (max
   difference 0.0 on 179,760 rows).
4. **Final retrain.** The exact command, run from `code/model`:
   ```bash
   python g8_final/final_retrain.py <proj_root> <new_out_dir> --recipe hp17 \
       --blend-file <tabpfn_mix_test.csv> --blend-weight 0.2 --seeds 20260807,20260808,20260809 --n-jobs 16
   ```
   It refuses to write into a folder that already has a manifest, refuses a blend file whose Ids differ from the forecast
   frame, and refuses a recipe that differs from the registered variant.
5. **Forecast-window input matrix.**
   `python g8_final/export_forecast_matrix.py <proj_root> <freeze_dir> forecast_matrix_open.parquet`
6. **End-term.** `code/endterm/endterm_exp.py` runs 5 repeats x 7 sizes x 3 methods x 2 models, plus the columns
   experiment. `make_plots.py` draws the plots. `test_coreset.py` checks that S lies in [2, 6].

Software used for the final retrain: Python 3.11.18, LightGBM 4.7.0, numpy 1.25.2, pandas 2.3.3, scikit-learn 1.7.2 (also
in `final/final_manifest.json`).

**A retrain is not bit-identical.** Compared with rehearsal 2 (same data, seeds and recipe, 16 threads on a loaded server),
the blended forecast differs by an RMS log difference of 0.0017. Two different seeds differ by 0.019. See
`docs/FREEZE_AUDIT.md`, section 4. Reproducibility here therefore means that the saved boosters reproduce the forecast
exactly (Level 0 and Level 1). For bit-identical refits, add `deterministic=True, force_col_wise=True`.

## Wording scan
```bash
grep -rniE "state of the art|leaderboard|public score|rank(ed|ing)" report README.md docs notebook
```
The scan should return only this block and the "do not say" lists in `docs/PRESENTATION_BRIEF.md` and
`docs/TEAM_HANDOVER.md`. Files under `code/` that ran on the server are kept byte-identical to the code that ran, so their
hashes still match the run records.
