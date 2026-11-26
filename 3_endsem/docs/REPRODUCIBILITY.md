# Reproducibility guide

Three levels, from cheapest to full.

## Level 1: rebuild the report's numbers and figures (laptop, under a minute)
Needs Python 3.11 with numpy, pandas, matplotlib, nbformat, nbconvert, ipykernel (`pip install -r requirements.txt`).

```bash
cd 3_endsem
python code/freeze_audit.py          # gate, context mix and rehearsal checks -> results/audit_*.csv, audit_summary.json
python code/make_figures.py          # figures/*.pdf, *.png from results/*.csv
python code/build_notebook.py        # writes notebook/DSC10_endterm.ipynb
jupyter nbconvert --to notebook --execute --inplace notebook/DSC10_endterm.ipynb
cd report && tectonic main.tex       # report/main.pdf
```
Inputs: `results/` (end-term run, search tables), `artifacts/` (out-of-fold predictions, TabPFN-TS forecast-window
predictions, rehearsal outputs), `final/` (frozen models and forecast), `../1_midterm/data/raw/` (raw data; the forecast frame
has no labels).

## Level 2: check the frozen artifact
```bash
cd 3_endsem
shasum -a 256 -c MANIFEST.sha256                       # every file in the folder
gunzip -k final/models/lightgbm_s2026080*.txt.gz       # boosters; SHA-256 of the unzipped files is in final/final_manifest.json
```
`final/freeze_checks.json` records the hard checks (rows, Id order, NaN, negatives, closed rows exactly 0). The checks were
recomputed on this machine from the saved file.

## Level 3: rerun the experiments (server, about 1-2 hours of CPU plus a GPU for TabPFN-TS)
The code in `code/` is byte-identical to the code that produced the outputs (SHA-256 in `final/FREEZE_COMMAND.txt` and
`MANIFEST.sha256`).
1. Feature tables: `code/src/eda_pipeline.py` and `code/src/model_pipeline.py` build the 37 training origins, the development folds and
   the forecast frame from the raw files. Origin hashes are in `training_origin_manifest.csv` (checked by `final_retrain.py`).
2. Development folds: `code/model/exp_runner.py <variant>` writes out-of-fold predictions. The registered variants include
   `cs02_fix1000` and `hp17_fix1000`; seeds are 20260807/08/09.
3. TabPFN-TS: `code/model/g10_neural/neural_runner.py` (folds) and `neural_testwindow.py --context 365|730` (forecast window,
   about 13-18 minutes on one A100). Deterministic: two seeds gave identical predictions (max difference 0.0 on 179,760 rows).
4. Final retrain (the exact command, run from `code/model`):
   ```bash
   python g8_final/final_retrain.py <proj_root> <new_out_dir> --recipe hp17 \
       --blend-file <tabpfn_mix_test.csv> --blend-weight 0.2 --seeds 20260807,20260808,20260809 --n-jobs 16
   ```
   It refuses to write into a folder that already has a manifest, refuses a blend file whose Ids differ from the forecast frame,
   and refuses a recipe that differs from the registered variant.
5. End-term: `code/endterm/endterm_exp.py` (5 repeats x 7 sizes x 3 methods x 2 models + columns) and `make_plots.py`;
   `test_coreset.py` checks that S lies in [2, 6].

Software used for the final retrain: Python 3.11.18, LightGBM 4.7.0, numpy 1.25.2, pandas 2.3.3, scikit-learn 1.7.2
(also in `final/final_manifest.json`). LightGBM with 16 threads reproduces the same boosters for the same data and seed. We
checked this by comparing the final retrain with rehearsal 2 (see `docs/FREEZE_AUDIT.md`, section 4).

## Wording scan
```bash
grep -rniE "state of the art|leaderboard|public score|rank(ed|ing)" report README.md docs notebook
```
The scan should return only this block. Files under `code/` are kept byte-identical to the code that ran, so their hashes
still match the run records.
