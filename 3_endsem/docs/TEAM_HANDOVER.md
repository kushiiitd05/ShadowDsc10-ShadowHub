# Team handover: finish the CSE558 end-term (group DSC 10)

Paste this whole file into your agent, or read it yourself, before touching anything. State as of **26 Nov 2026**. The
repository is `github.com/kushiiitd05/ShadowDsc10-ShadowHub`, and everything for the end-term is in `3_endsem/` on `main`.

**Deadlines:** submission **8 Dec 2026**. Presentations **9-10 Dec 2026**: 7 minutes of talk, then 3 minutes of
questions, with 6-9 slides.

## 1. What is finished (do not redo)
- **The study.** Coreset experiments (Track A) and the forecaster (Track B) are run, audited and frozen.
  - The forecaster has decision **GO**, with one stated condition: a retrain is not bit-identical.
  - No more experiments are planned except the timing rerun in section 4.
- **The report.** 9 pages (`report/main.pdf`, source `report/main.tex`). It includes:
  - a reading guide (two tracks);
  - the leakage check;
  - why these features;
  - why the blend is not coreset-ed;
  - every negative result.
- **Two executed notebooks.** Both run on a laptop or Colab and clone the repository themselves if needed.
  - `notebook/01_endterm_results.ipynb` rebuilds every table and figure, the gate and the leakage check.
  - `notebook/02_run_frozen_forecaster.ipynb` loads the frozen models and rebuilds the submitted file exactly.
- **The frozen forecaster** is in `final/`:
  - three LightGBM boosters, gzipped;
  - their exact input matrix;
  - the forecast file;
  - the SHA-256 manifest;
  - the freeze record and the checks.
- **The slide plan.** `docs/PRESENTATION_BRIEF.md` covers 9 slides, slide by slide: the claim, figure, notation, script,
  reasoning and source file, plus a question sheet and a "do not say" list.
- **Tests.** `python -m pytest -q code` from `3_endsem/` runs all the unit tests.

## 2. Read in this order
1. `docs/PRESENTATION_BRIEF.md`, section 0 ("the confusions we cleared"), then the rest.
2. `report/main.pdf`, especially the reading guide in section 1, the leakage box and section 8 (the questions).
3. `README.md`: the folder map and the headline numbers.
4. `results/experiment_ledger.md` and `docs/FREEZE_AUDIT.md`, only if you need to defend a choice in the questions.

## 3. The model in one paragraph (say it the same way everywhere)

**Track A, the rubric.**
- $x^\*$ is LightGBM `cs02`, trained on a fixed time split: train = labels up to 8 May 2015; test = 9 May-19 Jun 2015
  (46,830 rows, labelled).
- Ridge is the second model.
- The sensitivity coreset at 50k rows gives 0.1070 against 0.1102 for uniform and 0.1022 for the full model. It is 2.75x
  faster.

**Track B, what we submit.**
- `hp17` (the tuned LightGBM) x 3 seeds, blended 0.8/0.2 in log space with TabPFN-TS. TabPFN-TS is the equal mix of its
  365- and 730-day contexts, 0.10 each.
- It gains +0.0045 on development folds 2-4 and passes a gate fixed in advance, also with fold 4 removed.
- Its window, 20 Jun-31 Jul 2015, has no labels, so it has no test score.

## 4. What is left, and who
| # | Task | Owner | How |
|---|---|---|---|
| 1 | Build the deck (9 slides) | teammates | follow `docs/PRESENTATION_BRIEF.md` exactly; figures from `figures/*.png`; numbers only from `results/` or the report |
| 2 | Contribution lines in the report header (4 members) | each member | edit the table in `report/main.tex` (about line 130), rebuild with `cd report && tectonic main.tex`, then regenerate the manifest (section 5) |
| 3 | Exclusive-server timing rerun, 1-2 Dec | Kush (server access) | rerun the end-term timing on an idle server; if done, update Plot 2, the speed-up table, report section 4 and slides 4 and 8 |
| 4 | Rehearse with a timer (7:00) and the question sheet | all | brief, sections 1 and 4 |
| 5 | Final check before submitting | one person | section 5 below |
| 6 | Submit by 8 Dec | one person | report PDF + repository link, as the course asks |

## 5. Rules if you edit anything in `3_endsem/`
- Do **not** change `final/`, `artifacts/` or `results/`. They are the frozen evidence, and their hashes are recorded.
- After any edit, regenerate the checksum list and verify it:
  ```bash
  cd 3_endsem
  find . -type f ! -name MANIFEST.sha256 ! -name .DS_Store ! -path '*/__pycache__/*' ! -path './.pytest_cache/*' | sed 's|^\./||' | LC_ALL=C sort \
    | xargs shasum -a 256 > MANIFEST.sha256 && shasum -a 256 -c MANIFEST.sha256 | grep -v ': OK$'
  ```
  The second command prints nothing when everything matches.
- Run the wording scan from `docs/REPRODUCIBILITY.md`. It must only hit the "do not say" lists.
- **Never** mention competition platforms, public or leaderboard scores, rankings, "state of the art" or "beat the winner".
  Our numbers are not comparable to anything outside this project.
- **Never** give a score for the 20 Jun-31 Jul window, and never call the 0.1022 test number "unseen".
- No server names, IP addresses, key paths or personal details other than the group table.
- Commit one file per commit with a clear message, and open a pull request to `main` rather than pushing to it directly.

## 6. Contributions recorded so far
**Kush Tokas (2023296), end-term work.** The files are in `3_endsem/`.

| Contribution | Files |
|---|---|
| Model track: more than 40 ideas tested under a gate fixed in advance; the experiment harness | `results/experiment_ledger.md`, `code/model/exp_runner.py`, `code/model/helpers.py` |
| Searches: 20-config random search, 80-trial Optuna on two servers, 5 finalists on all folds | `results/hp_search_20configs.csv`, `results/g6c_pooled_80_trials.csv`, `results/g6c_finalists_5folds.csv`, `code/model/g6c_blend/` |
| TabPFN-TS member (365- and 730-day contexts, GPU) | `code/model/g10_neural/`, `artifacts/test_window/`, `artifacts/oof/tabpfn*` |
| Freeze: independent gate recomputation, audit, final retrain with hashes, fallbacks | `code/freeze_audit.py`, `docs/FREEZE_AUDIT.md`, `code/model/g8_final/`, `final/` |
| End-term coreset runs (Track A): split, x\*, sampling, sensitivities, timing, Plots 1-5 | `code/endterm/`, `results/endterm_run1_*`, `figures/fig_plot*` |
| Leakage check, feature importance, frozen-model re-run anywhere | `code/leakage_check.py`, `code/feature_importance.py`, `code/model/predict_frozen.py`, `final/forecast_matrix_open.parquet` |
| Report, notebooks, slide brief, repository | `report/main.tex`, `notebook/`, `docs/PRESENTATION_BRIEF.md`, `README.md` |

The other members add their own lines here and in the report header.
