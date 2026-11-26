"""Write the two notebooks (then execute them with nbconvert; see README).

    notebook/01_endterm_results.ipynb        every table and figure of report/main.pdf, the gate and the leakage check
    notebook/02_run_frozen_forecaster.ipynb  load the frozen models and rebuild the submitted forecast file

Both start with a setup cell that works inside the repository, on a fresh machine and on Colab (it clones the repository
with a sparse checkout when the files are not found).
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
REPO = "https://github.com/kushiiitd05/ShadowDsc10-ShadowHub.git"


def setup_cell(packages: tuple[str, ...]) -> nbf.NotebookNode:
    return code(f"""# Setup: inside the repository, on a fresh machine or on Colab (then it clones only what is needed, about 200 MB)
import os, subprocess, sys
from pathlib import Path

def find_root():
    here = Path.cwd()
    for c in (here, here.parent, here / '3_endsem', here / 'ShadowDsc10-ShadowHub' / '3_endsem'):
        if (c / 'final' / 'final_manifest.json').exists():
            return c.resolve()
    return None

ROOT = find_root()
if ROOT is None:
    subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', '{REPO}'], check=True)
    subprocess.run(['git', '-C', 'ShadowDsc10-ShadowHub', 'sparse-checkout', 'set', '3_endsem', '1_midterm/data/raw'], check=True)
    ROOT = find_root()
for module in {packages!r}:
    try:
        __import__(module)
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', module], check=True)
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / 'code'), str(ROOT / 'code' / 'model')]
print('found the 3_endsem folder; running from it')""")


READING_GUIDE = """**Two tracks, three model names.**
* **Track A, the course rubric:** $x^*$ = LightGBM recipe `cs02` (63 leaves, feature fraction 0.2, row bagging) on one fixed
  time split; ridge is the second model; every uniform, hub-stratified and coreset summary is fitted with the same settings.
* **Track B, the forecaster we submit:** `hp17` (the same LightGBM, tuned) x 3 seeds, blended 0.8 / 0.2 in log space with
  TabPFN-TS (365- and 730-day contexts, 0.10 each). Its forecast window (from 20 Jun 2015) has no labels, so it has
  development-fold scores only.
* **Why Track B is not coreset-ed:** TabPFN-TS is pretrained and reads each hub's history at prediction time; it has no training
  loss and no rows to weight, so sensitivities are not defined for it. The rubric also needs the same model refitted for
  5 repeats x 7 sizes x 3 methods."""

results_cells = [
    md("""# CSE558 End-term, group DSC 10 (1 of 2): coresets for 42-day hub demand, and the frozen forecaster

This notebook rebuilds every number, table and figure of `report/main.pdf` from the files in this folder. It fits no model:
the heavy runs (37 origin tables, 260 end-term fits, TabPFN-TS on a GPU, the final retrain) were done on a server and their
outputs are saved in `results/`, `artifacts/` and `final/`. It runs in under a minute on a laptop or on Colab.
Notebook 2 (`02_run_frozen_forecaster.ipynb`) re-runs the frozen models themselves.

""" + READING_GUIDE + """

**Sections**
1. Data, split, metric and the pipeline
2. Full model $x^*$ and ridge; which features the trees use
3. Plot 1: rows (uniform, hub-stratified, sensitivity coreset) and columns
4. Plot 2 and the speed-up table
5. Coreset: $S$, $\\alpha$, Plots 3-5 and slopes
6. Leakage check: every decision re-taken without fold 4
7. Track B: searches, finalists and the experiment ledger
8. Freeze audit: the gate recomputed from saved out-of-fold predictions
9. The frozen artifact: hashes and hard checks"""),
    setup_cell(("pandas", "matplotlib")),
    code("""import json
import numpy as np, pandas as pd
from IPython.display import Image, display
RES, FIG, ART, FIN = ROOT/'results', ROOT/'figures', ROOT/'artifacts', ROOT/'final'
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30); pd.set_option('display.max_colwidth', 90)
print(sys.version.split()[0], 'pandas', pd.__version__, 'numpy', np.__version__)"""),
    md("""## 1. Data, split, metric and the pipeline
One row = (forecast origin, hub, target day). 1,115 hubs, 42-day horizon, 342 feature columns built from history up to the
origin only. End-term split: train = origins whose label window ends by 2015-05-08; test = 2015-05-09 .. 2015-06-19
(46,830 rows). Metric: RMSLE = RMSE of `log1p`; $R^2$ on the log scale; closed days kept in test with prediction 0.

**Why log1p:** volumes are right-skewed and errors should be relative (a miss of 500 orders matters more at a small hub).
**Why keep closed days:** the forecast file must cover them; they are predicted as exactly 0, so their error is 0, which
inflates $R^2$. We therefore also report RMSLE without open-but-zero rows."""),
    code("""raw = ROOT.parent / '1_midterm' / 'data' / 'raw'
train = pd.read_csv(raw / 'orders_train.csv', usecols=['HubID', 'Date'])
test = pd.read_csv(raw / 'orders_test.csv')
print('raw training rows', len(train), 'hubs', train.HubID.nunique(), 'dates', train.Date.min(), '..', train.Date.max())
print('forecast frame rows', len(test), 'closed rows', int((test.IsOpen == 0).sum()), 'dates', test.Date.min(), '..', test.Date.max())
print('end-term training cells: 1,133,640 rows x 342 columns =', f'{1_133_640*342:,}')
display(Image(FIG / 'fig_pipeline.png'))"""),
    md("## 2. Full model $x^*$ (LightGBM) and the second model (ridge)"),
    code("""et = pd.read_csv(RES / 'endterm_run1_results.csv')
et[et.method == 'full'][['model', 'test_rmsle', 'test_rmsle_excl_open0', 'test_r2', 't_train']].round(4)"""),
    md("""LightGBM is 36% lower in RMSLE than ridge: demand depends on interactions (weekday x promotion x format, holidays,
data gaps) that trees capture directly. Ridge is kept because coreset theory is exact for it, so it checks the method.

**Why these features.** Every column is known at the forecast origin (history up to the origin, plus the calendar, promotion,
opening and holiday flags supplied in advance). Split gain of the frozen boosters by group (written by
`code/feature_importance.py`): the hub x weekday x promotion profile carries most of the gain, then the recent rolling levels.
Gain shares credit among correlated columns, so read groups, not single columns."""),
    code("""display(pd.read_csv(RES / 'feature_group_importance.csv'))
display(Image(FIG / 'fig_feature_groups.png'))"""),
    md("## 3. Plot 1: test RMSLE against summary size (median of 5 repeats)"),
    code("""rows = et[et.method != 'full'].pivot_table(index=['model', 'm'], columns='method', values='test_rmsle', aggfunc='median')
rows['coreset_minus_uniform'] = rows['sensitivity'] - rows['uniform']
display(rows.round(4))
display(Image(FIG / 'fig_plot1_rows.png'))"""),
    md("""**Reading.** For LightGBM the coreset beats uniform from 5k to 200k rows (by 0.0032 at 50k) and ties at 2k, where a tree
with min child 80 cannot use the weight variance. Hub-stratified is no better than uniform. For ridge all methods agree:
a 5k-row ridge is already within 0.003 of the full fit.

**Columns** (rows fixed at 200k): importance-chosen vs random columns. Trees need original columns, so no random projection."""),
    code("""display(Image(FIG / 'fig_plot1_columns.png'))"""),
    md("## 4. Plot 2: time, and the speed-up table"),
    code("""sp = pd.read_csv(RES / 'endterm_run1_speedup.csv')
display(sp[sp.model == 'lgbm'].round(4))
display(Image(FIG / 'fig_plot2_time.png'))"""),
    md("""Uniform is always faster (no build); the coreset is always more accurate from 5k rows. At the chosen size, 50k rows:
coreset 0.1070 at 2.75x, uniform 0.1102 at 5.18x, full 0.1022. **Limitation:** timings come from a shared server
(load 10-30); an exclusive rerun is still open."""),
    md("## 5. Coreset: total sensitivity, roughness, Plots 3-5"),
    code("""print(open(RES / 'endterm_run1_slopes_and_S.txt').read())
display(Image(FIG / 'fig_hist_sensitivity.png'))
display(Image(FIG / 'fig_plot345_lgbm.png'))
display(Image(FIG / 'fig_plot345_ridge.png'))"""),
    md("""$S$ must lie in $[2,6]$ for five rough models and one group (the denominator is the weighted **sum** of losses, so each
rough model contributes exactly 1). Measured 2.47 and 2.12. For ridge the coefficient error (Plot 5) falls at 0.66 times the rate
of the loss gap (theory: about 0.5). For trees the ratio is 0.33 and there is no guarantee; we call the method heuristic
sensitivity-motivated importance sampling."""),
    md("""## 6. Leakage check: would any decision change without fold 4?
The end-term test window is also development fold 4. No fold-4 label enters an end-term fit, a sampling probability, a column
choice, the number of rounds or ridge's lambda, but fold-4 scores were part of earlier recipe decisions. `code/leakage_check.py`
re-takes each of them with fold 4 removed, from saved files only."""),
    code("""import leakage_check
check = leakage_check.run(ROOT)
display(check)
print('every decision unchanged without fold 4:', bool(check.same_decision.all()))"""),
    md("""## 7. Track B: what the searches found
Gate (fixed before the runs): mean gain over folds 2-4 >= 0.0024 (twice the noise 0.0012), >= 2 of 3 folds improve, same sign on
seeds 20260807/08/09, still true without open-but-zero rows, fold 5 checked."""),
    code("""hp = pd.read_csv(RES / 'hp_search_20configs.csv')
display(hp.head(25))
op = pd.read_csv(RES / 'g6c_pooled_80_trials.csv').sort_values('objective')
print('Optuna trials', len(op), ' best', round(op.objective.iloc[0], 4), ' top-10 span', round(op.objective.iloc[9] - op.objective.iloc[0], 4))
display(op[['study', 'trial', 'feature_fraction', 'num_leaves', 'min_data_in_leaf', 'lambda_l2', 'objective']].head(8).round(4))
print('Five Optuna finalists on all five folds (seed 07) against hp17 on the same seed:')
display(pd.read_csv(RES / 'g6c_finalists_5folds.csv'))
display(Image(FIG / 'fig_optuna.png'))
display(Image(FIG / 'fig_tabpfn_grid.png'))"""),
    md("The full experiment ledger, negatives included, with the reason each idea failed:"),
    code("""print(open(RES / 'experiment_ledger.md').read())"""),
    md("""## 8. Freeze audit: the gate recomputed from saved out-of-fold predictions
`code/freeze_audit.py` reads the out-of-fold files of the baseline (`cs02`, lr 0.025, 1000 fixed rounds, 3 seeds), `hp17` (same
setting, 3 seeds) and TabPFN-TS (365- and 730-day contexts), rebuilds the blend and recomputes every gate number."""),
    code("""import freeze_audit as fa
panel, full_rows = fa.build_panel()
folds, gains = fa.gate_tables(panel)
display(gains.round(4))
for c in ('hp17', 'candidate', 'fallback_hp17_t365'):
    v = fa.gate_verdict(gains, c); print(c, 'PASSES' if v['passes'] else 'FAILS', v['checks'])
display(folds[folds.arm == '3seed'][['fold', 'rows', 'cs02_all', 'hp17_all', 'candidate_all', 'candidate_excl0']].round(4))
display(Image(FIG / 'fig_gate_gains.png'))
display(Image(FIG / 'fig_model_ladder.png'))"""),
    code("""mix = fa.context_mix_check(panel)
print(json.dumps(mix, indent=1))"""),
    md("## 9. The frozen artifact"),
    code("""man = json.loads((FIN / 'final_manifest.json').read_text())
run = man['_run']
print('recipe', run['recipe'], run['config']['variant'], '| blend weight', run['blend_weight'], run['blend']['external']['space'])
print('training origins', run['training_origins'], '| last label', run['last_label_date'], '| forecast origin', run['forecast_origin'])
print('versions', run['versions'])
for k, v in man.items():
    if k.startswith('lightgbm_'): print(k, v['sha256'])
print('forecast file sha256', run['submission_sha256'])
print(json.dumps(json.loads((FIN / 'freeze_checks.json').read_text()), indent=1))"""),
    md("""**Limits kept with the result:** `hp17` was the best of 19 configs and its gain shrank on fresh seeds; the context mix was
defined after seeing 365 and 730; the weight 0.2 sits in a flat curve; fold 4 is also the end-term test window (no decision changes
without it, section 6); folds 1, 4 and 5 are diagnostics, not untouched test sets; the real forecast window has no labels in this
project, so no real test score exists."""),
]

frozen_cells = [
    md("""# CSE558 End-term, group DSC 10 (2 of 2): run the frozen forecaster

Loads the three frozen LightGBM boosters and their exact input matrix, adds the TabPFN-TS forecast, and rebuilds the submitted
forecast file. No training, no server, no feature pipeline: about 10 seconds on a laptop or on Colab.

**Formula** (log1p space, closed hub-days exactly 0):
$$\\hat y = \\exp\\Big(0.8\\cdot\\tfrac13\\textstyle\\sum_{s}\\log(1+\\hat y^{\\,\\mathrm{hp17}}_s) + 0.2\\cdot\\log(1+\\hat y^{\\,\\mathrm{TabPFN\\text{-}TS\\,mix}})\\Big)-1$$
with seeds $s \\in \\{20260807, 20260808, 20260809\\}$; the TabPFN-TS input is the equal log mix of its 365- and 730-day
contexts, so each context carries 0.10 of the final log forecast and each LightGBM seed about 0.267.

| File | What it is |
|---|---|
| `final/models/lightgbm_s*.txt.gz` | the three boosters (`hp17`: 255 leaves, feature fraction 0.15, min child 160, L2 24.1, lr 0.025, 1000 trees) |
| `final/forecast_matrix_open.parquet` | their input: the 40,282 open hub-days of the forecast window x 342 columns (+ Id) |
| `artifacts/test_window/tabpfn_mix_test.csv` | TabPFN-TS forecast (computed on a GPU; deterministic) |
| `final/submission_final.csv` | the frozen forecast file this notebook must reproduce |"""),
    setup_cell(("pandas", "pyarrow", "lightgbm", "matplotlib")),
    code("""import numpy as np, pandas as pd, lightgbm
import predict_frozen as pf
print('lightgbm', lightgbm.__version__)
x_open = pd.read_parquet(ROOT / 'final' / 'forecast_matrix_open.parquet')
print('input matrix', x_open.shape, '(Id + feature columns); first columns:', list(x_open.columns[1:6]))"""),
    md("## 1. Each frozen booster reproduces its saved forecast"),
    code("""table, seeds = pf.frozen_forecast(ROOT)
for s, p in seeds.items():
    saved = np.load(ROOT / 'final' / f'test_pred_lightgbm_s{s}.npy')
    print(f'seed {s}: max |log1p difference| vs the saved forecast = {np.max(np.abs(np.log1p(p) - np.log1p(saved))):.1e}')"""),
    md("## 2. The blend reproduces the submitted file"),
    code("""frozen = pd.read_csv(ROOT / 'final' / 'submission_final.csv')
same_ids = bool((table.Id.values == frozen.Id.values).all())
diff = float(np.max(np.abs(np.log1p(table.OrderVolume) - np.log1p(frozen.OrderVolume))))
print('rows', len(table), '| same Id order:', same_ids, '| max |log1p difference|:', diff)
assert same_ids and diff < 1e-9
table.head()"""),
    md("## 3. What the forecast looks like: three hubs, last 120 days of history and the 42 forecast days"),
    code("""import matplotlib.pyplot as plt
raw = ROOT.parent / '1_midterm' / 'data' / 'raw'
hist = pd.read_csv(raw / 'orders_train.csv', usecols=['HubID', 'Date', 'OrderVolume'], parse_dates=['Date'])
frame = pd.read_csv(raw / 'orders_test.csv', usecols=['Id', 'HubID', 'Date'], parse_dates=['Date']).merge(table, on='Id')
fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
for ax, hub in zip(axes, (1, 262, 817)):
    h = hist[(hist.HubID == hub) & (hist.Date > hist.Date.max() - pd.Timedelta(days=120))]
    f = frame[frame.HubID == hub]
    ax.plot(h.Date, h.OrderVolume, color='0.45', lw=1, label='history (actual)')
    ax.plot(f.Date, f.OrderVolume, color='#1B36C4', lw=1.4, label='frozen forecast')
    ax.set_ylabel(f'hub {hub}')
axes[0].legend(loc='upper left'); axes[-1].set_xlabel('date')
plt.tight_layout(); plt.show()"""),
    md("""Closed days are exactly 0 in the forecast. The forecast window has no labels in this project, so there is no score for it:
the model's quality is measured on the development folds (notebook 1, section 8)."""),
]

for name, cells in (("01_endterm_results.ipynb", results_cells), ("02_run_frozen_forecaster.ipynb", frozen_cells)):
    nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    out = ROOT / "notebook" / name
    out.parent.mkdir(exist_ok=True)
    nbf.write(nb, out)
    print(out)
