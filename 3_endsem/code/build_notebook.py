"""Write notebook/DSC10_endterm.ipynb (then execute it with nbconvert; see README)."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("""# CSE558 End-term, group DSC 10: coresets for 42-day hub demand, and the frozen forecaster

This notebook rebuilds every number, table and figure of `report/main.pdf` from the files in this folder. It fits no model:
the heavy runs (37 origin tables, 260 end-term fits, TabPFN-TS on a GPU, the final retrain) were done on a server and their
outputs are saved in `results/`, `artifacts/` and `final/`. Running it top to bottom takes well under a minute on a laptop.

**Sections**
1. Data, split and metric
2. Full model $x^*$ and ridge
3. Plot 1: rows (uniform, hub-stratified, sensitivity coreset) and columns
4. Plot 2 and the speed-up table
5. Coreset: $S$, $\\alpha$, Plots 3-5 and slopes
6. Improving $x^*$: search results and the experiment ledger
7. Freeze audit: the gate recomputed from saved out-of-fold predictions
8. The frozen artifact: hashes and hard checks"""),
    code("""import json, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display
ROOT = Path.cwd().parent if Path.cwd().name == 'notebook' else Path.cwd()
sys.path.insert(0, str(ROOT / 'code'))
RES, FIG, ART, FIN = ROOT/'results', ROOT/'figures', ROOT/'artifacts', ROOT/'final'
pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)
print(sys.version.split()[0], 'pandas', pd.__version__, 'numpy', np.__version__)"""),
    md("""## 1. Data, split and metric
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
print('forecast frame rows', len(test), 'closed rows', int((test.IsOpen == 0).sum()))
print('end-term training cells: 1,133,640 rows x 342 columns =', f'{1_133_640*342:,}')"""),
    md("## 2. Full model $x^*$ (LightGBM) and the second model (ridge)"),
    code("""et = pd.read_csv(RES / 'endterm_run1_results.csv')
et[et.method == 'full'][['model', 'test_rmsle', 'test_rmsle_excl_open0', 'test_r2', 't_train']].round(4)"""),
    md("""LightGBM is 36% lower in RMSLE than ridge: demand depends on interactions (weekday x promotion x format, holidays,
data gaps) that trees capture directly. Ridge is kept because coreset theory is exact for it, so it checks the method."""),
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
    md("""## 6. Improving $x^*$: what the searches found

Gate (fixed before the runs): mean gain over folds 2-4 >= 0.0024 (twice the noise 0.0012), >= 2 of 3 folds improve, same sign on
seeds 20260807/08/09, still true without open-but-zero rows, fold 5 checked."""),
    code("""hp = pd.read_csv(RES / 'hp_search_20configs.csv')
display(hp.head(25))
op = pd.read_csv(RES / 'g6c_pooled_80_trials.csv').sort_values('objective')
print('Optuna trials', len(op), ' best', round(op.objective.iloc[0], 4), ' top-10 span', round(op.objective.iloc[9] - op.objective.iloc[0], 4))
display(op[['study', 'trial', 'feature_fraction', 'num_leaves', 'min_data_in_leaf', 'lambda_l2', 'objective']].head(8).round(4))
display(Image(FIG / 'fig_optuna.png'))
display(Image(FIG / 'fig_tabpfn_grid.png'))"""),
    md("The full experiment ledger, negatives included, with the reason each idea failed:"),
    code("""print(open(RES / 'experiment_ledger.md').read())"""),
    md("""## 7. Freeze audit: the gate recomputed from saved out-of-fold predictions
`code/freeze_audit.py` reads the out-of-fold files of the baseline (`cs02`, lr 0.025, 1000 fixed rounds, 3 seeds), `hp17` (same
setting, 3 seeds) and TabPFN-TS (365- and 730-day contexts), rebuilds the blend and recomputes every gate number."""),
    code("""import freeze_audit as fa
panel, full_rows = fa.build_panel()
folds, gains = fa.gate_tables(panel)
display(gains.round(4))
for c in ('hp17', 'candidate', 'fallback_hp17_t365'):
    v = fa.gate_verdict(gains, c); print(c, 'PASSES' if v['passes'] else 'FAILS', v['checks'])
display(folds[folds.arm == '3seed'][['fold', 'rows', 'cs02_all', 'hp17_all', 'candidate_all', 'candidate_excl0']].round(4))
display(Image(FIG / 'fig_gate_gains.png'))"""),
    code("""mix = fa.context_mix_check(panel)
print(json.dumps(mix, indent=1))"""),
    md("## 8. The frozen artifact"),
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
defined after seeing 365 and 730; the weight 0.2 sits in a flat curve chosen on folds 2-4 (including fold 4); folds 1, 4 and 5 are
diagnostics, not untouched test sets; the real forecast window has no labels in this project, so no real test score exists."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebook" / "DSC10_endterm.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(out)
