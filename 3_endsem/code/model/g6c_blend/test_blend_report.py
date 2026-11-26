import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blend_report as rep  # noqa: E402
import optuna_study as g6c  # noqa: E402


def _data(hubs=6, fold=5):
    rng = np.random.default_rng(0)
    rows = [(h, pd.Timestamp("2014-06-20") + pd.Timedelta(days=d)) for h in range(hubs) for d in range(2)]
    actual = rng.integers(50, 500, len(rows)).astype(float)
    lg = pd.DataFrame(rows, columns=["HubID", "Date"]).assign(IsOpen=1, ActualOrderVolume=actual, pred=actual * 1.1, fold=fold)
    tab = pd.DataFrame({"fold": fold, "HubID": lg["HubID"], "Date": lg["Date"].dt.strftime("%Y-%m-%d"),
                        "ActualOrderVolume": actual, "pred": actual * 0.9})
    return lg, tab


def test_evaluate_fold_blend_and_endpoints():
    lg, tab = _data()
    r0 = rep.evaluate_fold(lg, tab, 5, 0.0)
    r1 = rep.evaluate_fold(lg, tab, 5, 1.0)
    assert r0["rmsle_all"] == pytest.approx(np.sqrt(np.mean((np.log1p(lg["pred"]) - np.log1p(lg["ActualOrderVolume"])) ** 2)))
    assert r1["rmsle_all"] == pytest.approx(np.sqrt(np.mean((np.log1p(tab["pred"]) - np.log1p(lg["ActualOrderVolume"])) ** 2)))
    assert r0["n_rows"] == len(lg) == 12


def test_evaluate_fold_restricts_to_tabpfn_hubs():
    lg, tab = _data(hubs=6)
    tab = tab[tab["HubID"] < 4]
    r = rep.evaluate_fold(lg, tab, 5, 0.2)
    assert r["n_rows"] == 8 and r["n_hubs"] == 4


def test_evaluate_fold_rejects_hubs_missing_from_lgbm_and_other_folds():
    lg, tab = _data()
    with pytest.raises(ValueError):
        rep.evaluate_fold(lg[lg["HubID"] < 5], tab, 5, 0.2)
    with pytest.raises(ValueError):
        rep.evaluate_fold(lg, tab, 4, 0.2)


def test_markdown_table_formats_rows():
    out = rep.markdown_table(["a", "b"], [["x", 0.123456], ["y", None]], digits=4)
    assert out.splitlines()[0] == "| a | b |" and "0.1235" in out and "| y | - |" in out


def test_load_trials_pools_files(tmp_path):
    cols = list(rep.g6c.csv_columns())
    for name, study, t in (("a.csv", "A", 3), ("b.csv", "B", 4)):
        row = {c: "" for c in cols}
        row.update(study=study, trial=t, state="COMPLETE", objective=0.11, feature_fraction=0.2, num_leaves=63,
                   min_data_in_leaf=80, lambda_l2=1.0)
        pd.DataFrame([row]).to_csv(tmp_path / name, index=False)
    rows = rep.load_trials([tmp_path / "a.csv", tmp_path / "b.csv"])
    assert [(r["study"], r["trial"]) for r in rows] == [("A", 3), ("B", 4)]


def test_model_rows_use_study_trials_for_folds_2_3_and_oof_for_others():
    lg1, tab1 = _data(fold=1)
    lg4, tab4 = _data(fold=4)
    tab = pd.concat([tab1, tab4], ignore_index=True)
    lg = pd.concat([lg1, lg4], ignore_index=True)
    out = rep.score_folds(lg, tab, folds=(1, 4), w=0.2)
    assert set(out) == {1, 4} and out[1]["rmsle_all"] == pytest.approx(out[4]["rmsle_all"])


def test_study_tag():
    assert rep.study_tag("g6c_blend") == "a" and rep.study_tag("g6c_blend_b") == "b"
