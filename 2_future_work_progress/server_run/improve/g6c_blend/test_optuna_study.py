import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import optuna_study as st  # noqa: E402


class FakeTrial:
    def __init__(self):
        self.calls = []

    def suggest_float(self, name, low, high, *, log=False, step=None):
        self.calls.append(("float", name, low, high, log, step))
        return low

    def suggest_int(self, name, low, high):
        self.calls.append(("int", name, low, high))
        return low

    def suggest_categorical(self, name, choices):
        self.calls.append(("cat", name, tuple(choices)))
        return choices[0]


# ---- search space, controls, config
def test_search_space_has_no_blend_weight():
    t = FakeTrial()
    p = st.suggest_params(t)
    assert t.calls == [
        ("float", "feature_fraction", 0.10, 0.35, False, None),
        ("cat", "num_leaves", (63, 127, 255, 383, 511)),
        ("int", "min_data_in_leaf", 60, 400),
        ("float", "lambda_l2", 1.0, 100.0, True, None),
    ]
    assert set(p) == set(st.PARAM_NAMES) and st.BLEND_W == 0.2


def test_controls_are_cs02_then_hp17():
    assert [c["name"] for c in st.CONTROLS] == ["cs02", "hp17"]
    assert st.CONTROLS[0]["params"] == {"feature_fraction": 0.2, "num_leaves": 63, "min_data_in_leaf": 80, "lambda_l2": 1.0}
    assert st.CONTROLS[1]["params"] == {"feature_fraction": 0.15, "num_leaves": 255, "min_data_in_leaf": 160, "lambda_l2": 24.1}


def test_config_kwargs_fixed_recipe_and_no_mutation():
    p = dict(st.CONTROLS[0]["params"]); before = dict(p)
    assert st.config_kwargs(p) == {"colsample": 0.2, "num_leaves": 63, "min_child": 80, "reg_lambda": 1.0, "subsample_freq": 1,
                                   "extra_trees": False, "te": True, "target": "log", "fixed_rounds": 1000}
    assert p == before


# ---- blend and join
def test_blend_log_space_and_endpoints():
    l, t = np.array([100.0, 0.0, 50.0]), np.array([400.0, 0.0, 50.0])
    assert np.allclose(st.blend_log(l, t, 0.0), l) and np.allclose(st.blend_log(l, t, 1.0), t)
    mid = st.blend_log(l, t, 0.5)
    assert mid[1] == 0.0 and mid[0] == pytest.approx(np.expm1(0.5 * np.log1p(100) + 0.5 * np.log1p(400)))


def _frames(n=6):
    keys = pd.DataFrame({"HubID": np.arange(n), "Date": pd.Timestamp("2015-04-01")})
    valid = keys.assign(ActualOrderVolume=np.arange(n) * 10.0, IsOpen=1)
    tab = pd.DataFrame({"fold": 2, "HubID": keys["HubID"], "Date": keys["Date"].dt.strftime("%Y-%m-%d"),
                        "ActualOrderVolume": valid["ActualOrderVolume"], "pred": np.arange(n) * 11.0})
    return valid, tab


def test_align_tabpfn_order_and_rejections():
    valid, tab = _frames()
    out = st.align_tabpfn(valid.iloc[::-1].reset_index(drop=True), tab.sample(frac=1, random_state=1), 2, expected_rows=6)
    assert np.allclose(out, (np.arange(6) * 11.0)[::-1])
    with pytest.raises(ValueError, match="rows"):
        st.align_tabpfn(valid, tab.iloc[:-1], 2, expected_rows=6)
    bad = tab.copy(); bad.loc[0, "HubID"] = 99
    with pytest.raises(ValueError, match="match"):
        st.align_tabpfn(valid, bad, 2, expected_rows=6)
    bad = tab.copy(); bad.loc[2, "ActualOrderVolume"] += 1
    with pytest.raises(ValueError, match="actual"):
        st.align_tabpfn(valid, bad, 2, expected_rows=6)


def test_blend_scores_pairs_and_w0_equals_lgbm():
    a = np.array([10.0, 20.0, 0.0, 30.0]); o = np.array([1, 1, 0, 1])
    l, t = np.array([9.0, 22.0, 0.0, 28.0]), np.array([11.0, 19.0, 0.0, 31.0])
    s = st.blend_scores(a, o, l, t, 0.2)
    assert set(s) == {"lgbm_rmsle_all", "lgbm_excl_open0", "blend_rmsle_all", "blend_excl_open0"} and s["blend_rmsle_all"] != s["lgbm_rmsle_all"]
    assert st.blend_scores(a, o, l, t, 0.0)["blend_rmsle_all"] == pytest.approx(s["lgbm_rmsle_all"])


# ---- completeness of the objective file
def _tab(n=4, folds=(2, 3), drop=0):
    parts = [pd.DataFrame({"fold": f, "HubID": np.arange(n), "Date": "2015-04-01", "ActualOrderVolume": 10.0, "pred": 9.0}).iloc[: n - drop]
             for f in folds]
    return pd.concat(parts, ignore_index=True)


def test_require_complete():
    st.require_complete(_tab(), (2, 3), expected_rows=4)
    with pytest.raises(ValueError, match="incomplete"):
        st.require_complete(_tab(drop=1), (2, 3), expected_rows=4)
    with pytest.raises(ValueError, match="incomplete"):
        st.require_complete(_tab(folds=(2,)), (2, 3), expected_rows=4)
    st.require_complete(_tab(folds=(2, 3, 4)), (2,), expected_rows=4)


# ---- objective and rows
def _fs(l, b, lx=0.1, bx=0.1, sec=10.0, a=None, ax=None):
    return {"lgbm_rmsle_all": l, "lgbm_excl_open0": lx, "blend_rmsle_all": b, "blend_excl_open0": bx, "fit_seconds": sec,
            "alt_blend_rmsle_all": a, "alt_blend_excl_open0": ax}


def test_objective_mean_over_folds_and_validation():
    assert st.objective_value({2: _fs(0.12, 0.11), 3: _fs(0.10, 0.09)}) == pytest.approx(0.10)
    with pytest.raises(ValueError):
        st.objective_value({2: _fs(0.1, 0.1), 3: _fs(0.1, 0.1), 4: _fs(0.1, 0.1)})
    with pytest.raises(ValueError):
        st.objective_value({2: _fs(0.1, float("nan")), 3: _fs(0.1, 0.1)})
    assert st.objective_value({2: _fs(0.12, 0.11)}, folds=(2,)) == pytest.approx(0.11)


def test_trial_row_schema_is_study_independent_and_csv_pools(tmp_path):
    p = dict(st.CONTROLS[0]["params"])
    fr = {2: _fs(0.12, 0.11, 0.113, 0.108, 60.0, 0.112, 0.109), 3: _fs(0.10, 0.09, 0.1, 0.09, 80.0)}
    a = st.trial_row(3, p, fr, study="A", sampler_seed=1, model_seed=7, n_jobs=32, peak_rss_gb=13.0, state="COMPLETE",
                     finished_at="t", tag="cs02")
    b = st.trial_row(4, p, fr, study="B", sampler_seed=2, model_seed=7, n_jobs=24, peak_rss_gb=13.0, state="COMPLETE",
                     finished_at="t", tag="")
    assert list(a) == list(b) == list(st.csv_columns())
    assert a["objective"] == pytest.approx(0.10) and a["lgbm_mean"] == pytest.approx(0.11)
    assert a["fold2_alt_blend_rmsle_all"] == 0.112 and a["fold3_alt_blend_rmsle_all"] == ""   # missing alt stays empty
    assert a["total_fit_seconds"] == 140.0 and a["study"] == "A" and b["sampler_seed"] == 2 and a["model_seed"] == 7
    path = tmp_path / "t.csv"
    st.append_csv_row(path, a); st.append_csv_row(path, b)
    rows = list(csv.DictReader(path.open()))
    assert [r["study"] for r in rows] == ["A", "B"] and list(rows[0]) == list(st.csv_columns())


def _r(n, obj, ff, nl, mdl, l2, study="A", state="COMPLETE"):
    return {"trial": n, "study": study, "state": state, "feature_fraction": ff, "num_leaves": nl, "min_data_in_leaf": mdl,
            "lambda_l2": l2, "objective": obj}


def test_pooled_finalists_exclude_controls_duplicates_and_failed():
    rows = [_r(0, 0.1160, 0.2, 63, 80, 1.0), _r(1, 0.1120, 0.15, 255, 160, 24.1),
            _r(0, 0.1160, 0.2, 63, 80, 1.0, "B"), _r(1, 0.1120, 0.15, 255, 160, 24.1, "B"),     # B's own controls
            _r(2, 0.1100, 0.18, 383, 200, 30.0), _r(5, 0.1100, 0.18, 383, 200, 30.0, "B"),      # duplicate across studies
            _r(3, 0.1105, 0.25, 127, 90, 3.0, "B"), _r(4, 0.1110, 0.30, 511, 300, 50.0),
            _r(6, 0.1115, 0.12, 63, 400, 80.0, "B"), _r(7, 0.1118, 0.11, 127, 61, 2.0),
            _r(8, 0.1119, 0.13, 255, 70, 5.0, "B"), _r(9, 0.0900, 0.3, 63, 61, 2.0, state="FAIL")]
    top = st.select_finalists(rows, 5)
    assert [(r["study"], r["trial"]) for r in top] == [("A", 2), ("B", 3), ("A", 4), ("B", 6), ("A", 7)]


def test_variant_line_name_includes_study_tag_when_given():
    row = _r(17, 0.1, 0.1812345, 383, 211, 12.3456789)
    line = st.variant_source_line(row, "a")
    assert line.strip().startswith('"g6c_a_t17": Config(') and "blend" not in line
    for tok in ("colsample=0.1812345", "num_leaves=383", "min_child=211", "reg_lambda=12.3456789", "fixed_rounds=1000", "subsample_freq=1"):
        assert tok in line


# ---- run control
def test_choose_n_jobs_24_while_our_other_runs_or_rehearsal_alive():
    assert st.choose_n_jobs(["1 ../env311/bin/python exp_runner.py x"]) == 24
    assert st.choose_n_jobs(["1 python g5_pergroup/g5_runner.py"]) == 24
    assert st.choose_n_jobs(["1 python g8_final/final_retrain.py --dry"]) == 24
    assert st.choose_n_jobs(["1 python distill_train.py"]) == 32 and st.choose_n_jobs([]) == 32


def test_stop_reason():
    assert st.DEADLINE == datetime(2026, 11, 26, 10, 0)
    assert st.stop_reason(datetime(2026, 11, 26, 10, 0), 3) == "deadline reached"
    assert st.stop_reason(datetime(2026, 11, 25, 23, 0), 40) == "trial budget reached"
    assert st.stop_reason(datetime(2026, 11, 25, 23, 0), 39) is None


# ---- cli
def test_parse_args_defaults():
    a = st.parse_args([])
    assert a.study_name == "g6c_blend" and a.sampler_seed == 20260807 and a.model_seed == 20260807 and a.n_trials == 40
    assert a.stop_at == datetime(2026, 11, 26, 10, 0) and a.n_jobs is None and a.folds_objective == (2, 3)
    assert str(a.storage) == "../optuna_g6c.db" and str(a.tabpfn_oof) == "../tabpfn_oof_mix.csv.gz"
    assert str(a.out_dir) == "../exp_out_G6c" and str(a.proj_root) == "../proj" and a.tabpfn_oof_alt is None


def test_parse_args_second_study_overrides():
    a = st.parse_args(["--study-name", "g6c_blend_b", "--storage", "../optuna_g6c_b.db", "--sampler-seed", "20260808",
                       "--model-seed", "20260807", "--tabpfn-oof", "x.csv.gz", "--tabpfn-oof-alt", "y.csv.gz", "--n-trials", "12",
                       "--stop-at", "2026-11-26T09:30", "--n-jobs", "24", "--out-dir", "../exp_out_G6c_b", "--folds-objective", "3,2"])
    assert (a.study_name, a.sampler_seed, a.model_seed, a.n_trials, a.n_jobs) == ("g6c_blend_b", 20260808, 20260807, 12, 24)
    assert a.stop_at == datetime(2026, 11, 26, 9, 30) and a.folds_objective == (2, 3) and str(a.tabpfn_oof_alt) == "y.csv.gz"


@pytest.mark.parametrize("bad", [["--folds-objective", "2,4"], ["--folds-objective", "1,2,3"], ["--folds-objective", ""],
                                 ["--n-jobs", "0"], ["--n-trials", "0"], ["--stop-at", "tomorrow"]])
def test_parse_args_rejects_bad_values(bad):
    with pytest.raises(SystemExit):
        st.parse_args(bad)


# ---- optuna glue
def test_study_enqueues_two_controls_resumes_and_seed_is_a_parameter(tmp_path):
    def run(seed, name):
        s = st.build_study(f"sqlite:///{tmp_path / (name + str(seed) + '.db')}", study_name=name, sampler_seed=seed)
        s.optimize(lambda t: st.suggest_params(t)["feature_fraction"], n_trials=14)
        return s
    a, b, a2 = run(20260807, "one"), run(20260808, "two"), run(20260807, "three")
    assert a.study_name == "one"
    assert [t.params for t in a.trials[:2]] == [c["params"] for c in st.CONTROLS]
    assert a.trials[12].params == a2.trials[12].params and a.trials[12].params != b.trials[12].params
    assert "blend_w" not in a.trials[5].params
    again = st.build_study(f"sqlite:///{tmp_path / 'one20260807.db'}", study_name="one", sampler_seed=20260807)
    assert len(again.trials) == 14 and st.count_complete(again) == 14


def test_fail_stale_running(tmp_path):
    import optuna

    s = st.build_study(f"sqlite:///{tmp_path / 's.db'}")
    s.ask()
    assert st.fail_stale_running(s) == 1 and s.trials[0].state == optuna.trial.TrialState.FAIL
