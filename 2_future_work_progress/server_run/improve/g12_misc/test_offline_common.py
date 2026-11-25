from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import offline_common as oc


def _frame(n_hubs=6, days=10, seed=0, fold=2, start="2015-02-14"):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days)
    f = pd.MultiIndex.from_product([range(1, n_hubs + 1), dates], names=["HubID", "Date"]).to_frame(index=False)
    f["fold"] = fold
    f["IsOpen"] = 1
    f["y"] = rng.integers(50, 500, len(f)).astype(float)
    f["pred"] = f["y"] * np.exp(rng.normal(0, 0.1, len(f)))
    return f


def test_residual_ignores_closed_and_zero_rows():
    f = _frame()
    f.loc[0, "IsOpen"] = 0
    f.loc[0, "y"] = 0
    f.loc[1, "y"] = 0
    r = oc.residual(f)
    assert np.isnan(r[0]) and np.isnan(r[1]) and r[2:].notna().all()
    np.testing.assert_allclose(r[2], np.log1p(f.y[2]) - np.log1p(f.pred[2]))


def test_apply_shift_only_touches_correctable_rows_and_is_pure():
    f = _frame()
    f.loc[3, "IsOpen"] = 0
    f.loc[3, "pred"] = 0.0
    f.loc[4, "pred"] = 0.0
    before = f.copy()
    out = oc.apply_shift(f, np.full(len(f), 0.2))
    pd.testing.assert_frame_equal(f, before)
    assert out[3] == 0 and out[4] == 0
    np.testing.assert_allclose(np.log1p(out[0]) - np.log1p(f.pred[0]), 0.2)
    np.testing.assert_allclose(oc.apply_shift(f, np.zeros(len(f))), f.pred)


def test_scores_excl_open_zero():
    f = _frame()
    f.loc[0, "y"] = 0.0
    s = oc.scores(f)
    assert s["excl_open0"] < s["all"]


def test_hub_mean_residual_window_and_min_days():
    f = _frame(days=20)
    f["pred"] = np.expm1(np.log1p(f["y"]) - 0.3)               # constant log residual of +0.3
    out = oc.hub_mean_residual(f, 14, pd.Timestamp("2015-03-05"))
    np.testing.assert_allclose(out.to_numpy(), 0.3, atol=1e-9)
    assert oc.hub_mean_residual(f, 14, pd.Timestamp("2015-02-15"), min_days=3).isna().all()   # only 2 days in window


def test_ols_recovers_beta_and_skips_nan():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 2))
    e = x @ np.array([0.6, -0.2]) + rng.normal(scale=0.01, size=500)
    x2 = x.copy()
    x2[0, 0] = np.nan
    np.testing.assert_allclose(oc.ols_through_origin(x2, e), [0.6, -0.2], atol=0.02)
    assert (oc.ols_through_origin(np.full((5, 1), np.nan), np.ones(5)) == 0).all()


def test_leave_fold_out_never_uses_target_fold():
    designs = {2: (np.ones((10, 1)), np.full(10, 1.0)), 3: (np.ones((10, 1)), np.full(10, 2.0)), 4: (np.ones((10, 1)), np.full(10, 3.0))}
    assert oc.leave_fold_out(designs, 2)[0] == pytest.approx(2.5)
    assert oc.leave_fold_out(designs, 4)[0] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        oc.leave_fold_out({2: designs[2]}, 2)


def test_evaluate_correction_gains_when_signal_exists_and_zero_when_not():
    rng = np.random.default_rng(2)
    frames, designs = {}, {}
    for fold in (2, 3, 4):
        f = _frame(n_hubs=40, days=10, seed=fold, fold=fold)
        signal = np.repeat(rng.normal(0, 0.2, 40), 10)          # hub-level persistent residual
        f["pred"] = f["y"] * np.exp(-signal)
        frames[fold] = f
        designs[fold] = (signal + rng.normal(0, 0.02, len(f))).reshape(-1, 1)
    out = oc.evaluate_correction(frames, designs)
    assert (out["gain_all"] > 0.1).all() and (out["new_all"] < out["base_all"]).all()
    noise = {f: rng.normal(size=(len(frames[f]), 1)) for f in frames}
    out2 = oc.evaluate_correction(frames, noise)
    assert out2["gain_all"].abs().max() < 0.01
    missing = oc.evaluate_correction(frames, {2: designs[2], 3: designs[3]}, eval_folds=(2, 3, 4))
    assert np.isnan(missing.loc[missing.fold == 4, "new_all"]).all()


def test_gate_summary():
    pf = pd.DataFrame({"fold": [2, 3, 4], "gain_all": [0.003, 0.0025, 0.0], "gain_excl": [0.003, 0.001, 0.0]})
    g = oc.gate_summary(pf)
    assert g["n_pass"] == 2 and g["gate_all"] and not g["gate_excl"]
    assert not oc.gate_summary(pf.iloc[:2])["gate_all"]


def test_previous_fold_and_load_oof_duplicates(tmp_path):
    assert oc.previous_fold(3) == 2 and oc.previous_fold(1) is None and oc.previous_fold(5) is None
    f = _frame().rename(columns={"y": "ActualOrderVolume"})
    p = tmp_path / "o.csv.gz"
    f.to_csv(p, index=False)
    assert len(oc.load_oof(p)) == len(f)
    pd.concat([f, f.iloc[:1]]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        oc.load_oof(p)
