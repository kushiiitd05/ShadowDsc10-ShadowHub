from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import g12e_reconcile as e
import offline_common as oc


def _frame(fold=2, n_hubs=40, days=14, noise=0.0, seed=0):
    """Hubs in 2 formats; truth = hub level + format-day effect; predictions carry an extra hub-day noise the group mean can average out."""
    rng = np.random.default_rng(seed)
    fmt = (np.arange(n_hubs) % 2) + 1
    dates = pd.date_range(pd.Timestamp(oc.ORIGINS[fold]) + pd.Timedelta(days=1), periods=days)
    f = pd.MultiIndex.from_product([range(1, n_hubs + 1), dates], names=["HubID", "Date"]).to_frame(index=False)
    f["fold"], f["IsOpen"] = fold, 1
    level = np.repeat(rng.normal(6, 0.5, n_hubs), days)
    day_fx = np.tile(np.sin(np.arange(days)), n_hubs) * 0.2
    truth = level + day_fx
    f["y"] = np.expm1(truth)
    f["pred"] = np.expm1(truth + rng.normal(0, noise, len(f)))
    meta = pd.DataFrame({"HubID": range(1, n_hubs + 1), "HubFormat": fmt, "AssortmentTier": 1})
    return f, meta


def test_group_day_mean_and_hub_deviation():
    f, meta = _frame(noise=0.0)
    group = e.group_codes(f, meta, "format")
    lp = e.log_pred(f)
    mask = oc.correctable(f)
    gm = e.group_day_mean(lp, mask, group, f["Date"].to_numpy())
    sel = (group == 1) & (f["Date"] == f["Date"].iloc[0]).to_numpy()
    assert gm[sel][0] == pytest.approx(lp[sel].mean())
    dev = e.hub_deviation(f, lp, mask)
    assert pd.Series(dev).groupby(f["HubID"].to_numpy()).mean().abs().max() < 1e-9


def test_masked_rows_excluded_and_nan_design():
    f, meta = _frame()
    f.loc[f.index[:5], "pred"] = 0.0
    f.loc[f.index[5], "IsOpen"] = 0
    x = e.designs_for_fold(f, e.group_codes(f, meta, "format"), "pattern2")
    assert x.shape == (len(f), 2) and np.isnan(x[:6]).all() and np.isfinite(x[6:]).all()
    assert e.designs_for_fold(f, e.group_codes(f, meta, "format"), "level").shape == (len(f), 1)
    with pytest.raises(ValueError):
        e.designs_for_fold(f, e.group_codes(f, meta, "format"), "x")


def test_pattern_shrink_helps_when_noise_is_idiosyncratic_and_level_shrink_is_weak():
    frames, metas = {}, None
    for fold in (1, 2, 3, 4):
        frames[fold], metas = _frame(fold=fold, noise=0.15, seed=fold)
    out = e.run(frames, metas, folds=(1, 2, 3, 4))
    pat = out[(out.variant == "format_pattern") & (out.weights == "clip01")].set_index("fold")
    lev = out[(out.variant == "format_level") & (out.weights == "clip01")].set_index("fold")
    assert (pat["gain_all"] > 0.002).all()
    assert (lev["gain_all"] < pat["gain_all"]).all()                      # level shrink can only use a tiny weight: hubs differ in size
    assert (lev["beta"].map(lambda s: float(s.split(";")[0])) < 0.1).all()


def test_no_gain_when_predictions_are_exact_and_inputs_untouched():
    frames = {f: _frame(fold=f, noise=0.0, seed=f)[0] for f in (1, 2, 3, 4)}
    meta = _frame()[1]
    before = {k: v.copy() for k, v in frames.items()}
    out = e.run(frames, meta, folds=(2, 3))
    assert out["gain_all"].abs().max() < 1e-9
    for k in frames:
        pd.testing.assert_frame_equal(frames[k], before[k])


def test_unknown_grouping():
    f, meta = _frame()
    with pytest.raises(ValueError):
        e.group_codes(f, meta, "nope")
