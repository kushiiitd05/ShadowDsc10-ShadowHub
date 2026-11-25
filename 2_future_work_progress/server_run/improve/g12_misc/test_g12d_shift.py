from __future__ import annotations

import numpy as np
import pandas as pd

import g12d_shift as d
import offline_common as oc


def _frames(drift=0.2, n_hubs=30, seed=0):
    """Fold 1..4 windows laid out like the real folds; fold residual = hub drift (persistent) + noise."""
    rng = np.random.default_rng(seed)
    hub_drift = rng.normal(0, drift, n_hubs)
    frames = {}
    for fold in (1, 2, 3, 4):
        origin = pd.Timestamp(oc.ORIGINS[fold])
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=42)
        f = pd.MultiIndex.from_product([range(1, n_hubs + 1), dates], names=["HubID", "Date"]).to_frame(index=False)
        f["fold"], f["IsOpen"] = fold, 1
        f["y"] = rng.integers(100, 1000, len(f)).astype(float)
        noise = rng.normal(0, 0.05, len(f))
        f["pred"] = np.expm1(np.log1p(f["y"]) - np.repeat(hub_drift, 42) - noise)
        frames[fold] = f
    return frames, hub_drift


def test_own_recent_residual_uses_previous_fold_only():
    frames, drift = _frames()
    m = d.own_recent_residual(frames, 3, 14)
    assert m.index.tolist() == list(range(1, 31))
    np.testing.assert_allclose(m.to_numpy(), drift, atol=0.06)
    assert d.own_recent_residual(frames, 1, 14) is None and d.own_recent_residual(frames, 5, 14) is None
    # window must end at the previous fold's last day (== this fold's origin)
    assert frames[2]["Date"].max() == pd.Timestamp(oc.ORIGINS[3])


def test_design_shapes_and_horizon_buckets():
    frames, _ = _frames()
    one, three = d.design(frames, 3, 14, False), d.design(frames, 3, 14, True)
    assert one.shape == (len(frames[3]), 1) and three.shape == (len(frames[3]), 3)
    np.testing.assert_allclose(three.sum(axis=1), one[:, 0])           # each row sits in exactly one bucket
    assert d.design(frames, 1, 14, False) is None


def test_persistent_drift_is_corrected_and_fold1_skipped():
    frames, _ = _frames(drift=0.2)
    out = d.run(frames, folds=(1, 2, 3, 4))
    fit = out[(out.variant == "fit") & (out.window == 14)].set_index("fold")
    assert (fit.loc[[2, 3, 4], "gain_all"] > 0.05).all() and np.isnan(fit.loc[1, "new_all"])
    lit = out[(out.variant == "literal") & (out.window == 14)].set_index("fold")
    assert (lit.loc[[2, 3, 4], "gain_all"] > 0.05).all()
    assert (fit.loc[[2, 3, 4], "beta"].map(float) > 0.8).all()


def test_no_persistence_gives_no_gain():
    rng = np.random.default_rng(3)
    frames, _ = _frames(drift=0.0)
    for f in frames.values():                            # independent per-fold hub drift
        f["pred"] = np.expm1(np.log1p(f["y"]) - np.repeat(rng.normal(0, 0.2, 30), 42))
    out = d.run(frames, folds=(2, 3, 4))
    fit = out[(out.variant == "fit") & (out.window == 14)]
    assert fit["gain_all"].abs().max() < 0.01
    lit = out[(out.variant == "literal") & (out.window == 14)]
    assert (lit["gain_all"] < 0).all()                   # beta = 1 on pure noise hurts


def test_inputs_not_mutated():
    frames, _ = _frames()
    before = {k: v.copy() for k, v in frames.items()}
    d.run(frames, folds=(2, 3, 4))
    for k in frames:
        pd.testing.assert_frame_equal(frames[k], before[k])
