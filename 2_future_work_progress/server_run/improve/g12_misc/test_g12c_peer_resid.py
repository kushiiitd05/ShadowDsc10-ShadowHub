from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import g12c_peer_resid as c
import offline_common as oc
from feature_panel import build_panel
from synth_g3b import synthetic_raw


def test_group_peer_mean_excludes_self_and_handles_nan():
    values = pd.Series([1.0, 2.0, 3.0, 10.0, np.nan], index=[1, 2, 3, 4, 5])
    groups = pd.Series(["a", "a", "a", "b", "b"], index=[1, 2, 3, 4, 5])
    out = c.group_peer_mean(values, groups)
    assert out[1] == pytest.approx(2.5) and out[2] == pytest.approx(2.0) and out[3] == pytest.approx(1.5)
    assert out[4] != out[4]                      # the only other member (hub 5) has no value
    assert out[5] == pytest.approx(10.0)         # hub 5 itself has no value but its peer does


def test_group_peer_mean_singleton_group_is_nan():
    out = c.group_peer_mean(pd.Series([1.0, 2.0], index=[1, 2]), pd.Series(["a", "b"], index=[1, 2]))
    assert out.isna().all()


def test_corr_peer_mean():
    peers = pd.DataFrame([[2.0, 3.0], [1.0, np.nan], [np.nan, np.nan]], index=[1, 2, 3])
    out = c.corr_peer_mean(pd.Series([1.0, 3.0, np.nan], index=[1, 2, 3]), peers)
    assert out[1] == pytest.approx(3.0) and out[2] == pytest.approx(1.0) and np.isnan(out[3])


def test_corr_peers_use_history_up_to_origin_only():
    raw = synthetic_raw(n_hubs=12, n_days=900)
    panel = build_panel(raw)
    origin = pd.Timestamp("2014-10-02")
    a = c.corr_peers(panel, origin, 3)
    scr = raw.copy()
    scr.loc[scr["Date"] > origin, "OrderVolume"] = np.random.default_rng(0).permutation(scr.loc[scr["Date"] > origin, "OrderVolume"].to_numpy())
    pd.testing.assert_frame_equal(a, c.corr_peers(build_panel(scr), origin, 3))
    assert a.shape == (12, 3) and not np.isin(a.to_numpy(), []).any()
    assert all(int(hub) not in set(row.dropna().astype(int)) for hub, row in a.iterrows())      # no self-peers


def _frames_with_shared_shock(n_hubs=40, seed=0):
    """Hubs in two groups; the residual of a hub = group shock (persistent over folds) + hub noise."""
    rng = np.random.default_rng(seed)
    group = np.arange(1, n_hubs + 1) % 2
    shock = np.array([0.3, -0.3])[group]
    frames = {}
    for fold in (1, 2, 3, 4):
        dates = pd.date_range(pd.Timestamp(oc.ORIGINS[fold]) + pd.Timedelta(days=1), periods=42)
        f = pd.MultiIndex.from_product([range(1, n_hubs + 1), dates], names=["HubID", "Date"]).to_frame(index=False)
        f["fold"], f["IsOpen"] = fold, 1
        f["y"] = rng.integers(100, 1000, len(f)).astype(float)
        hub_noise = np.repeat(rng.normal(0, 0.15, n_hubs), 42)          # independent hub drift (not predictable by peers)
        f["pred"] = np.expm1(np.log1p(f["y"]) - np.repeat(shock, 42) - hub_noise - rng.normal(0, 0.03, len(f)))
        frames[fold] = f
    return frames, pd.Series(group, index=range(1, n_hubs + 1))


def test_group_peers_correct_shared_shock_and_fold1_skipped():
    frames, groups = _frames_with_shared_shock()
    raw = synthetic_raw(n_hubs=40, n_days=900)
    out = c.run(frames, groups, build_panel(raw), folds=(1, 2, 3, 4))
    g = out[(out.variant == "group") & (out.window == 28)].set_index("fold")
    assert (g.loc[[2, 3, 4], "gain_all"] > 0.01).all() and np.isnan(g.loc[1, "new_all"])
    joint = out[(out.variant == "own+group") & (out.window == 28)].set_index("fold")
    assert (joint.loc[[2, 3, 4], "gain_all"] > 0.01).all()
    assert set(out["variant"]) >= {"group", "corr5", "corr10", "own+group", "own+corr5"}


def test_unknown_kind_rejected():
    frames, groups = _frames_with_shared_shock()
    with pytest.raises(ValueError):
        c.peer_feature(frames, 3, 14, "bogus", groups, build_panel(synthetic_raw(n_hubs=40, n_days=900)))
