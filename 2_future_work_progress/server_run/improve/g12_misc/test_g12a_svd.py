from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import g12a_svd as sv
from feature_panel import build_panel
from synth_g3b import synthetic_target_frame


def _lowrank(h=40, t=120, k=3, noise=0.3, miss=0.15, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(h, k)) @ rng.normal(size=(k, t)) + rng.normal(size=(h, 1)) * 2
    y = truth + rng.normal(scale=noise, size=truth.shape)
    holes = rng.random(truth.shape) < miss
    return truth, np.where(holes, np.nan, y)


def test_denoise_recovers_low_rank_better_than_raw():
    truth, y = _lowrank()
    fit = sv.denoise(y, rank=3)
    ok = np.isfinite(y)
    raw_err = np.sqrt(np.mean((y[ok] - truth[ok]) ** 2))
    rec_err = np.sqrt(np.mean((fit["recon"][ok] - truth[ok]) ** 2))
    assert rec_err < 0.7 * raw_err
    hole_err = np.sqrt(np.mean((fit["recon"][~ok] - truth[~ok]) ** 2))        # imputed cells are also close
    assert hole_err < 0.6
    assert fit["U"].shape == (40, 3) and fit["Vt"].shape == (3, 120) and fit["s"].shape == (3,)


def test_denoise_is_pure_and_handles_all_nan_hub():
    _, y = _lowrank()
    y[5] = np.nan
    before = y.copy()
    fit = sv.denoise(y, rank=3)
    np.testing.assert_array_equal(np.isnan(y), np.isnan(before))
    assert np.isfinite(fit["recon"]).all() and fit["mu"][5] == 0.0


def test_procrustes_removes_rotation_and_sign_flips():
    rng = np.random.default_rng(1)
    u_ref, _ = np.linalg.qr(rng.normal(size=(50, 4)))
    q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    u = u_ref @ q
    rot = sv.procrustes_rotation(u, u_ref)
    np.testing.assert_allclose(u @ rot, u_ref, atol=1e-10)
    flipped = u_ref * np.array([1, -1, -1, 1])
    np.testing.assert_allclose(flipped @ sv.procrustes_rotation(flipped, u_ref), u_ref, atol=1e-10)
    np.testing.assert_allclose(rot.T @ rot, np.eye(4), atol=1e-10)


def test_kept_columns_drops_closed_days():
    y = np.full((10, 7), 1.0)
    y[:, 3] = np.nan
    y[:2, 5] = np.nan
    keep = sv.kept_columns(y, 0.3)
    assert not keep[3] and keep[5] and keep.sum() == 6


def _frame(raw, origin, h=3):
    return synthetic_target_frame(raw, origin, h)


def test_add_features_columns_shape_and_purity(raw, origin):
    frame = _frame(raw, origin)
    before = frame.copy()
    out = sv.add_features(frame, raw, origin, rank=5)
    pd.testing.assert_frame_equal(frame, before)
    new = [c for c in out.columns if c.startswith("g12a_")]
    assert len(out) == len(frame) and (out.index == frame.index).all()
    assert {"g12a_load1", "g12a_load5", "g12a_lvl28", "g12a_noise28", "g12a_ya_recon", "g12a_comp1_7d", "g12a_comp3_28d", "g12a_resid_sd91"} <= set(new)
    assert not [c for c in new if c.startswith("g12a_load") and int(c[9:]) > 5]
    assert out["g12a_lvl28"].notna().mean() > 0.9
    # the denoised level lives on the log1p scale of the raw level
    assert 3.0 < out["g12a_lvl28"].median() < 8.0


def test_features_ignore_data_after_origin(raw, origin):
    scrambled = raw.copy()
    after = scrambled["Date"] > origin
    scrambled.loc[after, "OrderVolume"] = 98765.0
    scrambled.loc[after, "IsOpen"] = 1
    frame = _frame(raw, origin, 10)
    pd.testing.assert_frame_equal(sv.add_features(frame, raw, origin, rank=5), sv.add_features(frame, scrambled, origin, rank=5))
    pd.testing.assert_frame_equal(sv.add_features(frame, raw, origin, rank=5), sv.add_features(frame, build_panel(scrambled), origin, rank=5))


def test_year_ago_recon_matches_panel_lookup_direction(raw, origin):
    frame = _frame(raw, origin, 10)
    out = sv.add_features(frame, raw, origin, rank=6)
    sub = out[out["Weekday"] != 7]
    ya = sub["g12a_ya_recon"]
    assert ya.notna().mean() > 0.9
    # year-ago target for a hub close to its raw level
    panel = build_panel(raw)
    val = np.log1p(panel.orders[:, int(panel.dates.get_loc(origin)) + 1 - 364])
    assert np.isfinite(ya.to_numpy()).any() and np.nanmedian(ya.to_numpy()) == pytest.approx(np.nanmedian(val[val > 0]), abs=0.6)


def test_multi_equals_single_and_cache_is_transparent(raw):
    o1, o2 = pd.Timestamp("2014-06-08"), pd.Timestamp("2014-08-17")
    stack = pd.concat([_frame(raw, o, 3).assign(_origin=o) for o in (o1, o2)], ignore_index=True)
    ref = sv.make_reference(build_panel(raw), o1, 5)
    multi = sv.add_features_multi(stack, raw, "_origin", rank=5, reference=ref)
    cache: dict = {}
    cached = sv.add_features_multi(stack, raw, "_origin", rank=5, reference=ref, cache=cache)
    again = sv.add_features_multi(stack, raw, "_origin", rank=5, reference=ref, cache=cache)
    pd.testing.assert_frame_equal(multi, cached)
    pd.testing.assert_frame_equal(multi, again)
    assert len(cache) == 2
    single = sv.add_features(_frame(raw, o2, 3), raw, o2, rank=5, reference=ref)
    cols = [c for c in single.columns if c.startswith("g12a_")]
    pd.testing.assert_frame_equal(multi.loc[multi["_origin"] == o2, cols].reset_index(drop=True), single[cols].reset_index(drop=True))


def test_alignment_makes_loadings_comparable_across_origins(raw):
    o1, o2 = pd.Timestamp("2014-06-08"), pd.Timestamp("2014-08-17")
    panel = build_panel(raw)
    ref = sv.make_reference(panel, o1, 4)
    a = sv.hub_features(panel.upto(o1), 4, ref)[0]
    b = sv.hub_features(panel.upto(o2), 4, ref)[0]
    corr = [np.corrcoef(a[f"g12a_load{j}"], b[f"g12a_load{j}"])[0, 1] for j in (1, 2)]
    assert corr[0] > 0.9                                       # without alignment this correlation has an arbitrary sign
    unaligned = sv.hub_features(panel.upto(o2), 4, None)[0]
    assert abs(np.corrcoef(a["g12a_load1"], unaligned["g12a_load1"])[0, 1]) > 0.9


def test_unknown_hub_and_short_history_give_nan(raw, origin):
    frame = _frame(raw, origin, 3)
    extra = frame.iloc[:3].assign(HubID=9999)
    out = sv.add_features(pd.concat([frame, extra], ignore_index=True), raw, origin, rank=5)
    assert out.tail(3).filter(like="g12a_").isna().all().all()
    early = sv.add_features(_frame(raw, pd.Timestamp("2013-01-06"), 3), raw, pd.Timestamp("2013-01-06"), rank=5)
    assert early.filter(like="g12a_").isna().all().all() or len(early) == len(_frame(raw, pd.Timestamp("2013-01-06"), 3))


def test_clash_and_missing_columns_raise(raw, origin):
    frame = _frame(raw, origin, 2)
    with pytest.raises(ValueError):
        sv.add_features(frame.assign(g12a_lvl7=1.0), raw, origin)
    with pytest.raises(ValueError):
        sv.add_features(frame.drop(columns=["Date"]), raw, origin)
