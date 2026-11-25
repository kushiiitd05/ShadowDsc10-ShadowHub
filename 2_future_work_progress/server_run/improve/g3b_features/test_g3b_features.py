from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import g3b_features as gf
from feature_panel import build_panel
from synth_g3b import GAP_HUBS, synthetic_raw, synthetic_target_frame

DAY = pd.Timedelta(days=1)


# ---------------------------------------------------------------- independent pandas references
def _obs(raw: pd.DataFrame, hub: int, lo: pd.Timestamp, hi: pd.Timestamp) -> pd.DataFrame:
    """Rows of one hub dated lo..hi inclusive."""
    return raw[(raw["HubID"] == hub) & (raw["Date"] >= lo) & (raw["Date"] <= hi)]


def _logs(rows: pd.DataFrame) -> np.ndarray:
    ok = rows[(rows["IsOpen"] == 1) & (rows["OrderVolume"] > 0)]
    return np.log1p(ok["OrderVolume"].to_numpy(float))


def _feat(raw, origin, groups=gf.ALL_GROUPS, horizon=42):
    return gf.add_features(synthetic_target_frame(raw, origin, horizon), raw, origin, groups)


def _row(out: pd.DataFrame, hub: int, date: pd.Timestamp) -> pd.Series:
    sel = out[(out["HubID"] == hub) & (out["Date"] == date)]
    assert len(sel) == 1
    return sel.iloc[0]


def _same(got: float, expect: float, tol: float = 1e-5) -> bool:
    return (np.isnan(got) and np.isnan(expect)) or got == pytest.approx(expect, abs=tol)


# ---------------------------------------------------------------- (A) gap_aware
@pytest.mark.parametrize("window", gf.GAP_WINDOWS)
def test_gap_window_stats_match_manual_on_observed_open_days(raw, origin, window) -> None:
    out = _feat(raw, origin, ("gap_aware",), horizon=2)
    for hub in (1, 2, 5, 9):       # no gap, but ~1% rows missing and Sunday/random closures
        rows = _obs(raw, hub, origin - (window - 1) * DAY, origin)
        logs = _logs(rows)
        r = out[out["HubID"] == hub].iloc[0]
        assert r[f"g3b_gap_n{window}"] == len(logs)
        assert r[f"g3b_gap_cov{window}"] == pytest.approx(len(rows) / window)
        assert r[f"g3b_gap_mean{window}"] == pytest.approx(logs.mean(), abs=1e-5)
        assert r[f"g3b_gap_median{window}"] == pytest.approx(np.median(logs), abs=1e-5)


def test_missing_dates_are_not_zeros(origin) -> None:
    # hub 1: constant 100 orders, then every other row removed -> mean must stay log1p(100), not be diluted
    dates = pd.date_range("2014-01-01", "2014-10-02", freq="D")
    rows = pd.DataFrame({"HubID": 1, "Date": dates, "OrderVolume": 100.0, "IsOpen": 1, "PromoActive": 0})
    rows = pd.concat([rows[rows["Date"] < "2014-08-01"], rows[rows["Date"] >= "2014-08-01"].iloc[::2]])
    out = gf.add_features(synthetic_target_frame(rows, origin, 1), rows, origin, ("gap_aware",))
    assert out["g3b_gap_mean28"].iloc[0] == pytest.approx(np.log1p(100.0), abs=1e-6)
    assert out["g3b_gap_cov28"].iloc[0] == pytest.approx(14 / 28, abs=0.04)


def test_low_coverage_falls_back_to_hub_expanding_stats() -> None:
    raw = synthetic_raw()
    origin = pd.Timestamp("2015-02-05")        # 184-day gap sits inside the 91/182/365-day windows of hubs 3, 4; 28-day window is clean
    out = _feat(raw, origin, ("gap_aware",), horizon=1)
    hub = GAP_HUBS[0]
    hist = _obs(raw, hub, pd.Timestamp("2000-01-01"), origin)
    expanding = _logs(hist)
    r = out[out["HubID"] == hub].iloc[0]
    for w in (91, 182, 365):
        assert r[f"g3b_gap_cov{w}"] < gf.MIN_COVERAGE
        assert r[f"g3b_gap_mean{w}"] == pytest.approx(expanding.mean(), abs=1e-5)
        assert r[f"g3b_gap_median{w}"] == pytest.approx(np.median(expanding), abs=1e-5)
        assert r[f"g3b_gap_n{w}"] <= r[f"g3b_gap_cov{w}"] * w + 1e-3   # the count stays the honest observed count
    rows28 = _obs(raw, hub, origin - 27 * DAY, origin)
    assert r["g3b_gap_cov28"] == 1.0 and r["g3b_gap_mean28"] == pytest.approx(_logs(rows28).mean(), abs=1e-5)
    other = out[out["HubID"] == 1].iloc[0]       # a hub without a gap keeps its own window values
    assert other["g3b_gap_cov365"] > 0.9


def test_hub_without_any_history_falls_back_to_group_median(raw) -> None:
    origin = pd.Timestamp("2014-04-01")
    extra = raw[(raw["HubID"] == 5)].assign(HubID=99)          # same format/tier as hub 5, but first row only after origin
    extra = extra[extra["Date"] > origin]
    full = pd.concat([raw, extra], ignore_index=True)
    frame = synthetic_target_frame(full, origin, 1)
    out = gf.add_features(frame, build_panel(full), origin, ("gap_aware",))
    got = out[out["HubID"] == 99].iloc[0]
    same = [h for h in raw["HubID"].unique() if (raw[raw["HubID"] == h][["HubFormat", "AssortmentTier"]].iloc[0].to_numpy()
                                                  == raw[raw["HubID"] == 5][["HubFormat", "AssortmentTier"]].iloc[0].to_numpy()).all()]
    medians = [np.median(_logs(_obs(raw, h, origin - 27 * DAY, origin))) for h in same]
    assert got["g3b_gap_n28"] == 0 and got["g3b_gap_cov28"] == 0
    assert got["g3b_gap_median28"] == pytest.approx(np.median(medians), abs=1e-5)


def test_lag364_value_flag_and_closed_days(raw, origin) -> None:
    out = _feat(raw, origin, ("gap_aware",), horizon=14)
    checked = {"value": 0, "closed": 0, "unobserved": 0}
    for hub in (1, 2, 3, 6):
        for date in pd.date_range(origin + DAY, periods=14, freq="D"):
            r = _row(out, hub, date)
            src = raw[(raw["HubID"] == hub) & (raw["Date"] == date - 364 * DAY)]
            if src.empty:
                assert r["g3b_gap_lag364_avail"] == 0 and np.isnan(r["g3b_gap_lag364"])
                checked["unobserved"] += 1
            elif src.iloc[0]["IsOpen"] == 1 and src.iloc[0]["OrderVolume"] > 0:
                assert r["g3b_gap_lag364_avail"] == 1 and r["g3b_gap_lag364"] == pytest.approx(np.log1p(src.iloc[0]["OrderVolume"]), abs=1e-5)
                checked["value"] += 1
            else:                                              # observed but closed: flag says "seen", value stays missing
                assert r["g3b_gap_lag364_avail"] == 1 and np.isnan(r["g3b_gap_lag364"])
                checked["closed"] += 1
    assert checked["value"] > 20 and checked["closed"] > 0


def test_gap_hub_lag364_unavailable_inside_gap() -> None:
    raw = synthetic_raw()
    origin = pd.Timestamp("2015-06-20")          # targets 2015-06-21.. -> d-364 falls in the 2014-06/07 gap boundary
    out = _feat(raw, origin, ("gap_aware",), horizon=42)
    r = _row(out, GAP_HUBS[0], pd.Timestamp("2015-07-15"))     # d-364 = 2014-07-16, inside the gap
    assert r["g3b_gap_lag364_avail"] == 0 and np.isnan(r["g3b_gap_lag364"])


# ---------------------------------------------------------------- (B) year_ago
def test_year_ago_smoothed_value_is_mean_of_three_same_weekday_days(raw, origin) -> None:
    out = _feat(raw, origin, ("year_ago",), horizon=21)
    seen_n = set()
    for hub in (1, 2, 4, 5, 6, 7, 8, 9):
        for date in pd.date_range(origin + DAY, periods=21, freq="D"):
            logs = []
            for k in (357, 364, 371):
                logs += list(_logs(raw[(raw["HubID"] == hub) & (raw["Date"] == date - k * DAY)]))
            r = _row(out, hub, date)
            assert r["g3b_ya_1y_smooth_n"] == len(logs)
            assert _same(r["g3b_ya_1y_smooth"], np.mean(logs) if logs else np.nan)
            seen_n.add(len(logs))
    assert 3 in seen_n and len(seen_n) > 1


@pytest.mark.parametrize("label,shift,origin_s", [("1y", 364, "2014-10-02"), ("2y", 728, "2015-08-01")])
def test_year_ago_window_features_match_manual(raw, label, shift, origin_s) -> None:
    origin = pd.Timestamp(origin_s)
    out = _feat(raw, origin, ("year_ago",), horizon=3)
    for hub in (1, 2, 5, 8):
        win = _obs(raw, hub, origin + DAY - shift * DAY, origin + 42 * DAY - shift * DAY)
        prev = _obs(raw, hub, origin - shift * DAY - 41 * DAY, origin - shift * DAY)
        r = out[out["HubID"] == hub].iloc[0]
        assert r[f"g3b_ya_{label}_win_mean"] == pytest.approx(_logs(win).mean(), abs=1e-5)
        expect_closed = int(((win["IsOpen"] == 0) | (win["OrderVolume"] == 0)).sum())
        assert r[f"g3b_ya_{label}_win_closed"] == expect_closed
        assert r[f"g3b_ya_{label}_win_promo"] == int((win["PromoActive"] == 1).sum())
        assert r[f"g3b_ya_{label}_win_cov"] == pytest.approx(len(win) / 42)
        assert r[f"g3b_ya_{label}_uplift"] == pytest.approx(_logs(win).mean() - _logs(prev).mean(), abs=1e-5)


def test_two_years_ago_is_nan_when_history_too_short(raw) -> None:
    out = _feat(raw, pd.Timestamp("2014-10-02"), ("year_ago",), horizon=3)     # origin - 728 + 1 = 2012-10-... before panel start
    for col in out.columns:
        if col.startswith("g3b_ya_2y_") and not col.endswith(("_cov", "_n")):
            assert out[col].isna().all(), col
    assert (out["g3b_ya_2y_win_cov"] == 0).all()


def test_year_ago_gap_hub_has_no_window_stats_when_year_ago_window_is_gap() -> None:
    raw = synthetic_raw()
    origin = pd.Timestamp("2015-07-20")        # year-ago window 2014-07-22..2014-09-01 lies entirely inside the gap of hubs 3,4
    out = _feat(raw, origin, ("year_ago",), horizon=2)
    r = out[out["HubID"] == GAP_HUBS[0]].iloc[0]
    assert r["g3b_ya_1y_win_cov"] == 0 and np.isnan(r["g3b_ya_1y_win_mean"])
    assert np.isnan(r["g3b_ya_1y_win_closed"]) and np.isnan(r["g3b_ya_1y_uplift"])


# ---------------------------------------------------------------- (C) peer_corr
def test_peer_correlation_matches_pandas_pairwise_pearson() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(size=(12, 365))
    y[:, rng.random(y.shape[1]) < 0.15] = np.nan
    y[4, :200] = np.nan                                     # hub 4 overlaps others on < 165 days
    corr = gf.peer_correlation(y, min_overlap=120)
    ref = pd.DataFrame(y.T).corr(min_periods=120).to_numpy().copy()
    np.fill_diagonal(ref, np.nan)
    np.fill_diagonal(corr, np.nan)
    assert np.allclose(corr, ref, equal_nan=True, atol=1e-9)


def test_peer_correlation_enforces_min_overlap() -> None:
    rng = np.random.default_rng(4)
    y = rng.normal(size=(3, 300))
    y[1, :200] = np.nan                                     # overlap(0,1) = 100 < 120 ; overlap(0,2) = 300
    corr = gf.peer_correlation(y, min_overlap=120)
    assert np.isnan(corr[0, 1]) and np.isnan(corr[1, 0]) and np.isfinite(corr[0, 2])
    assert np.isnan(gf.peer_correlation(y, min_overlap=101)[0, 1])         # exactly 100 overlapping days: the threshold is inclusive
    assert np.isfinite(gf.peer_correlation(y, min_overlap=100)[0, 1])


def test_top_peers_orders_by_correlation_and_pads_with_minus_one() -> None:
    corr = np.array([[np.nan, .9, .2, np.nan], [.9, np.nan, .5, -.3], [.2, .5, np.nan, np.nan], [np.nan, -.3, np.nan, np.nan]])
    idx, val = gf.top_peers(corr, 3)
    assert list(idx[0]) == [1, 2, -1] and list(idx[1]) == [0, 2, 3]
    assert np.allclose(val[1], [.9, .5, -.3]) and np.isnan(val[0, 2]) and (idx[3, 1:] == -1).all()


def test_peer_features_find_planted_peer_and_match_manual_levels() -> None:
    rng = np.random.default_rng(5)
    raw = synthetic_raw(n_hubs=10, n_days=900)
    # hub 2 = scaled copy of hub 1's daily signal (plus tiny noise): its best peer must be hub 1
    hub1 = raw[raw["HubID"] == 1].set_index("Date")
    clone = hub1.copy()
    clone["OrderVolume"] = (hub1["OrderVolume"] * 1.5 * rng.lognormal(0, 0.01, len(hub1))).round()
    clone["HubID"] = 2
    raw = pd.concat([raw[raw["HubID"] != 2], clone.reset_index()], ignore_index=True)
    origin = pd.Timestamp("2014-10-02")
    panel = build_panel(raw).upto(origin)
    y = gf._log_open(panel)
    idx, val = gf.top_peers(gf.peer_correlation(y[:, -365:], 120), gf.PEER_K)
    h1, h2 = int(np.searchsorted(panel.hubs, 1)), int(np.searchsorted(panel.hubs, 2))
    assert idx[h2, 0] == h1 and idx[h1, 0] == h2 and val[h2, 0] > 0.95
    out = gf.add_features(synthetic_target_frame(raw, origin, 1), raw, origin, ("peer_corr",))
    for hub_i in (h1, 5):
        r = out[out["HubID"] == panel.hubs[hub_i]].iloc[0]
        mask = idx[hub_i] >= 0
        assert r["g3b_peer_n"] == mask.sum() == gf.PEER_K
        assert r["g3b_peer_corr_mean"] == pytest.approx(val[hub_i][mask].mean(), abs=1e-5)
        for w in gf.PEER_LEVEL_WINDOWS:
            lvl = np.array([_logs(_obs(raw, h, origin - (w - 1) * DAY, origin)).mean() if len(_logs(_obs(raw, h, origin - (w - 1) * DAY, origin))) else np.nan
                            for h in panel.hubs])
            assert _same(r[f"g3b_peer_rel{w}"], np.nanmean(lvl[idx[hub_i][mask]]) - lvl[hub_i])


def test_peer_uplift_is_mean_of_peers_year_ago_uplift(raw) -> None:
    origin = pd.Timestamp("2014-10-02")
    panel = build_panel(raw).upto(origin)
    y = gf._log_open(panel)
    idx, _ = gf.top_peers(gf.peer_correlation(y[:, -365:], 120), gf.PEER_K)
    out = gf.add_features(synthetic_target_frame(raw, origin, 1), raw, origin, ("peer_corr", "year_ago"))
    uplift = out.set_index("HubID")["g3b_ya_1y_uplift"]
    for hub_i in (0, 6, 10):
        peers = panel.hubs[idx[hub_i][idx[hub_i] >= 0]]
        got = out.loc[out["HubID"] == panel.hubs[hub_i], "g3b_peer_uplift_1y"].iloc[0]
        assert got == pytest.approx(np.nanmean(uplift.loc[peers].to_numpy()), abs=1e-5)


def test_peer_group_works_without_year_ago_group(raw, origin) -> None:
    out = _feat(raw, origin, ("peer_corr",), horizon=1)
    assert out["g3b_peer_uplift_1y"].notna().any()
    assert not any(c.startswith("g3b_ya_") or c.startswith("g3b_gap_") for c in out.columns)


# ---------------------------------------------------------------- API / hygiene
def test_add_features_returns_new_frame_prefixed_float32(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin, 7)
    before = frame.copy()
    out = gf.add_features(frame, raw, origin)
    pd.testing.assert_frame_equal(frame, before)
    assert out is not frame and len(out) == len(frame) and (out.index == frame.index).all()
    new = [c for c in out.columns if c not in frame.columns]
    assert new and all(c.startswith("g3b_") for c in new)
    assert (out[new].dtypes == np.float32).all()
    for group in gf.ALL_GROUPS:
        assert gf.select_group_columns(new, (group,)), group
    assert set(gf.select_group_columns(new, gf.ALL_GROUPS)) == set(new)


def test_group_selection_is_exclusive(raw, origin) -> None:
    for group in gf.ALL_GROUPS:
        new = [c for c in _feat(raw, origin, (group,), 2).columns if c.startswith("g3b_")]
        assert new and set(new) == set(gf.select_group_columns(new, (group,)))


def test_rejects_clash_unknown_group_and_missing_columns(raw, origin) -> None:
    out = _feat(raw, origin, horizon=2)
    with pytest.raises(ValueError, match="already has"):
        gf.add_features(out, raw, origin)
    with pytest.raises(ValueError, match="unknown"):
        _feat(raw, origin, ("nope",))
    with pytest.raises(ValueError, match="lacks"):
        gf.add_features(synthetic_target_frame(raw, origin, 2).drop(columns="Date"), raw, origin)


def test_no_promo_column_drops_promo_counts(origin) -> None:
    raw = synthetic_raw().drop(columns=["PromoActive"])
    out = gf.add_features(synthetic_target_frame(raw, origin, 2).drop(columns="PromoActive"), raw, origin, ("year_ago",))
    assert not any("promo" in c for c in out.columns if c.startswith("g3b_"))
    assert "g3b_ya_1y_win_mean" in out.columns


def test_no_group_columns_uses_global_fallback(origin) -> None:
    raw = synthetic_raw().drop(columns=["HubFormat", "AssortmentTier"])
    out = gf.add_features(synthetic_target_frame(raw, origin, 2), raw, origin, ("gap_aware",))
    assert out["g3b_gap_mean365"].notna().all()


def test_unknown_hub_gets_nan_not_error(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin, 2)
    frame.loc[frame["HubID"] == 1, "HubID"] = 12345
    out = gf.add_features(frame, raw, origin)
    assert out.loc[out["HubID"] == 12345, [c for c in out.columns if c.startswith("g3b_")]].isna().all().all()


def test_narrow_integer_hub_dtype_joins_correctly(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin, 5)
    narrow = frame.astype({"HubID": "int16"})
    pd.testing.assert_frame_equal(gf.add_features(frame, raw, origin).filter(like="g3b_"),
                                  gf.add_features(narrow, raw, origin).filter(like="g3b_"))


# ---------------------------------------------------------------- leakage
def _scramble_after(raw: pd.DataFrame, origin: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = raw.copy()
    after = out["Date"] > origin
    n = int(after.sum())
    out.loc[after, "OrderVolume"] = rng.permutation(out.loc[after, "OrderVolume"].to_numpy()) * 7 + 1
    out.loc[after, "IsOpen"] = rng.integers(0, 2, n)
    out.loc[after, "PromoActive"] = rng.integers(0, 2, n)
    return out


@pytest.mark.parametrize("origin_s", ["2013-11-20", "2014-10-02", "2015-02-14"])
def test_features_unchanged_when_future_data_scrambled(raw, origin_s) -> None:
    origin = pd.Timestamp(origin_s)
    frame = synthetic_target_frame(raw, origin)
    base = gf.add_features(frame, raw, origin)
    pd.testing.assert_frame_equal(base, gf.add_features(frame, _scramble_after(raw, origin), origin))
    pd.testing.assert_frame_equal(base, gf.add_features(frame, raw[raw["Date"] <= origin], origin))
    # prebuilt full panel (future columns present until upto cuts them) must agree too
    pd.testing.assert_frame_equal(base, gf.add_features(frame, build_panel(_scramble_after(raw, origin)), origin))


def test_features_do_change_when_history_changes(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin)
    edited = raw.copy()
    last_week = (edited["Date"] > origin - 7 * DAY) & (edited["HubID"] == 1) & (edited["Date"] <= origin)
    edited.loc[last_week, ["OrderVolume", "IsOpen"]] = [5000.0, 1]
    year_ago = (edited["Date"] > origin - 364 * DAY) & (edited["Date"] <= origin - 350 * DAY) & (edited["HubID"] == 2)
    edited.loc[year_ago, ["OrderVolume", "IsOpen"]] = [5000.0, 1]
    a, b = gf.add_features(frame, raw, origin), gf.add_features(frame, edited, origin)
    for hub in (1, 2):
        assert not a[a["HubID"] == hub].filter(like="g3b_").equals(b[b["HubID"] == hub].filter(like="g3b_"))


def test_multi_origin_equals_single_and_is_leak_free(raw) -> None:
    o1, o2 = pd.Timestamp("2014-02-01"), pd.Timestamp("2014-10-02")
    frame = pd.concat([synthetic_target_frame(raw, o, 6).assign(_origin=o) for o in (o1, o2)], ignore_index=True)
    frame = frame.iloc[np.random.default_rng(0).permutation(len(frame))]            # shuffled rows, non-trivial index
    multi = gf.add_features_multi(frame, raw, "_origin")
    assert (multi.index == frame.index).all()
    for o in (o1, o2):
        part = frame[frame["_origin"] == o].drop(columns="_origin")
        pd.testing.assert_frame_equal(multi[multi["_origin"] == o].drop(columns="_origin"), gf.add_features(part, raw, o))
    scrambled = gf.add_features_multi(frame, _scramble_after(raw, o1), "_origin")        # data after the EARLIER origin
    pd.testing.assert_frame_equal(multi[multi["_origin"] == o1], scrambled[scrambled["_origin"] == o1])


def test_multi_rejects_duplicate_index_and_accepts_prebuilt_panel(raw) -> None:
    o = pd.Timestamp("2014-03-01")
    frame = synthetic_target_frame(raw, o, 3).assign(_origin=o)
    with pytest.raises(ValueError, match="unique"):
        gf.add_features_multi(pd.concat([frame, frame]), raw, "_origin")
    frame.index = frame.index + 100
    out = gf.add_features_multi(frame, build_panel(raw), "_origin")
    assert (out.index == frame.index).all() and out.filter(like="g3b_").shape[1] > 20
