from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import synthetic_raw, synthetic_target_frame
from extra_features import (
    ALL_GROUPS, add_features, add_features_multi, confirm_duplicates, find_duplicate_columns,
    ridge_trend, select_group_columns,
)
from feature_panel import build_panel


# ---------- ridge trend ----------
def test_ridge_trend_recovers_exact_line() -> None:
    x = np.arange(-90, 1)
    y = (2.0 + 0.05 * x)[None, :]
    slope, level = ridge_trend(y, window=91, ridge_frac=0.0, min_points=10)
    assert slope[0] == pytest.approx(0.05) and level[0] == pytest.approx(2.0)


def test_ridge_shrinks_slope_toward_zero() -> None:
    y = (0.05 * np.arange(-90, 1))[None, :]
    plain, _ = ridge_trend(y, 91, 0.0, 10)
    shrunk, _ = ridge_trend(y, 91, 0.5, 10)
    assert 0 < shrunk[0] < plain[0]


def test_ridge_trend_ignores_nan_and_needs_min_points() -> None:
    y = np.full((2, 91), np.nan)
    y[0, ::2] = 1.0
    y[1, :5] = 3.0
    slope, level = ridge_trend(y, 91, 0.0, 10)
    assert slope[0] == pytest.approx(0.0, abs=1e-9) and level[0] == pytest.approx(1.0)
    assert np.isnan(slope[1]) and np.isnan(level[1])


# ---------- group features ----------
def _features(raw, origin, groups=ALL_GROUPS, horizon=14):
    return add_features(synthetic_target_frame(raw, origin, horizon), raw, origin, groups)


def test_add_features_returns_new_frame_with_prefixed_columns(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin)
    cols_before = list(frame.columns)
    out = add_features(frame, raw, origin)
    assert list(frame.columns) == cols_before and out is not frame
    assert len(out) == len(frame) and (out.index == frame.index).all()
    new = [c for c in out.columns if c not in cols_before]
    assert new and all(c.startswith("g3_") for c in new)
    for group in ALL_GROUPS:
        assert select_group_columns(new, (group,)), group


def test_add_features_rejects_existing_column(raw, origin) -> None:
    out = _features(raw, origin)
    with pytest.raises(ValueError):
        add_features(out, raw, origin)


def test_trend_matches_manual_ols(raw, origin) -> None:
    out = _features(raw, origin, ("trend",))
    hub = 3
    h = raw[(raw["HubID"] == hub) & (raw["Date"] <= origin) & (raw["Date"] > origin - pd.Timedelta(days=91))]
    h = h[(h["IsOpen"] == 1) & (h["OrderVolume"] > 0)]
    x = (h["Date"] - origin).dt.days.to_numpy(float)
    y = np.log1p(h["OrderVolume"].to_numpy(float))
    slope = np.polyfit(x, y, 1)[0]
    got = out.loc[out["HubID"] == hub, "g3_trend_slope91"].iloc[0]
    assert got == pytest.approx(slope * 28 / (1 + 0.02 * 0 + 0.0), rel=0.05)
    assert out["g3_trend_level365"].notna().all()


def test_weekday_promo_median_matches_manual(raw, origin) -> None:
    out = _features(raw, origin, ("weekday_promo",), horizon=7)
    row = out.iloc[0]
    h = raw[(raw["HubID"] == row["HubID"]) & (raw["Date"] <= origin) & (raw["Date"] > origin - pd.Timedelta(days=56))]
    h = h[(h["IsOpen"] == 1) & (h["OrderVolume"] > 0) & (h["Weekday"] == row["Weekday"])]
    sel = h[h["PromoActive"] == row["PromoActive"]]
    expect = np.median(np.log1p(sel["OrderVolume"])) if len(sel) else np.nan
    got = row["g3_dow_promo_median8w"]
    assert (np.isnan(got) and np.isnan(expect)) or got == pytest.approx(expect, rel=1e-5)
    assert row["g3_dow_n8w"] == len(h)


def test_closure_streaks_on_handmade_panel() -> None:
    dates = pd.date_range("2014-01-01", periods=120, freq="D")
    open_ = np.ones(120, int)
    open_[[50, 51, 100, 117, 118, 119]] = 0   # runs {50,51} {100} {117..119}, all inside 90d; origin = last day
    raw = pd.DataFrame({"HubID": 1, "Date": dates, "OrderVolume": np.where(open_ == 1, 10.0, 0.0), "IsOpen": open_,
                        "PromoActive": 0})
    origin = dates[-1]
    out = add_features(synthetic_target_frame(raw, origin, 3), raw, origin, ("streaks",))
    r = out.iloc[0]
    assert r["g3_streak_close_run_len"] == 3 and r["g3_streak_close_since_last"] == 0
    assert r["g3_streak_close_runs90"] == 3 and r["g3_streak_close_days90"] == 6


def test_closure_since_last_when_open_at_origin() -> None:
    dates = pd.date_range("2014-01-01", periods=60, freq="D")
    open_ = np.ones(60, int)
    open_[50] = 0
    raw = pd.DataFrame({"HubID": 1, "Date": dates, "OrderVolume": 5.0 * open_, "IsOpen": open_, "PromoActive": 0})
    out = add_features(synthetic_target_frame(raw, dates[-1], 2), raw, dates[-1], ("streaks",))
    assert out.iloc[0]["g3_streak_close_run_len"] == 0 and out.iloc[0]["g3_streak_close_since_last"] == 9


def test_promo_streaks_on_handmade_panel() -> None:
    dates = pd.date_range("2014-01-01", periods=60, freq="D")
    promo = np.zeros(60, int)
    promo[[10, 55, 56, 57, 58, 59]] = 1
    raw = pd.DataFrame({"HubID": 1, "Date": dates, "OrderVolume": 5.0, "IsOpen": 1, "PromoActive": promo})
    out = add_features(synthetic_target_frame(raw, dates[-1], 2), raw, dates[-1], ("streaks",))
    r = out.iloc[0]
    assert r["g3_streak_promo_run_len"] == 5 and r["g3_streak_promo_since_last"] == 0
    assert r["g3_streak_promo_days28"] == 5 and r["g3_streak_promo_days91"] == 6


def test_region_feature_is_zero_mean_within_group(raw, origin) -> None:
    out = _features(raw, origin, ("region",), horizon=1)
    col = "g3_region_HubFormat_rel91"
    assert col in out.columns
    med = out.groupby(out["HubID"] % 2)[col].median()
    assert np.allclose(med, 0.0, atol=1e-6)


def test_region_absent_without_group_columns(origin) -> None:
    raw = synthetic_raw().drop(columns=["HubFormat", "AssortmentTier"])
    out = add_features(synthetic_target_frame(raw, origin, 2), raw, origin, ("region",))
    assert not select_group_columns(out.columns, ("region",))


def test_no_promo_column_skips_promo_features(origin) -> None:
    raw = synthetic_raw().drop(columns=["PromoActive"])
    frame = synthetic_target_frame(raw, origin, 2).drop(columns=["PromoActive"])
    out = add_features(frame, raw, origin)
    assert not any("promo" in c for c in out.columns if c.startswith("g3_"))


# ---------- leakage ----------
def _scramble_after(raw: pd.DataFrame, origin: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = raw.copy()
    after = out["Date"] > origin
    n = int(after.sum())
    out.loc[after, "OrderVolume"] = rng.permutation(out.loc[after, "OrderVolume"].to_numpy()) * 7 + 1
    out.loc[after, "IsOpen"] = rng.integers(0, 2, n)
    out.loc[after, "PromoActive"] = rng.integers(0, 2, n)
    return out


def test_features_unchanged_when_future_data_changes(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin)
    base = add_features(frame, raw, origin)
    changed = add_features(frame, _scramble_after(raw, origin), origin)
    pd.testing.assert_frame_equal(base, changed)


def test_features_unchanged_when_future_rows_removed(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin)
    pd.testing.assert_frame_equal(add_features(frame, raw, origin), add_features(frame, raw[raw["Date"] <= origin], origin))


def test_features_do_change_when_history_changes(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin)
    edited = raw.copy()
    edited.loc[edited["Date"] == origin - pd.Timedelta(days=1), "IsOpen"] = 0   # history on/before the origin is allowed input
    assert not add_features(frame, raw, origin).equals(add_features(frame, edited, origin))


def test_multi_origin_equals_single_and_is_leak_free(raw) -> None:
    o1, o2 = pd.Timestamp("2014-02-01"), pd.Timestamp("2014-04-20")
    frame = pd.concat([synthetic_target_frame(raw, o, 5).assign(_origin=o) for o in (o1, o2)], ignore_index=True)
    multi = add_features_multi(frame, raw, "_origin")
    for o in (o1, o2):
        part = frame[frame["_origin"] == o].drop(columns="_origin")
        single = add_features(part, raw, o)
        pd.testing.assert_frame_equal(multi[multi["_origin"] == o].drop(columns="_origin"), single)
    # data after the EARLIER origin must not change that origin's rows
    scrambled = add_features_multi(frame, _scramble_after(raw, o1), "_origin")
    pd.testing.assert_frame_equal(multi[multi["_origin"] == o1], scrambled[scrambled["_origin"] == o1])


def test_multi_accepts_prebuilt_panel_and_keeps_index(raw) -> None:
    o = pd.Timestamp("2014-03-01")
    frame = synthetic_target_frame(raw, o, 3).assign(_origin=o)
    frame.index = frame.index + 100
    out = add_features_multi(frame, build_panel(raw), "_origin")
    assert (out.index == frame.index).all()


# ---------- duplicate columns ----------
def test_find_duplicate_columns_basic() -> None:
    f = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1.0, 2.0, 3.0, 4.0], "c": [4, 3, 2, 1], "d": [1, 2, 3, 4]})
    assert find_duplicate_columns(f) == {"b": "a", "d": "a"}


def test_duplicates_handle_nan_strings_and_datetimes() -> None:
    d = pd.to_datetime(["2020-01-01", None, "2020-01-03"])
    f = pd.DataFrame({"x": [1.0, np.nan, 3.0], "y": [1.0, np.nan, 3.0], "s": ["a", None, "c"], "t": ["a", None, "c"],
                      "d1": d, "d2": d, "z": [1.0, np.nan, 4.0]})
    assert find_duplicate_columns(f) == {"y": "x", "t": "s", "d2": "d1"}


def test_constant_columns_are_not_reported_as_duplicates() -> None:
    f = pd.DataFrame({"k1": [0, 0, 0], "k2": [0, 0, 0], "v": [1, 2, 3]})
    assert find_duplicate_columns(f) == {}
    assert find_duplicate_columns(f, ignore_constant=False) == {"k2": "k1"}


def test_near_duplicates_are_not_reported() -> None:
    f = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0 + 1e-9]})
    assert find_duplicate_columns(f) == {}


def test_find_duplicates_does_not_mutate_and_respects_subset() -> None:
    f = pd.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 2]})
    snap = f.copy()
    assert find_duplicate_columns(f, columns=["b", "c"]) == {"c": "b"}
    pd.testing.assert_frame_equal(f, snap)


def test_confirm_duplicates_requires_equality_in_every_frame() -> None:
    a = pd.DataFrame({"u": [1, 2, 3], "v": [1, 2, 3]})
    b = pd.DataFrame({"u": [1, 2, 3], "v": [1, 2, 4]})
    pairs = find_duplicate_columns(a)
    assert pairs == {"v": "u"}
    assert confirm_duplicates(pairs, a, a) == pairs
    assert confirm_duplicates(pairs, a, b) == {}


def test_narrow_integer_key_dtypes_join_correctly(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin, 10)
    narrow = frame.astype({"Weekday": "int8", "PromoActive": "int8", "HubID": "int16"})
    pd.testing.assert_frame_equal(add_features(frame, raw, origin).filter(like="g3_"),
                                  add_features(narrow, raw, origin).filter(like="g3_"))
