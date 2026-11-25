"""Leakage tests on the synthetic history: labels dated after the origin never matter; schedule matters only up to origin + 42 days."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import g3c_features as gf
from synth_g3c import synthetic_target_frame

DAY = pd.Timedelta(days=1)


def scramble_labels_after(raw: pd.DataFrame, origin: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = raw.copy()
    after = out["Date"] > origin
    out.loc[after, "OrderVolume"] = rng.permutation(out.loc[after, "OrderVolume"].to_numpy()) * 7 + 1
    return out


def scramble_schedule_after(raw: pd.DataFrame, origin: pd.Timestamp, seed: int = 2) -> pd.DataFrame:
    """Everything dated after origin + 42 days: schedule columns replaced with random values (the visible window stays intact)."""
    rng = np.random.default_rng(seed)
    out = raw.copy()
    after = out["Date"] > origin + 42 * DAY
    n = int(after.sum())
    out.loc[after, "IsOpen"] = rng.integers(0, 2, n)
    out.loc[after, "PromoActive"] = rng.integers(0, 2, n)
    out.loc[after, "SchoolClosureFlag"] = rng.integers(0, 2, n)
    out.loc[after, "RegionalHoliday"] = rng.integers(0, 4, n)
    out.loc[after, "OrderVolume"] = rng.permutation(out.loc[after, "OrderVolume"].to_numpy()) * 3 + 1
    return out


@pytest.mark.parametrize("origin_s", ["2013-11-20", "2014-04-10", "2014-10-02", "2015-02-14"])
def test_features_unchanged_when_labels_after_origin_are_scrambled(raw, origin_s) -> None:
    origin = pd.Timestamp(origin_s)
    frame = synthetic_target_frame(raw, origin)
    base = gf.add_features(frame, raw, origin)
    pd.testing.assert_frame_equal(base, gf.add_features(frame, scramble_labels_after(raw, origin), origin))
    pd.testing.assert_frame_equal(base, gf.add_features(frame, gf.build_calendar(scramble_labels_after(raw, origin)), origin))
    pd.testing.assert_frame_equal(base, gf.add_features(frame, scramble_schedule_after(raw, origin), origin))


def test_labels_inside_the_window_are_ignored_but_history_labels_are_not(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin)
    base = gf.add_features(frame, raw, origin)
    window = (raw["Date"] > origin) & (raw["Date"] <= origin + 42 * DAY)
    assert base.equals(gf.add_features(frame, raw.assign(OrderVolume=np.where(window, 99999.0, raw["OrderVolume"])), origin))
    last_week = (raw["Date"] > origin - 7 * DAY) & (raw["Date"] <= origin) & (raw["HubID"] == 1)
    edited = raw.assign(OrderVolume=np.where(last_week, 5000.0, raw["OrderVolume"]), IsOpen=np.where(last_week, 1, raw["IsOpen"]))
    changed = gf.add_features(frame, edited, origin)
    assert not changed.filter(like="g3c_promo_").equals(base.filter(like="g3c_promo_"))


def test_future_schedule_does_matter_inside_the_window_only(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin)
    base = gf.add_features(frame, raw, origin).set_index(["HubID", "Date"])
    target = origin + 10 * DAY
    edited = raw.assign(RegionalHoliday=np.where((raw["Date"] == target) & (raw["HubID"] == 1), 1, raw["RegionalHoliday"]))
    new = gf.add_features(frame, edited, origin).set_index(["HubID", "Date"])
    assert np.isnan(base.loc[(1, target), "g3c_hh_any_n"]) and new.loc[(1, target), "g3c_hh_any_n"] > 0
    assert new.loc[(1, target + DAY), "g3c_cal_hol_dprev"] == 1
    beyond = origin + 43 * DAY
    edited2 = raw.assign(RegionalHoliday=np.where(raw["Date"] == beyond, 1, raw["RegionalHoliday"]))
    pd.testing.assert_frame_equal(base.reset_index(), gf.add_features(frame, edited2, origin))


def test_multi_origin_equals_single_and_is_leak_free(raw) -> None:
    o1, o2 = pd.Timestamp("2014-02-01"), pd.Timestamp("2014-10-02")
    frame = pd.concat([synthetic_target_frame(raw, o, 12).assign(_origin=o) for o in (o1, o2)], ignore_index=True)
    frame = frame.iloc[np.random.default_rng(0).permutation(len(frame))]
    multi = gf.add_features_multi(frame, raw, "_origin")
    assert (multi.index == frame.index).all()
    for o in (o1, o2):
        part = frame[frame["_origin"] == o].drop(columns="_origin")
        pd.testing.assert_frame_equal(multi[multi["_origin"] == o].drop(columns="_origin"), gf.add_features(part, raw, o))
    scrambled = gf.add_features_multi(frame, scramble_labels_after(raw, o1), "_origin")
    pd.testing.assert_frame_equal(multi[multi["_origin"] == o1], scrambled[scrambled["_origin"] == o1])


def test_multi_rejects_duplicate_index_and_accepts_prebuilt_calendar(raw) -> None:
    o = pd.Timestamp("2014-03-01")
    frame = synthetic_target_frame(raw, o, 3).assign(_origin=o)
    with pytest.raises(ValueError, match="unique"):
        gf.add_features_multi(pd.concat([frame, frame]), raw, "_origin")
    frame.index = frame.index + 100
    out = gf.add_features_multi(frame, gf.build_calendar(raw), "_origin")
    assert (out.index == frame.index).all() and out.filter(like="g3c_").shape[1] == 36
