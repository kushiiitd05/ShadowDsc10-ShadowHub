"""Opt-in checks on the real raw files (skipped when orders_train.csv is not on this machine)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import g3c_features as gf
import g3c_runner as g3c

CANDIDATES = [Path("/Users/bournesmartasfuck_kush/Desktop/Dsc/ShadowHub-repo/1_midterm/data/raw")]
RAW_DIR = next((p for p in CANDIDATES if (p / "orders_train.csv").exists()), None)
pytestmark = pytest.mark.skipif(RAW_DIR is None, reason="real raw files not available")

DAY = pd.Timedelta(days=1)
ORIGINS = [pd.Timestamp("2014-10-02"), pd.Timestamp("2015-03-27")]
EASTER_2014, EASTER_2015 = ["2014-04-18", "2014-04-21"], ["2015-04-03", "2015-04-06"]


@pytest.fixture(scope="module")
def history() -> pd.DataFrame:
    return g3c.load_history(Path("/nonexistent"), RAW_DIR)


@pytest.fixture(scope="module")
def train(history) -> pd.DataFrame:
    return history[history["OrderVolume"].notna()]


@pytest.fixture(scope="module")
def cal(history) -> gf.Calendar:
    return gf.build_calendar(history)


def _target(cal: gf.Calendar, origin: pd.Timestamp, days: int = 42) -> pd.DataFrame:
    dates = pd.date_range(origin + DAY, periods=days)
    return pd.MultiIndex.from_product([cal.panel.hubs, dates], names=["HubID", "Date"]).to_frame(index=False)


def _scramble_labels(history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    out = history.copy()
    after = out["Date"] > origin
    out.loc[after, "OrderVolume"] = rng.permutation(out.loc[after, "OrderVolume"].to_numpy()) * 7 + 1
    return out


def _scramble_far_schedule(history: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    out = history.copy()
    far = out["Date"] > origin + 42 * DAY
    n = int(far.sum())
    for col, high in (("IsOpen", 2), ("PromoActive", 2), ("SchoolClosureFlag", 2), ("RegionalHoliday", 4)):
        out.loc[far, col] = rng.integers(0, high, n)
    return out


# ---------------------------------------------------------------- what the codes look like
def test_regional_holiday_codes_in_the_real_file(history, train) -> None:
    assert set(history["RegionalHoliday"].unique()) == {0, 1, 2, 3}
    easter = pd.to_datetime(["2013-03-29", "2013-04-01", *EASTER_2014, *EASTER_2015])
    by_code = {c: set(train.loc[train["RegionalHoliday"] == c, "Date"]) for c in (1, 2, 3)}
    assert by_code[2] == set(easter)
    assert by_code[3] == set(pd.to_datetime(["2013-12-25", "2013-12-26", "2014-12-25", "2014-12-26"]))
    assert len(by_code[1]) >= 25
    future = history[history["OrderVolume"].isna()]
    assert len(future) == 46830 and (future["RegionalHoliday"] == 0).all()           # the real test window has no holiday at all
    per_date = train[train["RegionalHoliday"] > 0].groupby("Date")["HubID"].count() / 1115
    assert (per_date > 0.99).sum() >= 20 and (per_date < 0.9).sum() >= 8            # mostly national dates, some regional ones


# ---------------------------------------------------------------- leakage
@pytest.mark.parametrize("origin", ORIGINS)
def test_real_data_scramble_after_origin_is_leak_free(history, cal, origin) -> None:
    frame = _target(cal, origin)
    t0 = time.perf_counter()
    base = gf.add_features(frame, cal, origin)
    assert time.perf_counter() - t0 < 30
    assert base.filter(like="g3c_").shape[1] == 36
    labels = _scramble_labels(history, origin)
    pd.testing.assert_frame_equal(base, gf.add_features(frame, labels, origin))
    pd.testing.assert_frame_equal(base, gf.add_features(frame, gf.build_calendar(labels), origin))
    pd.testing.assert_frame_equal(base, gf.add_features(frame, _scramble_far_schedule(history, origin), origin))
    pd.testing.assert_frame_equal(base, gf.add_features(frame, history[history["Date"] <= origin + 42 * DAY], origin))


@pytest.mark.parametrize("origin", ORIGINS)
def test_real_data_features_do_react_to_history_and_to_window_schedule(history, cal, origin) -> None:
    frame = _target(cal, origin)
    base = gf.add_features(frame, cal, origin)
    window = history["Date"].between(origin + DAY, origin + 42 * DAY)
    shifted = history.assign(PromoActive=np.where(window, 1 - history["PromoActive"], history["PromoActive"]))
    assert not gf.add_features(frame, shifted, origin).filter(like="g3c_cal_promo").equals(base.filter(like="g3c_cal_promo"))
    old = history["Date"].between(origin - 60 * DAY, origin) & (history["HubID"] == 1)
    edited = history.assign(OrderVolume=np.where(old, history["OrderVolume"] * 2 + 1, history["OrderVolume"]),
                            IsOpen=np.where(old, 1, history["IsOpen"]))
    assert not gf.add_features(frame, edited, origin).filter(like="g3c_promo").equals(base.filter(like="g3c_promo"))


# ---------------------------------------------------------------- Easter 2015 <- Easter 2014
def test_easter_2015_gets_history_from_easter_2014(train, cal) -> None:
    origin = pd.Timestamp("2015-03-27")
    out = gf.add_features(_target(cal, origin, 42), cal, origin, ("holiday",)).set_index(["HubID", "Date"])
    log = lambda frame: np.log1p(frame["OrderVolume"])  # noqa: E731
    easter = train[(train["RegionalHoliday"] == 2) & (train["Date"] <= origin) & (train["IsOpen"] == 1) & (train["OrderVolume"] > 0)]
    checked = 0
    for date in EASTER_2015:
        rows = out.xs(pd.Timestamp(date), level="Date")
        assert (train.loc[train["Date"] == date, "RegionalHoliday"] == 2).all()
        for hub, row in rows.iterrows():
            hub_easter = easter[easter["HubID"] == hub].sort_values("Date")
            assert row["g3c_hh_type_n"] == len(hub_easter)
            if len(hub_easter):
                checked += 1
                assert row["g3c_hh_type_mean"] == pytest.approx(log(hub_easter).mean(), abs=1e-4)
                assert row["g3c_hh_type_last_log"] == pytest.approx(log(hub_easter).iloc[-1], abs=1e-4)
                assert row["g3c_hh_type_last_days"] == (origin - hub_easter["Date"].iloc[-1]).days
                assert hub_easter["Date"].iloc[-1] in pd.to_datetime(EASTER_2014)         # most recent same-type holiday = Easter 2014
        assert rows["g3c_hh_type_n"].gt(0).sum() >= 8
    assert checked >= 16
    for date in ("2015-04-04", "2015-04-05"):                                           # Easter Saturday/Sunday carry code 0
        assert out.xs(pd.Timestamp(date), level="Date").filter(like="g3c_hh_").isna().all().all()


def test_without_easter_2014_history_the_easter_2015_features_lose_it(history, cal) -> None:
    origin = pd.Timestamp("2015-03-27")
    frame = _target(cal, origin)
    frame = frame[frame["Date"].isin(pd.to_datetime(EASTER_2015))]
    full = gf.add_features(frame, cal, origin, ("holiday",))
    without = history[~history["Date"].isin(pd.to_datetime(EASTER_2014))]
    cut = gf.add_features(frame, without, origin, ("holiday",))
    assert (cut["g3c_hh_type_n"] <= full["g3c_hh_type_n"]).all() and (cut["g3c_hh_type_n"] < full["g3c_hh_type_n"]).any()
    assert cut["g3c_hh_type_last_days"].dropna().gt(350).all()                             # only Easter 2013 is left


def test_easter_calendar_covers_the_weekend_between(cal) -> None:
    origin = pd.Timestamp("2015-03-27")
    out = gf.add_features(_target(cal, origin), cal, origin, ("calendar",)).set_index(["HubID", "Date"])
    sunday = out.xs(pd.Timestamp("2015-04-05"), level="Date")
    assert (sunday["g3c_cal_hol_dprev"] == 2).all() and (sunday["g3c_cal_hol_dnext"] == 1).all() and (sunday["g3c_cal_hol_week"] == 1).all()


# ---------------------------------------------------------------- runtime / schema
def test_real_data_37_origins_run_in_minutes(cal) -> None:
    origins = pd.date_range(end=ORIGINS[0], periods=37, freq="7D")
    stack = pd.concat([_target(cal, o).assign(_origin=o) for o in origins], ignore_index=True)
    t0 = time.perf_counter()
    out = gf.add_features_multi(stack, cal, "_origin")
    elapsed = time.perf_counter() - t0
    print(f"37 origins x {len(stack) // 37} rows: {elapsed:.1f}s")
    assert elapsed < 180 and len(out) == len(stack)
    assert out.filter(like="g3c_").dtypes.eq(np.float32).all()


def test_g3b_on_the_extended_calendar_panel_equals_g3b_on_the_train_panel(history, cal) -> None:
    import g3b_features as g3b_feat
    from feature_panel import build_panel
    origin = ORIGINS[1]
    frame = _target(cal, origin, 10)
    a = g3b_feat.add_features(frame, cal.panel, origin)
    b = g3b_feat.add_features(frame, build_panel(history[history["OrderVolume"].notna()]), origin)
    pd.testing.assert_frame_equal(a, b)
