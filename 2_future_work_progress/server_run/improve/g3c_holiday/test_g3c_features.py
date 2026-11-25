from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import g3c_features as gf
import oracles as oc
from synth_g3c import synthetic_raw, synthetic_target_frame

DAY = pd.Timedelta(days=1)
HH = ["any_mean", "any_median", "any_n", "any_ratio", "type_mean", "type_median", "type_n", "type_ratio", "type_last_log", "type_last_days"]
ORIGINS = ["2014-04-10", "2014-06-10", "2014-12-10", "2015-03-27"]       # windows with Easter, regional holiday, Christmas/New Year, Easter 2015


def _rows(frame: pd.DataFrame, n: int = 150, seed: int = 0) -> pd.DataFrame:
    return frame.iloc[np.random.default_rng(seed).permutation(len(frame))[:n]]


def _assert_close(got: pd.Series, want: float, what: str) -> None:
    if np.isnan(want):
        assert np.isnan(got), what
    else:
        assert got == pytest.approx(want, rel=1e-4, abs=1e-4), what


# ---------------------------------------------------------------- (A) holiday_history
@pytest.mark.parametrize("origin_s", ORIGINS)
def test_holiday_history_matches_bruteforce(raw, origin_s) -> None:
    origin = pd.Timestamp(origin_s)
    frame = synthetic_target_frame(raw, origin)
    out = gf.add_features(frame, raw, origin, ("holiday",))
    holiday_rows = out[out["Date"].map(lambda d: d in set(raw.loc[raw["RegionalHoliday"] > 0, "Date"]))]
    assert len(holiday_rows) > 0
    for _, row in pd.concat([holiday_rows, _rows(out, 60)]).iterrows():
        want = oc.holiday_history(raw, int(row["HubID"]), row["Date"], origin)
        for key in HH:
            _assert_close(row[f"g3c_hh_{key}"], want[key], f"{key} hub {row['HubID']} {row['Date'].date()}")


def test_holiday_history_nan_when_not_a_holiday_and_n_zero_without_history(raw) -> None:
    origin = pd.Timestamp("2014-04-10")                  # Easter 2014 is in the window; code 2 was seen only in 2013
    out = gf.add_features(synthetic_target_frame(raw, origin), raw, origin, ("holiday",))
    code = out.merge(raw[["HubID", "Date", "RegionalHoliday"]], on=["HubID", "Date"])["RegionalHoliday"].to_numpy()
    cols = out.filter(like="g3c_hh_")
    assert cols[code == 0].isna().all().all()
    assert cols[code > 0].shape[0] > 0 and cols.loc[code > 0, "g3c_hh_any_n"].notna().all()
    # hub 1 is open on every holiday -> history exists; hub 5 is closed on national holidays -> n == 0 and NaN statistics
    easter = out[(out["Date"] == "2014-04-18") & (out["HubID"].isin([1, 5]))].set_index("HubID")
    assert easter.loc[1, "g3c_hh_type_n"] == 2 and easter.loc[5, "g3c_hh_type_n"] == 0
    assert np.isnan(easter.loc[5, ["g3c_hh_type_mean", "g3c_hh_type_median", "g3c_hh_type_ratio", "g3c_hh_type_last_log",
                                    "g3c_hh_type_last_days"]].astype(float)).all()


def test_holiday_history_same_type_vs_any_type(raw) -> None:
    origin = pd.Timestamp("2014-12-10")
    out = gf.add_features(synthetic_target_frame(raw, origin), raw, origin, ("holiday",))
    christmas = out[(out["Date"] == "2014-12-25") & (out["HubID"] == 1)].iloc[0]
    assert christmas["g3c_hh_type_n"] == 2                                # 2013-12-25/26 only (code 3)
    assert christmas["g3c_hh_any_n"] > christmas["g3c_hh_type_n"]
    assert christmas["g3c_hh_type_last_days"] == (origin - pd.Timestamp("2013-12-26")).days


def test_holiday_in_the_window_is_not_history(raw) -> None:
    origin = pd.Timestamp("2014-04-10")
    out = gf.add_features(synthetic_target_frame(raw, origin), raw, origin, ("holiday",))
    monday = out[(out["Date"] == "2014-04-21") & (out["HubID"] == 1)].iloc[0]
    assert monday["g3c_hh_type_n"] == 2 and monday["g3c_hh_type_last_days"] == (origin - pd.Timestamp("2013-04-01")).days


# ---------------------------------------------------------------- (B) holiday_calendar
@pytest.mark.parametrize("origin_s", ORIGINS)
def test_calendar_matches_bruteforce(raw, origin_s) -> None:
    origin = pd.Timestamp(origin_s)
    frame = synthetic_target_frame(raw, origin)
    out = gf.add_features(frame, raw, origin, ("calendar",))
    edge = out[out["Horizon"].isin([1, 2, 41, 42])]                    # window edges are where truncation matters
    for _, row in pd.concat([edge.head(40), _rows(out, 120)]).iterrows():
        want = oc.calendar_row(raw, int(row["HubID"]), row["Date"], origin)
        for key, value in want.items():
            _assert_close(row[f"g3c_cal_{key}"], value, f"{key} hub {row['HubID']} {row['Date'].date()}")


def test_calendar_only_looks_inside_the_window(raw) -> None:
    origin = pd.Timestamp("2014-04-10")                                # Easter Monday 2014-04-21; May 1 is 21 days after it
    out = gf.add_features(synthetic_target_frame(raw, origin), raw, origin, ("calendar",)).set_index(["HubID", "Date"])
    row = out.loc[(1, pd.Timestamp("2014-04-18"))]
    assert row["g3c_cal_hol_week"] == 1 and row["g3c_cal_hol_next7"] == 1 and row["g3c_cal_hol_dnext"] == 3
    assert out.loc[(1, pd.Timestamp("2014-04-22")), "g3c_cal_hol_dprev"] == 1
    # last day of the window: the day after is unknown, so it contributes nothing
    last = out.loc[(1, origin + 42 * DAY)]
    assert last["g3c_cal_hol_next7"] == 0 and last["g3c_cal_hol_dnext"] == 14


def test_calendar_closed_run_flags(raw) -> None:
    origin = pd.Timestamp("2014-04-10")
    out = gf.add_features(synthetic_target_frame(raw, origin), raw, origin, ("calendar",)).set_index(["HubID", "Date"])
    # hub 4 is closed on Sundays and national holidays: Easter Friday is closed, Easter Saturday... the Tuesday after Easter Monday is first-after
    tue = out.loc[(4, pd.Timestamp("2014-04-22"))]
    assert tue["g3c_cal_closed_run_before"] >= 1 and tue["g3c_cal_first_after_closed"] in (0.0, 1.0)
    for date in ("2014-04-22",):
        assert out.loc[(4, pd.Timestamp(date)), "g3c_cal_first_after_closed"] == float(
            raw.loc[(raw["HubID"] == 4) & (raw["Date"] == date), "IsOpen"].iloc[0] == 1)
    always_open = out.loc[(1, pd.Timestamp("2014-04-15"))]
    assert always_open["g3c_cal_closed_run_before"] == 0 and always_open["g3c_cal_first_after_closed"] == 0


def test_calendar_missing_hub_rows_take_the_unanimous_holiday_of_the_date(raw) -> None:
    origin = pd.Timestamp("2014-04-10")
    hole = raw[~((raw["HubID"] == 1) & (raw["Date"] == "2014-04-18"))]            # Easter Friday row of hub 1 is missing
    frame = synthetic_target_frame(raw, origin)
    frame = frame[frame["Date"] != "2014-04-18"]
    a = gf.add_features(frame, raw, origin, ("calendar",)).set_index(["HubID", "Date"])
    b = gf.add_features(frame, hole, origin, ("calendar",)).set_index(["HubID", "Date"])
    pd.testing.assert_frame_equal(a.filter(like="hol"), b.filter(like="hol"))


# ---------------------------------------------------------------- (C) sunday_special
def test_sunday_matches_bruteforce_and_is_nan_off_sundays(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin)
    out = gf.add_features(frame, raw, origin, ("sunday",))
    cols = out.filter(like="g3c_sun_")
    assert list(cols.columns) == ["g3c_sun_log_mean", "g3c_sun_n_open", "g3c_sun_open_share"]
    assert cols[out["Date"].dt.day_of_week != 6].isna().all().all()
    sundays = out[out["Date"].dt.day_of_week == 6]
    assert len(sundays) > 20
    for _, row in sundays.iterrows():
        want = oc.sunday_row(raw, int(row["HubID"]), row["Date"], origin)
        for key, value in want.items():
            _assert_close(row[f"g3c_sun_{key}"], value, f"{key} hub {row['HubID']}")


def test_sunday_hub_profiles_are_distinct(raw) -> None:
    origin = pd.Timestamp("2014-06-20")                                  # hubs 3 and 4 have no rows from July on
    out = gf.add_features(synthetic_target_frame(raw, origin), raw, origin, ("sunday",))
    sunday = out[out["Date"] == "2014-06-22"].set_index("HubID")
    assert sunday.loc[1, "g3c_sun_open_share"] > 0.9 and sunday.loc[6, "g3c_sun_open_share"] == 0
    assert np.isnan(sunday.loc[6, "g3c_sun_log_mean"]) and sunday.loc[6, "g3c_sun_n_open"] == 0
    assert 0.2 < sunday.loc[3, "g3c_sun_open_share"] < 0.8


# ---------------------------------------------------------------- (D) promo_hub_ratio
@pytest.mark.parametrize("origin_s", ["2014-04-10", "2014-10-02"])
def test_promo_ratio_matches_bruteforce(raw, origin_s) -> None:
    origin = pd.Timestamp(origin_s)
    frame = synthetic_target_frame(raw, origin, 3)
    out = gf.add_features(frame, raw, origin, ("promo",)).drop_duplicates("HubID").set_index("HubID")
    for hub in out.index:
        for w in (182, 365):
            want = oc.promo_row(raw, hub, origin, w)
            _assert_close(out.loc[hub, f"g3c_promo_ratio{w}"], want["ratio"], f"ratio{w} hub {hub}")
            assert out.loc[hub, f"g3c_promo_n{w}"] == want["n"]


def test_promo_ratio_recovers_the_planted_uplift(raw, origin) -> None:
    out = gf.add_features(synthetic_target_frame(raw, origin, 3), raw, origin, ("promo",)).drop_duplicates("HubID").set_index("HubID")
    ratio = out.loc[[1, 2, 6, 7, 8], "g3c_promo_ratio365"]
    assert ratio.between(np.log(1.3) - 0.05, np.log(1.3) + 0.05).all()          # orders x (1 + 0.3 promo): log ratio ~ 0.26


def test_promo_ratio_nan_when_hub_never_open_on_promo_days(raw, origin) -> None:
    edited = raw.assign(PromoActive=np.where(raw["HubID"] == 5, 0, raw["PromoActive"]))
    out = gf.add_features(synthetic_target_frame(edited, origin, 3), edited, origin, ("promo",)).drop_duplicates("HubID").set_index("HubID")
    assert np.isnan(out.loc[5, "g3c_promo_ratio182"]) and out.loc[5, "g3c_promo_n182"] == 0


# ---------------------------------------------------------------- API
def test_columns_are_prefixed_float32_and_groups_exclusive(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin, 5)
    full = gf.add_features(frame, raw, origin)
    extra = [c for c in full.columns if c not in frame.columns]
    assert all(c.startswith("g3c_") for c in extra) and (full[extra].dtypes == np.float32).all()
    assert len(extra) == len(set(extra)) == 10 + 19 + 3 + 4
    for group in gf.ALL_GROUPS:
        part = gf.add_features(frame, raw, origin, (group,))
        assert [c for c in part.columns if c not in frame.columns] == gf.select_group_columns(extra, (group,))
    assert sorted(sum((gf.select_group_columns(extra, (g,)) for g in gf.ALL_GROUPS), [])) == sorted(extra)
    assert list(frame.columns) == ["HubID", "Date", "Weekday", "Horizon"]               # input untouched


def test_rejects_clash_unknown_group_and_missing_columns(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin, 3)
    with pytest.raises(ValueError, match="already has"):
        gf.add_features(gf.add_features(frame, raw, origin, ("sunday",)), raw, origin, ("sunday",))
    with pytest.raises(ValueError, match="unknown"):
        gf.add_features(frame, raw, origin, ("nope",))
    with pytest.raises(ValueError, match="lacks"):
        gf.add_features(frame.drop(columns="Date"), raw, origin)
    with pytest.raises(ValueError, match="lacks"):
        gf.build_calendar(raw.drop(columns="RegionalHoliday"))


def test_unknown_hub_and_narrow_hub_dtype(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin, 5)
    ghost = pd.concat([frame, frame.head(3).assign(HubID=999)], ignore_index=True)
    out = gf.add_features(ghost, raw, origin)
    assert out.filter(like="g3c_").iloc[-3:].isna().all().all()
    narrow = gf.add_features(frame.astype({"HubID": "int16"}), raw, origin)
    pd.testing.assert_frame_equal(gf.add_features(frame, raw, origin).filter(like="g3c_"), narrow.filter(like="g3c_"))


def test_hub_without_any_history_gets_nan_history_features_but_known_schedule_features() -> None:
    raw = synthetic_raw(n_days=800)
    origin = pd.Timestamp("2014-10-02")
    hist_cut = raw.loc[~((raw["HubID"] == 1) & (raw["Date"] <= origin))]            # hub 1 has schedule but no history yet
    frame = synthetic_target_frame(raw, origin, 10)
    out = gf.add_features(frame, hist_cut, origin)
    one = out[out["HubID"] == 1]
    stats = one.filter(like="g3c_hh_").drop(columns=["g3c_hh_any_n", "g3c_hh_type_n"])
    assert stats.isna().all().all() and one["g3c_promo_ratio365"].isna().all()
    assert (one.loc[one["g3c_hh_any_n"].notna(), ["g3c_hh_any_n", "g3c_hh_type_n"]] == 0).all().all()
    assert one["g3c_cal_promo_week"].notna().all()


def test_calendar_object_and_dataframe_agree(raw, origin) -> None:
    frame = synthetic_target_frame(raw, origin)
    pd.testing.assert_frame_equal(gf.add_features(frame, raw, origin), gf.add_features(frame, gf.build_calendar(raw), origin))
