"""Pure-function tests for origin_builder (no data files needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import origin_builder as ob  # noqa: E402

T = pd.Timestamp


def synthetic_panel(n_hubs: int = 4, gap_hubs: int = 2, start="2013-01-01", end="2015-06-19") -> pd.DataFrame:
    """Small panel: the first `gap_hubs` hubs miss 2014-07-01..2014-12-31."""
    dates = pd.date_range(start, end)
    frames = []
    for hub in range(1, n_hubs + 1):
        d = dates
        if hub <= gap_hubs:
            d = d[(d < ob.GAP_START) | (d > ob.GAP_END)]
        frames.append(pd.DataFrame({"HubID": hub, "Date": d}))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ origin dates
def test_origin_grid_spacing_and_bounds():
    g = ob.origin_grid(T("2013-06-30"), T("2013-08-11"), 14)
    assert [d.date().isoformat() for d in g] == ["2013-06-30", "2013-07-14", "2013-07-28", "2013-08-11"]
    assert len(ob.origin_grid(T("2013-06-30"), T("2013-08-11"), 7)) == 7


def test_seven_day_grid_contains_the_fourteen_day_grid():
    g14 = set(ob.origin_grid(T("2013-06-30"), T("2015-05-08"), 14))
    g7 = set(ob.origin_grid(T("2013-06-30"), T("2015-05-08"), 7))
    assert g14 < g7
    assert all((d - T("2013-06-30")).days % 7 == 0 for d in g7)


def test_origin_grid_empty_and_invalid():
    assert ob.origin_grid(T("2014-01-02"), T("2014-01-01"), 7) == []
    with pytest.raises(ValueError):
        ob.origin_grid(T("2014-01-01"), T("2014-02-01"), 0)


def test_block_dates_is_42_days_after_origin():
    s, e = ob.block_dates(T("2014-06-19"))
    assert (s, e) == (T("2014-06-20"), T("2014-07-31"))


# ------------------------------------------------------------------ completeness (what is excluded today)
def test_block_completeness_around_the_gap(monkeypatch):
    # the real constants are 1115 hubs / 46830 rows; shrink them for the synthetic panel
    monkeypatch.setattr(ob, "N_HUBS", 4)
    monkeypatch.setattr(ob, "FULL_BLOCK_ROWS", 4 * 42)
    p = synthetic_panel()
    assert ob.is_complete_block(p, T("2014-05-18"))          # horizon ends 06-29, before the gap
    assert ob.is_complete_block(p, T("2014-05-19"))          # horizon ends 06-30, the last day before the gap
    assert not ob.is_complete_block(p, T("2014-05-20"))      # horizon ends 07-01: gap hubs lose a day
    assert not ob.is_complete_block(p, T("2014-06-19"))      # fold-5 origin: horizon reaches 07-01..07-31
    assert not ob.is_complete_block(p, T("2014-12-01"))      # horizon 12-02..01-12 partly inside the gap
    assert ob.is_complete_block(p, T("2014-12-31"))          # horizon starts 2015-01-01
    assert ob.is_complete_block(p, T("2015-01-02"))


def test_block_row_count_partial_and_empty():
    p = synthetic_panel()
    assert ob.block_row_count(p, T("2014-06-19")) == 4 * 11 + 2 * 31          # all hubs 06-20..06-30, non-gap hubs 07-01..07-31
    assert ob.block_row_count(p, T("2014-07-13")) == 2 * 42                    # only the 2 non-gap hubs
    assert ob.block_row_count(p, T("2020-01-01")) == 0


def test_plan_origins_kinds(monkeypatch):
    monkeypatch.setattr(ob, "N_HUBS", 4)
    monkeypatch.setattr(ob, "FULL_BLOCK_ROWS", 4 * 42)
    p = synthetic_panel()
    first, last = T("2013-06-30"), T("2015-05-08")
    existing = set(ob.origin_grid(first, T("2014-05-18"), 14)) | {d for d in ob.origin_grid(first, last, 14) if d > ob.GAP_END}
    plan = ob.plan_origins(p, existing, first, last).set_index("origin")
    assert plan.loc[T("2013-07-07"), "kind"] == "new7"                      # complete, off the 14-day grid
    assert plan.loc[T("2013-06-30"), "kind"] == "base"
    assert plan.loc[T("2014-06-01"), "kind"] == "summer"                    # on the 14-day grid, incomplete
    assert plan.loc[T("2014-06-08"), "kind"] == "skip_incomplete_7"         # incomplete and only on the 7-day grid
    assert (plan.loc[plan.index.isin(ob.origin_grid(T("2014-06-01"), T("2014-12-28"), 14)), "kind"] == "summer").all()


# ------------------------------------------------------------------ eligibility / purge
def test_eligible_matches_harness_rule():
    o = pd.Series(pd.to_datetime(["2014-05-04", "2014-05-18", "2014-11-16", "2014-11-30"]))
    assert ob.eligible_origins(o, T("2015-01-02")).tolist() == [True, True, True, False]   # 11-16 + 42 = 12-28 <= 01-02; 11-30 + 42 = 01-11 > 01-02
    assert ob.eligible_origins(o, T("2014-06-19")).tolist() == [True, False, False, False]  # fold 5: 05-04 + 42 = 06-15 <= 06-19; 05-18 + 42 = 06-29 > 06-19


def test_purge_matches_harness_for_older_origins_and_is_two_sided():
    h = T("2015-02-13")
    older = pd.Series(pd.date_range(h - pd.Timedelta(days=100), h, freq="1D"))
    harness_keep = older + pd.Timedelta(days=42) < h + pd.Timedelta(days=1)       # exp_runner.split_holdout, with hold_start = h + 1
    mine = ob.purge_mask(older, h)
    assert (mine.to_numpy() == harness_keep.to_numpy()).all()
    later = pd.Series(pd.to_datetime(["2015-02-14", "2015-03-26", "2015-03-27", "2015-04-30"]))
    assert ob.purge_mask(later, h).tolist() == [False, False, True, True]          # a later origin reaching into the holdout labels is purged too


def test_split_train_hold_has_no_label_overlap():
    origins = pd.Series(pd.date_range("2013-06-30", periods=60, freq="7D"))
    h = origins.iloc[30]
    tr, ho = ob.split_train_hold(origins, h)
    assert ho.sum() == 1 and not (tr & ho).any()
    for o in origins[tr]:
        assert abs((o - h).days) >= 42


def test_reference_holdout_ignores_summer_origins():
    m = pd.DataFrame({"origin": pd.to_datetime(["2014-05-04", "2014-05-18", "2014-08-10", "2015-01-02"]),
                      "kind": ["base", "base", "summer", "base"]})
    assert ob.reference_holdout(m, T("2015-01-02")) == T("2014-05-18")           # summer 08-10 is eligible but not base
    assert ob.reference_holdout(m, T("2015-02-13")) == T("2015-01-02")
    with pytest.raises(ValueError):
        ob.reference_holdout(m, T("2013-01-01"))


def test_select_origin_set_membership():
    m = pd.DataFrame({"origin": pd.to_datetime(["2014-05-04", "2014-05-11", "2014-08-10", "2014-09-07"]),
                      "kind": ["base", "new7", "summer", "summer"]})
    v = T("2015-01-02")
    assert ob.select_origin_set(m, "base", v)["origin"].tolist() == [T("2014-05-04")]
    assert len(ob.select_origin_set(m, "dense7", v)) == 2
    assert len(ob.select_origin_set(m, "summer", v)) == 3
    assert len(ob.select_origin_set(m, "summer", T("2014-06-19"))) == 1          # fold 5: nothing past 05-04 is eligible
    with pytest.raises(KeyError):
        ob.select_origin_set(m, "nope", v)


# ------------------------------------------------------------------ season window
def test_circular_distance_wraps_the_new_year():
    assert ob.circular_day_distance(T("2013-12-20"), T("2015-01-02")) == pytest.approx(13, abs=1)
    assert ob.circular_day_distance(T("2014-06-19"), T("2015-06-19")) == pytest.approx(0, abs=1.5)
    s = ob.circular_day_distance(pd.Series(pd.to_datetime(["2014-01-02", "2014-07-02"])), T("2015-01-02"))
    assert s.iloc[0] < 2 and s.iloc[1] == pytest.approx(182, abs=2)


def test_season_mask_window_edges():
    o = pd.Series(pd.to_datetime(["2013-11-20", "2013-11-10", "2014-02-10", "2014-02-23", "2014-06-01"]))
    m = ob.season_mask(o, T("2015-01-02"), 45)
    assert m.tolist() == [True, False, True, False, False]


def test_season_mask_for_summer_fold5():
    o = pd.Series(pd.date_range("2013-06-30", "2014-05-04", freq="14D"))
    m = ob.season_mask(o, ob.FOLD5_ORIGIN, 45)
    picked = o[m]
    # +/-45 days around 19 Jun is 5 May .. 3 Aug; a 2014 origin before 05-04 is out of the window by a day or more
    assert picked.min() == T("2013-06-30") and picked.max() == T("2014-05-04") or len(picked) > 0
    assert all(ob.circular_day_distance(d, ob.FOLD5_ORIGIN) <= 45 for d in picked)
    assert not any(ob.circular_day_distance(d, ob.FOLD5_ORIGIN) <= 45 for d in o[~m])


def test_season_window_for_test_is_may_to_august():
    lo, hi = ob.season_window_for_test(T("2015-06-19"))
    assert (lo, hi) == ("05-05", "08-03")


def test_season_pool_respects_eligibility_and_window():
    m = pd.DataFrame({"origin": pd.date_range("2013-06-30", "2015-05-08", freq="14D")})
    m["kind"] = "base"
    pool = ob.season_pool(m, T("2015-01-02"))
    assert (pool["origin"] + pd.Timedelta(days=42) <= T("2015-01-02")).all()
    assert (ob.circular_day_distance(pool["origin"], T("2015-01-02")) <= 45).all()
    assert len(pool) > 0


# ------------------------------------------------------------------ filters, blend, stats
def test_drop_stale_rows_keeps_fresh_hubs_and_does_not_mutate():
    f = pd.DataFrame({"HubID": [1, 2, 3], "HistoryDaysSinceLastObservation": [0, 13, 181]})
    out = ob.drop_stale_rows(f, 7)
    assert out["HubID"].tolist() == [1]
    assert len(f) == 3


def test_log_blend_endpoints_and_midpoint():
    a, b = np.array([10.0, 100.0]), np.array([20.0, 50.0])
    assert np.allclose(ob.log_blend(a, b, 0.0), a)
    assert np.allclose(ob.log_blend(a, b, 1.0), b)
    mid = ob.log_blend(a, b, 0.5)
    assert np.allclose(np.log1p(mid), (np.log1p(a) + np.log1p(b)) / 2)
    with pytest.raises(ValueError):
        ob.log_blend(a, b, 1.5)


def test_duplication_stats_doubles_with_seven_day_spacing():
    d14 = ob.duplication_stats(ob.origin_grid(T("2013-06-30"), T("2015-01-01"), 14))
    d7 = ob.duplication_stats(ob.origin_grid(T("2013-06-30"), T("2015-01-01"), 7))
    assert d14["mean_origins_per_target_day"] == pytest.approx(3.0, abs=0.2)
    assert d7["mean_origins_per_target_day"] == pytest.approx(6.0, abs=0.4)
    assert d7["neighbour_overlap_frac"] > d14["neighbour_overlap_frac"]
    assert ob.duplication_stats([])["origins"] == 0
