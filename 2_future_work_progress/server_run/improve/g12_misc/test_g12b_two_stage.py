from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import g12b_two_stage as ts


def _frame(origins=("2014-01-05", "2014-02-02", "2014-03-02"), n_hubs=4, seed=0, with_y=True):
    rng = np.random.default_rng(seed)
    rows = []
    for o in origins:
        for hub in range(1, n_hubs + 1):
            for h in range(1, 43):
                rows.append((hub, pd.Timestamp(o), h))
    f = pd.DataFrame(rows, columns=["HubID", "_origin", "Horizon"])
    f["Date"] = f["_origin"] + pd.to_timedelta(f["Horizon"], unit="D")
    f["Weekday"] = f["Date"].dt.day_of_week + 1
    f["IsOpen"] = (f["Weekday"] != 7).astype(int)
    f["PromoActive"] = (rng.random(len(f)) < 0.3).astype(int)
    f["HubFormat"] = f["HubID"] % 2
    f["x"] = rng.normal(size=len(f))
    if with_y:
        f["ActualOrderVolume"] = np.where(f["IsOpen"] == 1, rng.integers(10, 100, len(f)), 0).astype(float)
    return f


COLS = ["HubID", "Weekday", "IsOpen", "PromoActive", "HubFormat", "x", "Horizon"]


def test_week_of():
    assert ts.week_of(np.array([1, 7, 8, 14, 15, 42])).tolist() == [0, 0, 1, 1, 2, 5]


def test_week_table_aggregates_correctly():
    f = _frame()
    t = ts.week_table(f, COLS)
    assert len(t) == 4 * 3 * 6
    row = t[(t.HubID == 2) & (t._origin == pd.Timestamp("2014-02-02")) & (t._week == 2)].iloc[0]
    src = f[(f.HubID == 2) & (f._origin == pd.Timestamp("2014-02-02")) & ts.week_of(f.Horizon).__eq__(2)]
    assert len(src) == 7
    assert row["_total"] == src["ActualOrderVolume"].sum() and row["n_open_days"] == src["IsOpen"].sum()
    assert row["x"] == pytest.approx(src["x"].mean()) and row["PromoActive_days"] == src["PromoActive"].sum()
    assert row["HubFormat"] == src["HubFormat"].iloc[0] and row["week_index"] == 2
    assert "Weekday" not in t.columns and "Horizon" not in t.columns
    assert "_total" not in ts.stage1_feature_names(t) and "HubID" in ts.stage1_feature_names(t)


def test_week_table_pure_and_no_target_for_valid_frames():
    f = _frame()
    before = f.copy()
    ts.week_table(f, COLS)
    pd.testing.assert_frame_equal(f, before)
    assert "_total" not in ts.week_table(f.drop(columns="ActualOrderVolume"), COLS).columns


def test_blocked_splits_purge_overlapping_windows():
    origins = pd.date_range("2013-06-30", periods=37, freq="14D")
    splits = ts.blocked_splits(origins.to_numpy(), 6)
    assert len(splits) == 6 and sum(len(h) for h, _ in splits) == 37
    for held, train in splits:
        assert len(train) > 0
        d = np.abs((train[:, None] - held[None, :]) / np.timedelta64(1, "D"))
        assert d.min() >= 42 and not np.isin(train, held).any()
    # the purge removes exactly the origins whose 42-day windows overlap the block (2 on each side for 14-day spacing: gaps of 14 and 28 days)
    held, train = splits[2]
    assert len(train) == 37 - len(held) - 4


def test_crossfit_never_sees_own_origin_or_neighbours():
    f = _frame(origins=pd.date_range("2013-06-30", periods=9, freq="14D").strftime("%Y-%m-%d").tolist(), n_hubs=2)
    t = ts.week_table(f, COLS)
    seen = []

    def fp(train, pred):
        seen.append((set(train["_origin"]), set(pred["_origin"])))
        return np.full(len(pred), np.log1p(train["_total"]).mean())

    out = ts.crossfit_predict(t, fp, n_blocks=3)
    assert out.notna().all() and len(seen) == 3
    for train_o, pred_o in seen:
        for a in pred_o:
            assert all(abs((a - b).days) >= 42 for b in train_o)


def test_crossfit_prediction_does_not_depend_on_own_labels():
    f = _frame(origins=pd.date_range("2013-06-30", periods=9, freq="14D").strftime("%Y-%m-%d").tolist(), n_hubs=2)
    t = ts.week_table(f, COLS)
    fp = lambda train, pred: np.full(len(pred), np.log1p(train["_total"]).mean())  # noqa: E731
    base = ts.crossfit_predict(t, fp, 3)
    t2 = t.copy()
    t2.loc[t2["_origin"] == t2["_origin"].min(), "_total"] *= 1000.0
    changed = ts.crossfit_predict(t2, fp, 3)
    first = t["_origin"] == t["_origin"].min()
    pd.testing.assert_series_equal(base[first], changed[first])


def test_attach_columns_and_alignment():
    f = _frame(n_hubs=3)
    t = ts.week_table(f, COLS)
    pred = pd.Series(np.log1p(t["_total"].to_numpy()), index=t.index)          # oracle stage 1, to check the join
    out = ts.attach(f, t, pred, full=True)
    assert len(out) == len(f) and (out.index == f.index).all()
    expected = np.log1p(f.groupby(["HubID", "_origin", ts.week_of(f.Horizon)])["ActualOrderVolume"].transform("sum"))
    np.testing.assert_allclose(out["g12b_s1_week"].to_numpy(), expected.to_numpy(), rtol=1e-5)
    assert {"g12b_s1_per_open", "g12b_s1_w1", "g12b_s1_w6"} <= set(out.columns)
    row = out[(out.HubID == 1) & (out._origin == pd.Timestamp("2014-01-05"))].iloc[0]
    assert row["g12b_s1_w1"] == pytest.approx(out[(out.HubID == 1) & (out._origin == row["_origin"]) & (out.Horizon == 1)]["g12b_s1_week"].iloc[0])
    small = ts.attach(f, t, pred, full=False)
    assert "g12b_s1_w1" not in small.columns and "g12b_s1_per_open" in small.columns
    with pytest.raises(ValueError):
        ts.attach(out, t, pred, full=False)


def test_build_stage1_hold_and_valid_use_one_model_trained_on_train_only():
    train = _frame(origins=pd.date_range("2013-06-30", periods=9, freq="14D").strftime("%Y-%m-%d").tolist(), n_hubs=2)
    hold = _frame(origins=("2014-01-05",), n_hubs=2, seed=3)
    valid = _frame(origins=("2014-03-02",), n_hubs=2, seed=4, with_y=False).drop(columns=["_origin"])
    calls = []

    def fp(tr, pr):
        calls.append((len(tr), len(pr)))
        return np.full(len(pr), 7.0)

    out = ts.build_stage1(train, hold, valid, pd.Timestamp("2014-03-02"), COLS, fp, n_blocks=3)
    assert len(calls) == 4 and calls[-1][1] == 2 * 6 * 2                       # three cross-fit fits + the final fit on hold+valid weeks
    assert out["valid"][0]["_origin"].eq(pd.Timestamp("2014-03-02")).all() and len(out["valid"][0]) == 12
    assert out["train"][1].notna().all() and (out["hold"][1] == 7.0).all() and (out["valid"][1] == 7.0).all()
    assert "_total" not in out["valid"][0].columns
