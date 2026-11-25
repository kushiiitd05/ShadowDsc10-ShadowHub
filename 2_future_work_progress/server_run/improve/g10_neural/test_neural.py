"""Tests for the neural pilot helpers: no-leak history window, forecast alignment, closed-day zeroing, blend evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from neural_eval import blend_log, evaluate, gate_verdict  # noqa: E402
from neural_prep import (build_future_frame, build_panel, build_static, calendar_features, forecast_to_frame,  # noqa: E402
                         futr_columns, history_window, volume_from_log)

ORIGIN = pd.Timestamp("2015-01-10")
H = 5


def make_base(n_days: int = 20, hubs=(1, 2, 3), start="2015-01-01", drop=()) -> pd.DataFrame:
    rows = []
    for hub in hubs:
        for k, d in enumerate(pd.date_range(start, periods=n_days)):
            if (hub, d) in drop:
                continue
            rows.append({"HubID": hub, "Date": d, "OrderVolume": float(100 * hub + k), "IsOpen": int(d.day_of_week != 6),
                         "PromoActive": k % 2, "RegionalHoliday": 0, "SchoolClosureFlag": int(k % 5 == 0),
                         "HubFormat": hub % 2, "AssortmentTier": hub % 3, "CompetitorDistanceLog1p": 5.0 + hub, "LoyaltyProgram": hub % 2})
    return pd.DataFrame(rows)


def make_valid(hubs=(1, 2, 3)) -> pd.DataFrame:
    rows = []
    for hub in hubs:
        for k, d in enumerate(pd.date_range(ORIGIN + pd.Timedelta(days=1), periods=H)):
            rows.append({"HubID": hub, "Date": d, "Horizon": k + 1, "IsOpen": int(d.day_of_week != 6), "PromoActive": 1, "RegionalHoliday": 0,
                         "SchoolClosureFlag": 0, "ActualOrderVolume": 50.0 * hub, "HubFormat": hub % 2, "AssortmentTier": hub % 3,
                         "CompetitorDistanceLog1p": 5.0 + hub, "LoyaltyProgram": hub % 2})
    return pd.DataFrame(rows)


# ---- no-leak history window ------------------------------------------------
def test_history_window_drops_everything_after_origin():
    out = history_window(make_base(), ORIGIN)
    assert pd.to_datetime(out["Date"]).max() == ORIGIN
    assert (pd.to_datetime(out["Date"]) <= ORIGIN).all()


def test_history_window_empty_raises():
    with pytest.raises(ValueError):
        history_window(make_base(), pd.Timestamp("2014-01-01"))


def test_panel_ends_at_origin_and_ignores_future_values():
    base = make_base()
    poisoned = base.assign(OrderVolume=np.where(pd.to_datetime(base["Date"]) > ORIGIN, 1e6, base["OrderVolume"]),
                           PromoActive=np.where(pd.to_datetime(base["Date"]) > ORIGIN, 1, base["PromoActive"]))
    a, b = build_panel(base, ORIGIN), build_panel(poisoned, ORIGIN)
    assert a["ds"].max() == ORIGIN
    pd.testing.assert_frame_equal(a, b)  # values after the origin cannot influence the panel
    assert a["y"].max() < np.log1p(1e4)


def test_panel_is_complete_daily_grid_with_missing_days_as_closed():
    gap = pd.Timestamp("2015-01-04")
    panel = build_panel(make_base(drop={(2, gap)}), ORIGIN)
    assert panel.groupby("unique_id").size().nunique() == 1
    row = panel[(panel["unique_id"] == "2") & (panel["ds"] == gap)].iloc[0]
    assert row["y"] == 0 and row["open"] == 0
    assert panel[["y", *futr_columns()]].notna().all().all()


def test_panel_target_is_log1p_and_does_not_mutate_input():
    base = make_base()
    before = base.copy()
    panel = build_panel(base, ORIGIN)
    pd.testing.assert_frame_equal(base, before)
    r = panel[(panel["unique_id"] == "1") & (panel["ds"] == pd.Timestamp("2015-01-03"))].iloc[0]
    assert r["y"] == pytest.approx(np.log1p(102.0), rel=1e-5)


# ---- future frame / alignment ----------------------------------------------
def test_future_frame_covers_exactly_the_forecast_days():
    fut = build_future_frame(make_valid(), ORIGIN, H)
    assert len(fut) == 3 * H
    assert fut["ds"].min() == ORIGIN + pd.Timedelta(days=1) and fut["ds"].max() == ORIGIN + pd.Timedelta(days=H)
    assert set(futr_columns()) <= set(fut.columns)


def test_future_frame_rejects_wrong_window_or_duplicates():
    with pytest.raises(ValueError):
        build_future_frame(make_valid(), ORIGIN - pd.Timedelta(days=1), H)
    dup = pd.concat([make_valid(), make_valid().iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError):
        build_future_frame(dup, ORIGIN, H)


def test_panel_and_future_columns_are_identical_sets():
    panel, fut = build_panel(make_base(), ORIGIN), build_future_frame(make_valid(), ORIGIN, H)
    assert [c for c in panel.columns if c != "y"] == list(fut.columns)


def test_static_frame_one_row_per_hub_sorted():
    st = build_static(make_valid())
    assert st["unique_id"].tolist() == ["1", "2", "3"] and st.notna().all().all()


def make_forecast(valid: pd.DataFrame, fn) -> pd.DataFrame:
    fc = pd.DataFrame({"unique_id": valid["HubID"].astype(str), "ds": pd.to_datetime(valid["Date"])})
    fc["m"] = [fn(h, d) for h, d in zip(valid["HubID"], fc["ds"])]
    return fc


def test_forecast_to_frame_aligns_by_key_not_position():
    valid = make_valid()
    fc = make_forecast(valid, lambda h, d: np.log1p(10.0 * h + d.day))
    shuffled = fc.sample(frac=1.0, random_state=3).reset_index(drop=True)
    out = forecast_to_frame(shuffled, valid, "m")
    open_rows = out["IsOpen"] == 1
    expected = np.array([10.0 * h + d.day for h, d in zip(out["HubID"], out["Date"])])
    np.testing.assert_allclose(out.loc[open_rows, "pred"], expected[open_rows], rtol=1e-6)
    assert list(out["HubID"]) == list(valid["HubID"])  # fold row order preserved


def test_forecast_to_frame_missing_or_duplicate_raises():
    valid = make_valid()
    fc = make_forecast(valid, lambda h, d: 1.0)
    with pytest.raises(ValueError):
        forecast_to_frame(fc.iloc[1:], valid, "m")
    with pytest.raises(ValueError):
        forecast_to_frame(pd.concat([fc, fc.iloc[:1]]), valid, "m")


# ---- closed-day zeroing ----------------------------------------------------
def test_volume_from_log_zeroes_closed_clips_and_handles_nan():
    out = volume_from_log(np.array([np.log1p(100.0), np.log1p(100.0), -3.0, np.nan]), np.array([1, 0, 1, 1]))
    np.testing.assert_allclose(out, [100.0, 0.0, 0.0, 0.0], atol=1e-9)


def test_forecast_to_frame_forces_zero_on_closed_days():
    valid = make_valid()
    out = forecast_to_frame(make_forecast(valid, lambda h, d: 5.0), valid, "m")
    assert (out.loc[out["IsOpen"] == 0, "pred"] == 0).all() and (out.loc[out["IsOpen"] == 1, "pred"] > 0).all()
    assert (out["pred"] >= 0).all()


def test_calendar_features_known_values():
    cal = calendar_features(pd.Series(pd.to_datetime(["2015-01-04", "2015-01-31"])))  # a Sunday; a month end
    assert cal.loc[0, "dow6"] == 1 and cal.loc[0, "dow0"] == 0
    assert cal.loc[1, "month_end"] == 1 and cal.loc[0, "month_end"] == 0


# ---- blend evaluation ------------------------------------------------------
def make_oof(fold_scale: dict[int, float], member_noise: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    for fold in fold_scale:
        for hub in range(30):
            for day in range(5):
                actual = float(rng.integers(50, 500))
                is_open = 0 if (hub, day) == (0, 0) else 1
                zero = (hub, day) in ((0, 0), (1, 1))  # a closed zero row and an open-but-zero row
                rows.append({"fold": fold, "HubID": hub, "Date": f"2015-02-{day + 1:02d}", "IsOpen": is_open,
                             "ActualOrderVolume": 0.0 if zero else actual})
    base = pd.DataFrame(rows)
    scale = base["fold"].map(fold_scale).to_numpy()
    log_a = np.log1p(base["ActualOrderVolume"].to_numpy())
    champ = base.assign(pred=np.where(base["IsOpen"] == 1, np.expm1(log_a + rng.normal(0, scale)), 0.0))
    member = base.assign(pred=np.where(base["IsOpen"] == 1, np.expm1(log_a + rng.normal(0, member_noise, len(base))), 0.0))
    return champ, member


def test_blend_log_endpoints_and_zero():
    c, m = np.array([100.0, 0.0]), np.array([200.0, 0.0])
    np.testing.assert_allclose(blend_log(c, m, 0.0), c, atol=1e-9)
    np.testing.assert_allclose(blend_log(c, m, 1.0), m, atol=1e-9)
    assert blend_log(c, m, 0.3)[1] == 0.0


def test_evaluate_uses_identical_rows_and_excludes_open_zero():
    champ, member = make_oof({2: 0.1, 3: 0.1, 4: 0.1}, 0.1)
    member = member.iloc[5:].reset_index(drop=True)  # member misses some rows -> evaluation must use the intersection
    table = evaluate(champ, member)
    n_all = table[(table["rows"] == "all") & (table["fold"] == 2)]["n"].iloc[0]
    n_ex = table[(table["rows"] == "excl_open0") & (table["fold"] == 2)]["n"].iloc[0]
    assert n_ex < n_all
    assert table[table["rows"] == "all"]["n"].sum() == len(member)


def test_independent_equal_member_gives_gain_and_gate_passes():
    champ, member = make_oof({2: 0.2, 3: 0.2, 4: 0.2}, 0.2)
    table = evaluate(champ, member)
    v = gate_verdict(table, [2, 3, 4])
    assert v["mean_gain"] > 0.0024 and v["passed"] and v["folds_improved"] == 3


def test_useless_member_fails_gate():
    champ, member = make_oof({2: 0.05, 3: 0.05, 4: 0.05}, 0.8)
    v = gate_verdict(evaluate(champ, member), [2, 3, 4])
    assert not v["passed"]


def test_gate_requires_two_folds():
    table = pd.DataFrame([{"fold": f, "rows": "all", **{f"gain{w}": g for w in (0.1, 0.2, 0.3)}} for f, g in ((2, 0.01), (3, -0.001), (4, -0.001))])
    v = gate_verdict(table, [2, 3, 4])
    assert v["mean_gain"] > 0.0024 - 1 and not v["passed"] and v["folds_improved"] == 1


# ---- runner plumbing (mock project layout, no neural library needed) ---------
def test_load_fold_history_stops_at_origin_and_n_hubs_subsets(tmp_path):
    from neural_runner import load_fold

    proc = tmp_path / "data/processed"
    (proc / "folds").mkdir(parents=True)
    make_base().to_csv(proc / "base_train.csv.gz", index=False)
    make_valid().assign(ForecastOrigin=ORIGIN).to_csv(proc / "folds/fold_9_2015-01-10.csv.gz", index=False)
    hist, valid, origin = load_fold(tmp_path, 9, None)
    assert origin == ORIGIN and hist["Date"].max() == ORIGIN and hist["Date"].max() < valid["Date"].min()
    hist2, valid2, _ = load_fold(tmp_path, 9, 2)
    assert set(hist2["HubID"]) == set(valid2["HubID"]) == {1, 2}


def test_run_fold_end_to_end_with_stub_model(tmp_path, monkeypatch):
    import neural_runner

    proc = tmp_path / "data/processed"
    (proc / "folds").mkdir(parents=True)
    make_base().to_csv(proc / "base_train.csv.gz", index=False)
    make_valid().assign(ForecastOrigin=ORIGIN).to_csv(proc / "folds/fold_9_2015-01-10.csv.gz", index=False)

    def stub(name, hist, valid, origin, args):  # perfect-knowledge-free stub: predicts log1p of the hub's last observed volume
        last = hist.sort_values("Date").groupby("HubID")["OrderVolume"].last()
        fut = build_future_frame(valid, origin, 5)
        fut["log_pred"] = np.log1p(fut["unique_id"].astype(int).map(last).to_numpy())
        return fut[["unique_id", "ds", "log_pred"]]

    monkeypatch.setattr(neural_runner, "predict_member", stub)
    monkeypatch.setattr(neural_runner, "HORIZON", 5)
    args = type("A", (), {"model": "stub", "n_hubs": None})()
    oof, scores = neural_runner.run_fold(tmp_path, 9, args)
    assert len(oof) == 15 and list(oof.columns) == neural_runner.OOF_COLUMNS
    assert (oof.loc[oof["IsOpen"] == 0, "pred"] == 0).all() and np.isfinite(scores["rmsle_all"])


# ---- closed-day imputation ---------------------------------------------------
def test_impute_closed_interpolates_history_only_and_keeps_open_days():
    from neural_prep import impute_closed

    panel = build_panel(make_base(), ORIGIN)
    out = impute_closed(panel)
    assert len(out) == len(panel) and (out["ds"] == panel["ds"]).all() and (out["unique_id"] == panel["unique_id"]).all()
    open_rows = panel["open"] == 1
    np.testing.assert_allclose(out.loc[open_rows, "y"], panel.loc[open_rows, "y"])  # open days untouched
    assert (out.loc[~open_rows, "y"] > 0).all() and out["ds"].max() == ORIGIN and out["y"].notna().all()


def test_impute_closed_ignores_values_after_origin():
    from neural_prep import impute_closed

    base = make_base()
    poisoned = base.assign(OrderVolume=np.where(pd.to_datetime(base["Date"]) > ORIGIN, 1e6, base["OrderVolume"]))
    pd.testing.assert_frame_equal(impute_closed(build_panel(base, ORIGIN)), impute_closed(build_panel(poisoned, ORIGIN)))


def test_impute_closed_exact_linear_value_and_all_closed_hub():
    from neural_prep import impute_closed

    base = make_base(hubs=(1, 2))
    base.loc[(base["HubID"] == 2), "IsOpen"] = 0  # hub 2 never open
    panel = impute_closed(build_panel(base, ORIGIN))
    sunday = panel[(panel["unique_id"] == "1") & (panel["ds"] == pd.Timestamp("2015-01-04"))].iloc[0]  # a Sunday, between Sat/Mon
    sat, mon = (np.log1p(100.0 + k) for k in (2, 4))  # 2015-01-03 is index 2, 2015-01-05 index 4
    assert sunday["y"] == pytest.approx((sat + mon) / 2, rel=1e-5)
    assert (panel.loc[panel["unique_id"] == "2", "y"] == 0).all()


# ---- TabPFN-TS frame layout ----------------------------------------------------
def test_tabpfn_frames_context_ends_at_origin_and_future_is_after():
    from tabpfn_members import COVARIATES, parse_tabpfn_output, to_tabpfn_frames

    panel, fut = build_panel(make_base(), ORIGIN), build_future_frame(make_valid(), ORIGIN, H)
    ctx, fu = to_tabpfn_frames(panel, fut, 7)
    assert ctx["timestamp"].max() == ORIGIN and fu["timestamp"].min() > ORIGIN
    assert ctx.groupby("item_id").size().eq(7).all() and fu.groupby("item_id").size().eq(H).all()
    assert list(ctx.columns) == ["item_id", "timestamp", "target", *COVARIATES] and "target" not in fu.columns
    out = parse_tabpfn_output(fu.assign(target=1.5).set_index(["item_id", "timestamp"]))
    assert list(out.columns) == ["unique_id", "ds", "log_pred"] and (out["log_pred"] == 1.5).all()


def test_static_frame_fills_missing_metadata_with_indicator():
    valid = make_valid()
    valid.loc[valid["HubID"] == 2, "CompetitorDistanceLog1p"] = np.nan
    st = build_static(valid)
    assert st.notna().all().all()
    assert st["CompetitorDistanceLog1p_missing"].tolist() == [0.0, 1.0, 0.0]
