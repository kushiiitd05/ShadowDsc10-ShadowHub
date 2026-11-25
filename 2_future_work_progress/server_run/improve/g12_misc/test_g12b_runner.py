from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import g12b_runner as rn
import g12b_two_stage as ts
from test_g12b_two_stage import COLS, _frame

ORIGINS = pd.date_range("2013-06-30", periods=9, freq="14D").strftime("%Y-%m-%d").tolist()


def _ctx(fold=3):
    train = _frame(origins=ORIGINS, n_hubs=3)
    hold = _frame(origins=("2014-01-05",), n_hubs=3, seed=2)
    valid = _frame(origins=("2014-03-02",), n_hubs=3, seed=3).drop(columns=["_origin"])
    valid["ForecastOrigin"] = pd.Timestamp("2014-03-02")
    return {"train": train, "hold": hold, "valid": valid, "origin": pd.Timestamp("2014-03-02"), "columns": [c for c in COLS if c != "Horizon"] + ["Horizon"], "fold": fold}


def _stub_fit_predict(*args, **kwargs):
    return lambda train, pred: np.full(len(pred), np.log1p(train["_total"]).mean())


def test_make_fit_predict_runs_lightgbm_on_week_tables():
    pytest.importorskip("lightgbm")
    f = _frame(origins=ORIGINS, n_hubs=6)
    t = ts.week_table(f, COLS)
    fp = rn.make_fit_predict({"HubID", "HubFormat"}, 1, 2, rounds=20, lr=0.2)
    out = fp(t[t["_origin"] < t["_origin"].max()], t[t["_origin"] == t["_origin"].max()])
    assert out.shape == ((t["_origin"] == t["_origin"].max()).sum(),) and np.isfinite(out).all()


def test_augment_ctx_adds_columns_and_keeps_input():
    ctx = _ctx()
    before = {k: ctx[k].copy() for k in ("train", "hold", "valid")}
    stage1 = ts.build_stage1(ctx["train"], ctx["hold"], ctx["valid"], ctx["origin"], COLS, _stub_fit_predict(), n_blocks=3)
    out = rn.augment_ctx(ctx, stage1, full=True)
    for k in ("train", "hold", "valid"):
        pd.testing.assert_frame_equal(ctx[k], before[k])
        assert len(out[k]) == len(ctx[k]) and out[k]["g12b_s1_week"].notna().all() and "g12b_s1_w3" in out[k].columns
    assert "_origin" in out["valid"].columns and "_origin" not in ctx["valid"].columns
    small = rn.augment_ctx(ctx, stage1, full=False)
    assert "g12b_s1_w1" not in small["valid"].columns
    cols = rn.variant_ctx(small, "g12b_ts_min")["columns"]
    assert cols[: len(ctx["columns"])] == ctx["columns"] and set(cols[len(ctx["columns"]):]) == {"g12b_s1_week", "g12b_s1_per_open"}


def test_valid_stage1_ignores_valid_labels():
    ctx = _ctx()
    other = {**ctx, "valid": ctx["valid"].assign(ActualOrderVolume=1e6)}
    fp = _stub_fit_predict()
    a = ts.build_stage1(ctx["train"], ctx["hold"], ctx["valid"], ctx["origin"], COLS, fp, n_blocks=3)
    b = ts.build_stage1(other["train"], other["hold"], other["valid"], other["origin"], COLS, fp, n_blocks=3)
    pd.testing.assert_series_equal(a["valid"][1], b["valid"][1])


def test_stage1_quality_reports_rmse():
    ctx = _ctx()
    stage1 = ts.build_stage1(ctx["train"], ctx["hold"], ctx["valid"], ctx["origin"], COLS, _stub_fit_predict(), n_blocks=3)
    q = rn.stage1_quality(ctx, stage1)
    assert set(q) == {"s1_rmse_hold", "s1_rmse_valid"} and all(v >= 0 for v in q.values())


def test_main_end_to_end_with_stubs(tmp_path: Path, monkeypatch):
    seen: dict = {}

    def fake_fit(ctx, cfg, seed, n_jobs, lr, rounds):
        seen[ctx["variant"]] = list(ctx["columns"])
        oof = ctx["valid"][["HubID", "Date", "Horizon"]].assign(IsOpen=1, ActualOrderVolume=10.0, pred=9.0, fold=ctx["fold"])
        return oof, {"rmsle_all": 0.1, "rmsle_excl_open0": 0.1, "bias_open": 0.0, "fold": ctx["fold"], "best_iteration": 5, "fit_seconds": 0.1}

    monkeypatch.setattr(rn.base, "prepare_fold", lambda root, fold, seed: {**_ctx(fold), "fold": fold})
    monkeypatch.setattr(rn.base, "fit_variant", fake_fit)
    monkeypatch.setattr(rn, "make_fit_predict", _stub_fit_predict)
    out = tmp_path / "o"
    assert rn.main([str(tmp_path), str(out), "--variants", "g12b_ts,g12b_ts_min", "--folds", "3,4", "--seed", "7", "--blocks", "3"]) == 0
    exp = pd.read_csv(out / "experiments.csv")
    assert set(exp["variant"]) == {"g12b_ts", "g12b_ts_min"} and json.loads(exp.loc[0, "config"])["recipe"] == "cs02"
    oof = pd.read_csv(out / "oof_g12b_ts_s7.csv.gz")
    assert set(oof["fold"]) == {3, 4} and {"HubID", "Date", "pred", "ActualOrderVolume"} <= set(oof.columns)
    assert len(seen["g12b_ts"]) == len(seen["g12b_ts_min"]) + 6
    assert (out / "stage1_quality_s7.csv").exists()


def test_unknown_variant_rejected(tmp_path):
    with pytest.raises(SystemExit):
        rn.main([str(tmp_path), str(tmp_path / "o"), "--variants", "nope"])
